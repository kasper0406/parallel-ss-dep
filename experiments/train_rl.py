import argparse
import time
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

from experiments.model import TinyLM
from experiments.layers import DeltaNetAttention
from experiments.thinking import (
    generate_thought_trajectories,
    compute_grpo_advantages,
    mask_token_logit,
    ThoughtTrajectory
)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d_model", type=int, default=576)
    p.add_argument("--n_layers", type=int, default=30)
    p.add_argument("--n_heads", type=int, default=9)
    p.add_argument("--d_head", type=int, default=64)
    p.add_argument("--T", type=int, default=512)
    p.add_argument("--max_T", type=int, default=0,
                   help="Maximum sequence length for position embeddings. 0 disables them.")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--arch", type=str, default="deltanet")
    p.add_argument("--dataset", type=str, default="codeparrot/codeparrot-clean")
    p.add_argument("--text_field", type=str, default="content")
    
    # RL / GRPO Hyperparams
    p.add_argument("--grpo_n_group", type=int, default=8,
                   help="Number of trajectories to sample per token.")
    p.add_argument("--grpo_epsilon", type=float, default=0.2,
                   help="PPO clipping epsilon.")
    p.add_argument("--grpo_kl_beta", type=float, default=0.01,
                   help="Weight for KL penalty against reference model.")
    p.add_argument("--grpo_ponder_cost", type=float, default=0.01,
                   help="Penalty per thinking step (full value; see "
                        "--grpo_ponder_warmup_steps for curriculum).")
    p.add_argument("--grpo_ponder_shape", type=str, default="linear",
                   choices=["linear", "quadratic"],
                   help="'linear' = cost * depth; 'quadratic' = cost * depth^2. "
                        "Quadratic makes shallow thinking cheap while deep "
                        "thinking has to clearly pay for itself.")
    p.add_argument("--grpo_ponder_counterfactual", action="store_true",
                   help="Reward thinking only when CE actually drops vs the "
                        "depth-0 baseline. Wasted thinking pays its ponder "
                        "cost but doesn't lower the floor reward.")
    p.add_argument("--grpo_separate_ponder_norm", action="store_true",
                   help="Z-score task reward within the GRPO group, then "
                        "subtract the absolute ponder cost — prevents the "
                        "group normalisation from squashing the (small) "
                        "ponder magnitude into noise. Ignored when "
                        "--grpo_ponder_counterfactual is set.")
    p.add_argument("--grpo_ponder_warmup_steps", type=int, default=0,
                   help="Curriculum: ramp ponder_cost from 0 to its full "
                        "value linearly over this many steps. 0 disables "
                        "(full cost from step 1). Suggested ~200-500 for "
                        "cold-start runs to give the gate + memory time to "
                        "discover what thinking buys before being penalised.")
    p.add_argument("--max_depth", type=int, default=10)
    
    # Working memory (in-sequence, bounded, write-gated). Reads only at
    # think positions; writes at every real token, gated. Replaces the
    # corpus-RAG approach that previously sat behind --enable_rag.
    p.add_argument("--use_memory", action="store_true",
                   help="Enable bounded working memory inside the model.")
    p.add_argument("--mem_size", type=int, default=1024,
                   help="Max number of write-gated entries kept in the buffer.")
    p.add_argument("--mem_dim", type=int, default=0,
                   help="Memory key/value dim. 0 = use d_model.")

    p.add_argument("--save_ckpt", type=str, default="checkpoints/think_rl.pt")
    p.add_argument("--load_ckpt", type=str, help="Load a pre-trained BPTT checkpoint.")
    p.add_argument("--tokenizer", type=str, default="HuggingFaceTB/SmolLM2-135M",
                   help="HF tokenizer to use. Must match the load_ckpt's "
                        "training tokenizer for sane vocab + token ids.")
    p.add_argument("--tb_dir", type=str, help="TensorBoard log directory.")
    p.add_argument("--think_checkpointing", action="store_true",
                   help="Wrap the policy forward in torch.utils.checkpoint to cut "
                        "activation memory for the loss-bearing pass. Mandated for "
                        "max_depth > 2 per GEMINI.md.")
    p.add_argument("--min_decision_pos", type=int, default=16,
                   help="Minimum context length when sampling the decision "
                        "position. ≥2 avoids fla's decode/step path; ≥16 keeps "
                        "DeltaNet's chunked kernels engaged.")
    p.add_argument("--hard_pos_sampling", action="store_true",
                   help="Pre-pass the input through the model to compute "
                        "per-position CE, then sample the decision position "
                        "from the [hard_ce_min, hard_ce_max] band. Eliminates "
                        "trivial-easy and pathological-extreme positions which "
                        "give RL no useful signal.")
    p.add_argument("--hard_ce_min", type=float, default=1.5,
                   help="Lower bound on per-position CE for hard sampling.")
    p.add_argument("--hard_ce_max", type=float, default=6.0,
                   help="Upper bound on per-position CE for hard sampling.")
    
    # Shared from train_lm
    p.add_argument("--feedback", type=str, default="film")
    p.add_argument("--feedback_pairs", type=str, default="2,28")
    p.add_argument("--feedback_self_k", type=int, default=3)
    
    args = p.parse_args()
    torch.manual_seed(args.seed)
    
    print(f"GRPO Training: {args.arch} scale {args.d_model}d")
    
    # 1. Setup Model & Reference
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    # Use `len(tok)` (full size including added tokens) rather than vocab_size
    # so the thinking_token_id slot is one past the highest existing id.
    thinking_token_id = len(tok)
    model_vocab_size = len(tok) + 1
    # If we're loading a checkpoint whose embed table was sized differently
    # (e.g. distillation pads vocab up to a multiple of 64 + an optional gate
    # slot), match it. Otherwise the resize logic below truncates the
    # pretrained embedding table.
    if args.load_ckpt:
        try:
            ck_cfg = torch.load(args.load_ckpt, map_location="cpu",
                                weights_only=False).get("config", {})
            ck_vocab = int(ck_cfg.get("vocab_size", 0))
            if ck_vocab >= model_vocab_size:
                model_vocab_size = ck_vocab + (0 if ck_vocab > thinking_token_id else 1)
                # If the ckpt had its own thinking_token_id, respect it; else
                # use ours, but make sure it's representable in the embed.
                if ck_cfg.get("thinking_token_id") is not None:
                    thinking_token_id = int(ck_cfg["thinking_token_id"])
                else:
                    thinking_token_id = model_vocab_size - 1
                print(f"  matched ckpt vocab: model_vocab_size={model_vocab_size}, "
                      f"thinking_token_id={thinking_token_id}")
        except Exception as e:
            print(f"  warn: could not read ckpt cfg ({e}); using tokenizer-derived vocab")
    
    fb_pairs = []
    if args.feedback_pairs:
        for p_str in args.feedback_pairs.split(";"):
            t, s = map(int, p_str.split(","))
            fb_pairs.append((t, s))
            
    pad_token_id_int = int(tok.eos_token_id) if tok.eos_token_id is not None else 0
    model_config = dict(
        vocab_size=model_vocab_size, d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, d_head=args.d_head, max_T=args.max_T,
        feedback_mode=args.feedback, feedback_pairs=tuple(fb_pairs),
        feedback_self_k=args.feedback_self_k,
        output_gate=True,
        use_memory=bool(args.use_memory),
        mem_size=int(args.mem_size),
        mem_dim=int(args.mem_dim) if args.mem_dim > 0 else args.d_model,
        thinking_token_id=int(thinking_token_id),
        pad_token_id=int(pad_token_id_int),
        attention_cls=DeltaNetAttention,
    )
    # Extra fields saved to ckpt cfg (but not passed to TinyLM ctor).
    ckpt_cfg_extras = dict(tokenizer=args.tokenizer, arch=args.arch)
    
    model = TinyLM(**model_config).to("cuda")
    # Only materialise the reference model if a KL penalty is actually used —
    # it's a full 217M-param copy that otherwise just wastes GPU memory.
    if args.grpo_kl_beta > 0:
        ref_model = TinyLM(**model_config).to("cuda")
        ref_model.eval()
        for p_param in ref_model.parameters():
            p_param.requires_grad_(False)
    else:
        ref_model = None

    if args.load_ckpt:
        print(f"Loading checkpoint: {args.load_ckpt}")
        ckpt = torch.load(args.load_ckpt, map_location="cuda")
        sd = ckpt["state_dict"]

        # Handle vocab size mismatch for 'from scratch' start
        for key in ["embed.weight", "lm_head.weight"]:
            if key in sd and sd[key].shape != model.state_dict()[key].shape:
                print(f"  Resizing {key} from {sd[key].shape} to {model.state_dict()[key].shape}")
                new_param = model.state_dict()[key].clone()
                n_copy = min(sd[key].shape[0], new_param.shape[0])
                new_param[:n_copy] = sd[key][:n_copy]
                sd[key] = new_param

        model.load_state_dict(sd, strict=False)
        if ref_model is not None:
            ref_model.load_state_dict(sd, strict=False)
        
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

    if args.use_memory:
        print(f"Working memory: enabled, mem_size={args.mem_size}, "
              f"mem_dim={args.mem_dim if args.mem_dim > 0 else args.d_model}")

    # 3. Setup Dataset
    ds = load_dataset(args.dataset, streaming=True, split="train")
    class TokenizedDataset(torch.utils.data.IterableDataset):
        def __init__(self, ds, tok, text_field, T):
            self.ds = ds
            self.tok = tok
            self.text_field = text_field
            self.T = T
        def __iter__(self):
            for x in self.ds:
                ids = self.tok(x[self.text_field], truncation=True, max_length=self.T)["input_ids"]
                if len(ids) < 10: continue
                # Pad to T
                if len(ids) < self.T:
                    ids = ids + [self.tok.eos_token_id] * (self.T - len(ids))
                yield {"input_ids": torch.tensor(ids, dtype=torch.long)}
            
    train_loader = DataLoader(TokenizedDataset(ds, tok, args.text_field, args.T), batch_size=args.batch)
    train_iter = iter(train_loader)
    
    # 3. Training Loop
    if args.tb_dir:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(log_dir=args.tb_dir)
    else:
        tb = None
        
    print(f"\n{'step':>6}  {'reward':>8}  {'advantages':>8}  {'think_rate':>8}  {'avg_depth':>8}")
    
    # Wrap the policy forward when --think_checkpointing is requested. Per
    # GEMINI.md this is mandated for max_depth > 2 to keep activation memory
    # within budget during the loss-bearing pass.
    from torch.utils.checkpoint import checkpoint as _grad_checkpoint

    def policy_forward(input_ids_grad: torch.Tensor):
        if args.think_checkpointing:
            # checkpoint() doesn't pass kwargs into the wrapped fn directly; use
            # a closure so the model still receives the gate flag.
            def _fwd(ids):
                return model(ids, return_gate=True)
            return _grad_checkpoint(_fwd, input_ids_grad, use_reentrant=False)
        return model(input_ids_grad, return_gate=True)

    pad_token_id = int(tok.eos_token_id) if tok.eos_token_id is not None else 0
    min_pos = max(2, int(args.min_decision_pos))


    for step in range(args.steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_ids = batch["input_ids"].to("cuda")
        B, T = input_ids.shape

        # Per-row decision position. Pick from the non-pad region so we don't
        # ask the model to predict an eos pad as the target. The loader
        # right-pads with `eos_token_id` (see TokenizedDataset.__iter__), so
        # the run of pad tokens lives at the tail.
        is_pad = input_ids == pad_token_id
        # First pad index per row (or T if no pad)
        first_pad = torch.where(
            is_pad.any(dim=1),
            is_pad.float().argmax(dim=1),
            torch.full((B,), T, device=is_pad.device),
        )
        # Upper bound for pos: predict a token that is itself non-pad → pos < first_pad
        upper = torch.clamp(first_pad, max=T - 1)
        lower = torch.full_like(upper, min_pos)
        # If upper <= lower for some row, fall back to lower (will be a noisy
        # signal but won't crash). Guarantees pos ≥ min_pos ≥ 2.
        span = (upper - lower).clamp(min=1)
        offset = (torch.rand(B, device=upper.device) * span.float()).long()
        pos = (lower + offset).clamp(min=min_pos, max=T - 1)

        # Hard-example sampling: pre-pass the model to compute per-position CE,
        # then resample `pos` from the [hard_ce_min, hard_ce_max] band (per
        # row). Falls back to the random pos above when no positions fall in
        # the band. Eliminates trivial tokens (which the model already nails
        # and where thinking is wasted) and pathological tokens (which are
        # fundamentally unguessable and where thinking cannot help either).
        if args.hard_pos_sampling:
            with torch.no_grad():
                logits_pre = model(input_ids)            # (B, T, V)
                # CE at position p predicts input_ids[:, p+1]. Per-position CE
                # for positions 0..T-2; position T-1 has no target.
                shift_logits = logits_pre[:, :-1].float()
                shift_targets = input_ids[:, 1:]
                ce_per_pos = F.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.size(-1)),
                    shift_targets.reshape(-1),
                    reduction="none",
                ).reshape(B, T - 1)
                # Eligibility mask: position is in [min_pos-1, first_pad-2]
                # (we predict the *next* token, so the decision is at p-1 and
                # the target is at p; map back: decision pos = p means we
                # predict input_ids[:, p+1], so we need ce_per_pos[:, p] where
                # p ∈ [min_pos-1, first_pad-2]).
                eligible = torch.zeros_like(ce_per_pos, dtype=torch.bool)
                pos_idx = torch.arange(T - 1, device=ce_per_pos.device).unsqueeze(0)
                # ce_per_pos[b, p] corresponds to predicting input_ids[b, p+1].
                # We want decision_pos (where the model emits) such that
                # min_pos ≤ decision_pos < first_pad. decision_pos = p means
                # we predict input_ids[b, decision_pos] = input_ids[b, p+1]
                # for input_ids index p+1. So decision_pos = p + 1; range
                # min_pos ≤ p + 1 ≤ first_pad - 1, i.e. p ∈ [min_pos-1, first_pad-2].
                lower_p = (torch.full_like(first_pad, min_pos) - 1).unsqueeze(1)
                upper_p = (first_pad - 2).unsqueeze(1)
                eligible = (pos_idx >= lower_p) & (pos_idx <= upper_p)
                in_band = (ce_per_pos >= args.hard_ce_min) & (ce_per_pos <= args.hard_ce_max)
                hard = eligible & in_band                  # (B, T-1)
                for b in range(B):
                    cands = hard[b].nonzero(as_tuple=False).flatten()
                    if cands.numel() > 0:
                        pick = cands[torch.randint(0, cands.numel(), (1,))]
                        # decision_pos = chosen p + 1 (predicting next token)
                        pos[b] = int(pick.item()) + 1
        pos_list = pos.tolist()

        initial_contexts = [input_ids[i, :pos_list[i]].tolist() for i in range(B)]
        target_ids = input_ids[range(B), pos].tolist()

        # Phase 1: Rollout
        trajectory_groups = generate_thought_trajectories(
            model, initial_contexts, target_ids,
            n_group=args.grpo_n_group,
            max_depth=args.max_depth,
            thinking_token_id=thinking_token_id,
            block_size=args.T,
            device="cuda",
            pad_token_id=pad_token_id,
        )

        # Phase 2: Advantages.
        # Curriculum: ramp ponder_cost linearly from 0 to its full value over
        # `grpo_ponder_warmup_steps`. Avoids the cold-start trap where the
        # gate + memory haven't yet discovered what thinking buys and the
        # ponder cost prunes thinking before it can demonstrate value.
        if args.grpo_ponder_warmup_steps > 0:
            ramp = min(1.0, step / float(args.grpo_ponder_warmup_steps))
        else:
            ramp = 1.0
        ponder_cost_eff = args.grpo_ponder_cost * ramp
        advantages = compute_grpo_advantages(
            trajectory_groups, ponder_cost_eff,
            ponder_shape=args.grpo_ponder_shape,
            counterfactual=args.grpo_ponder_counterfactual,
            separate_ponder_norm=args.grpo_separate_ponder_norm,
        ).to("cuda")

        # Pure-task rewards (for logging) — exclude the ponder term so the
        # reported PPL is true perplexity of the optimised token. Also bucket
        # CE by trajectory depth to see whether thinking actually reduces CE.
        task_rewards = torch.zeros((B, args.grpo_n_group))
        all_depths: list[int] = []
        depth_ce: dict[int, list[float]] = {}
        for i, group in enumerate(trajectory_groups):
            for j, t in enumerate(group):
                target = torch.tensor(t.target_id)
                logits = t.final_logits.unsqueeze(0)
                ce = F.cross_entropy(logits, target.unsqueeze(0))
                task_rewards[i, j] = -ce.item()
                all_depths.append(t.depth)
                depth_ce.setdefault(t.depth, []).append(float(ce.item()))

        # Phase 3: Batched GRPO update.
        # Build one (B*N, T_max) tensor from every trajectory's final context,
        # run a single grad-enabled policy forward, gather the last-position
        # gate logit per row, and assemble the loss in one vector op. This
        # collapses the previous 16 sequential graphs into 1.
        # ----------------------------------------------------------------
        # Full-trajectory GRPO credit assignment.
        #
        # Each trajectory of depth d made d+1 gate decisions (one per
        # iteration in the rollout). Its policy log-prob is the SUM of
        # log P(action_i) over all those decisions. We recompute every
        # gate logit in one batched forward and gather at the per-decision
        # positions, then sum.
        #
        # For a trajectory with final context  initial + [think]*depth
        # (length N + depth), the gate decisions happened at positions
        # [N-1, N, ..., N+depth-1] in that context. After left-padding to
        # max_len, those become [N-1+pad, ..., N+depth-1+pad].
        # ----------------------------------------------------------------
        total_trajs = B * args.grpo_n_group
        from experiments.thinking import MIN_ROLLOUT_LEN

        ctxs: list[list[int]] = []
        per_traj_decisions: list[list[int]] = []   # per-traj decision positions IN CTX
        per_traj_actions:   list[list[int]] = []   # aligned actions (1=Think, 0=Emit)
        per_traj_old_logp:  list[list[float]] = [] # aligned rollout log-probs
        for i in range(B):
            for j in range(args.grpo_n_group):
                traj = trajectory_groups[i][j]
                ctx_full = traj.initial_context + [thinking_token_id] * traj.depth
                ctx = ctx_full[-args.T:]
                ctx_len = len(ctx)
                # How many of the (initial + think) suffix are visible after
                # the [-args.T:] crop: depth is small (≤ max_depth), and we
                # require initial-tokens to remain (the rollout already
                # caps initial_context this way).
                initial_in_ctx = ctx_len - traj.depth
                # decision positions in ctx coordinates
                decisions = [initial_in_ctx - 1 + k for k in range(traj.depth + 1)]
                # Drop any that fell off the left edge (negative positions)
                # together with their corresponding actions / old logprobs.
                actions = list(traj.actions)             # length depth+1
                old_logps = list(traj.action_logprobs)    # length depth+1
                kept = [(p, a, lp) for p, a, lp in zip(decisions, actions, old_logps) if p >= 0]
                if not kept:
                    # Pathological: nothing recoverable. Skip — should be impossible
                    # given the rollout's MIN_ROLLOUT_LEN guard.
                    kept = [(0, int(traj.actions[-1]), float(traj.action_logprobs[-1]))]
                ctxs.append(ctx)
                per_traj_decisions.append([p for p, _, _ in kept])
                per_traj_actions.append([a for _, a, _ in kept])
                per_traj_old_logp.append([lp for _, _, lp in kept])

        max_len = max(MIN_ROLLOUT_LEN, max(len(c) for c in ctxs))
        max_decisions = max(len(d) for d in per_traj_decisions)

        ids_grad = torch.full(
            (total_trajs, max_len), pad_token_id, dtype=torch.long, device="cuda",
        )
        # Padded decision-position / action / old-logprob tensors, with a
        # boolean validity mask.
        dec_pos_t   = torch.zeros((total_trajs, max_decisions), dtype=torch.long, device="cuda")
        actions_t   = torch.zeros((total_trajs, max_decisions), dtype=torch.long, device="cuda")
        old_logp_t  = torch.zeros((total_trajs, max_decisions), dtype=torch.float32, device="cuda")
        valid_t     = torch.zeros((total_trajs, max_decisions), dtype=torch.bool,  device="cuda")
        for k, (ctx, decisions, actions, oldlp) in enumerate(zip(
                ctxs, per_traj_decisions, per_traj_actions, per_traj_old_logp)):
            ids_grad[k, -len(ctx):] = torch.tensor(ctx, dtype=torch.long, device="cuda")
            pad_offset = max_len - len(ctx)
            for di, (p, a, lp) in enumerate(zip(decisions, actions, oldlp)):
                dec_pos_t[k, di]  = p + pad_offset
                actions_t[k, di]  = int(a)
                old_logp_t[k, di] = float(lp)
                valid_t[k, di]    = True

        _, _ = policy_forward(ids_grad)  # populates model._last_gate_logits (BN, T)
        row_idx = torch.arange(total_trajs, device="cuda").unsqueeze(1).expand(-1, max_decisions)
        gate_logits = model._last_gate_logits[row_idx, dec_pos_t]   # (BN, D)

        # log P(action | gate_logit): σ(gate) = P(Emit), so
        #   action==Think (1) → logsigmoid(-gate)
        #   action==Emit  (0) → logsigmoid( gate)
        log_p_per_dec = torch.where(
            actions_t == 1,
            F.logsigmoid(-gate_logits),
            F.logsigmoid(gate_logits),
        )
        # Mask invalid decisions to zero (so they contribute 0 to the sum).
        valid_f = valid_t.to(log_p_per_dec.dtype)
        log_p_traj  = (log_p_per_dec * valid_f).sum(dim=1)              # (BN,)
        old_logp_traj = (old_logp_t.to(log_p_per_dec.dtype) * valid_f).sum(dim=1)

        adv_flat = advantages.reshape(-1).to(log_p_traj.dtype)
        ratio  = torch.exp(log_p_traj - old_logp_traj)
        surr1  = ratio * adv_flat
        surr2  = torch.clamp(ratio, 1.0 - args.grpo_epsilon,
                              1.0 + args.grpo_epsilon) * adv_flat
        policy_loss = -torch.minimum(surr1, surr2).mean()

        # ---- KL to reference, also over the full trajectory --------------
        if args.grpo_kl_beta > 0 and ref_model is not None:
            with torch.no_grad():
                _ = ref_model(ids_grad, return_gate=True)
                ref_gate_logits = ref_model._last_gate_logits[row_idx, dec_pos_t]
                ref_log_p_per_dec = torch.where(
                    actions_t == 1,
                    F.logsigmoid(-ref_gate_logits),
                    F.logsigmoid(ref_gate_logits),
                )
                ref_log_p_traj = (ref_log_p_per_dec * valid_f).sum(dim=1)
            kl_loss = args.grpo_kl_beta * (log_p_traj - ref_log_p_traj).mean()
        else:
            kl_loss = torch.zeros((), device="cuda", dtype=log_p_traj.dtype)

        loss = policy_loss + kl_loss
        # Compatibility shims for the diagnostics block below that still
        # references `cur_gate_logits` / `actions_t`.
        cur_gate_logits = gate_logits[valid_t]                # (sum_valid,)
        actions_t_flat  = actions_t[valid_t]                  # (sum_valid,)

        opt.zero_grad()
        loss.backward()
        opt.step()

        # Logging
        if step % 10 == 0:
            avg_task_reward = task_rewards.mean().item()
            think_rate = sum(1 for d in all_depths if d > 0) / len(all_depths)
            avg_depth = sum(all_depths) / len(all_depths)
            current_ppl = float(torch.exp(torch.tensor(-avg_task_reward)).item())

            print(f"{step:>6}  {avg_task_reward:>8.4f}  {advantages.std().item():>8.4f}  "
                  f"{think_rate:>8.4f}  {avg_depth:>8.4f}  ppl={current_ppl:.2f}")

            # Diagnostics: gate distribution, weight-norm bootstrap, depth-CE.
            gate_sigmoid = torch.sigmoid(cur_gate_logits.detach()).float()
            gate_mean = gate_sigmoid.mean().item()
            gate_std = gate_sigmoid.std().item() if gate_sigmoid.numel() > 1 else 0.0
            gate_head_norm = float(model.gate_head.weight.detach().norm().item())

            # Memory-side diagnostics: weight norms + mean write-gate value over
            # the loss-bearing forward's positions. If any of these stay
            # bit-flat, the memory path is dead — same failure mode the
            # old rag_projection had.
            if model.use_memory:
                mem = model.memory
                mem_proj_norm = float(mem.W_proj.weight.detach().norm().item())
                mem_write_norm = float(mem.W_write.weight.detach().norm().item())
                mem_query_norm = float(mem.W_q.weight.detach().norm().item())
                mem_value_norm = float(mem.W_v.weight.detach().norm().item())
                # Read the last write-gate produced by the most recent
                # policy_forward (just above) — these are the actual gates
                # computed on the loss-bearing tokens.
                last_g = getattr(mem, "_last_write_gate", None)
                write_gate_mean = float(last_g.mean().item()) if last_g is not None else 0.0
            else:
                mem_proj_norm = mem_write_norm = mem_query_norm = mem_value_norm = 0.0
                write_gate_mean = 0.0

            if tb:
                tb.add_scalar("grpo/loss", loss.item(), step)
                tb.add_scalar("grpo/policy_loss", policy_loss.item(), step)
                tb.add_scalar("grpo/kl_loss", kl_loss.item(), step)
                tb.add_scalar("grpo/think_rate", think_rate, step)
                tb.add_scalar("grpo/avg_depth", avg_depth, step)
                tb.add_scalar("grpo/task_reward", avg_task_reward, step)
                tb.add_scalar("grpo/ppl", current_ppl, step)
                tb.add_scalar("grpo/gate_mean", gate_mean, step)
                tb.add_scalar("grpo/gate_std", gate_std, step)
                tb.add_scalar("grpo/gate_head_norm", gate_head_norm, step)
                if model.use_memory:
                    tb.add_scalar("mem/proj_norm", mem_proj_norm, step)
                    tb.add_scalar("mem/write_norm", mem_write_norm, step)
                    tb.add_scalar("mem/query_norm", mem_query_norm, step)
                    tb.add_scalar("mem/value_norm", mem_value_norm, step)
                    tb.add_scalar("mem/write_gate_mean", write_gate_mean, step)
                for d_bucket, ces in depth_ce.items():
                    tb.add_scalar(
                        f"grpo/depth_ce_d{d_bucket}",
                        sum(ces) / len(ces),
                        step,
                    )
                tb.add_histogram("grpo/depth_dist",
                                 torch.tensor(all_depths, dtype=torch.float), step)
                
        if step > 0 and step % 1000 == 0:
            torch.save({
                "step": step,
                "state_dict": model.state_dict(),
                "config": {**model_config, **ckpt_cfg_extras},
            }, args.save_ckpt)

    # Final save at end-of-training (the % 1000 schedule above skips step 999).
    torch.save({
        "step": args.steps - 1,
        "state_dict": model.state_dict(),
        "config": {**model_config, **ckpt_cfg_extras},
    }, args.save_ckpt)
    print(f"Saved final checkpoint to {args.save_ckpt}")

if __name__ == "__main__":
    main()
