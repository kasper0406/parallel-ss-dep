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
                   help="Penalty per thinking step.")
    p.add_argument("--max_depth", type=int, default=10)
    
    # RAG Config
    p.add_argument("--enable_rag", action="store_true")
    p.add_argument("--rag_n_chunks", type=int, default=2000)
    p.add_argument("--rag_dataset", type=str, default="codeparrot/codeparrot-clean")
    
    p.add_argument("--save_ckpt", type=str, default="checkpoints/think_rl.pt")
    p.add_argument("--load_ckpt", type=str, help="Load a pre-trained BPTT checkpoint.")
    p.add_argument("--tb_dir", type=str, help="TensorBoard log directory.")
    
    # Shared from train_lm
    p.add_argument("--feedback", type=str, default="film")
    p.add_argument("--feedback_pairs", type=str, default="2,28")
    p.add_argument("--feedback_self_k", type=int, default=3)
    
    args = p.parse_args()
    torch.manual_seed(args.seed)
    
    print(f"GRPO Training: {args.arch} scale {args.d_model}d")
    
    # 1. Setup Model & Reference
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    thinking_token_id = len(tok)
    model_vocab_size = len(tok) + 1
    
    fb_pairs = []
    if args.feedback_pairs:
        for p_str in args.feedback_pairs.split(";"):
            t, s = map(int, p_str.split(","))
            fb_pairs.append((t, s))
            
    model_config = dict(
        vocab_size=model_vocab_size, d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, d_head=args.d_head, max_T=args.max_T,
        feedback_mode=args.feedback, feedback_pairs=tuple(fb_pairs),
        feedback_self_k=args.feedback_self_k,
        output_gate=True,
        attention_cls=DeltaNetAttention
    )
    
    model = TinyLM(**model_config).to("cuda")
    ref_model = TinyLM(**model_config).to("cuda")
    ref_model.eval()
    for p_param in ref_model.parameters():
        p_param.requires_grad_(False)
        
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
        ref_model.load_state_dict(sd, strict=False)
        
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    
    # 2. Setup RAG Database (In-Memory Tensor)
    rag_keys = None
    if args.enable_rag:
        print(f"Initializing In-Memory RAG DB ({args.rag_n_chunks} chunks)...")
        rag_ds = load_dataset(args.rag_dataset, streaming=True, split="train")
        chunks = []
        with torch.no_grad():
            for i, x in enumerate(rag_ds):
                ids = tok(x[args.text_field], truncation=True, max_length=256, return_tensors="pt")["input_ids"].to("cuda")
                if ids.shape[1] < 50: continue
                _, hidden = ref_model(ids, return_hidden=True)
                chunks.append(hidden.mean(dim=1)) # (1, d_model)
                if len(chunks) >= args.rag_n_chunks: break
                if len(chunks) % 500 == 0: print(f"  Embedded {len(chunks)}/{args.rag_n_chunks}")
        rag_keys = torch.cat(chunks, dim=0) # (N, d_model)
        rag_keys = F.normalize(rag_keys, p=2, dim=1)
        print(f"RAG DB Ready: {rag_keys.shape}")

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
    
    for step in range(args.steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
            
        input_ids = batch["input_ids"].to("cuda")
        B, T = input_ids.shape
        
        # Pick one random token position per row to optimize (for efficiency)
        # Or optimize all? For now, let's pick a random mid-sequence position.
        pos = torch.randint(1, T-1, (B,))
        
        initial_contexts = [input_ids[i, :pos[i]].tolist() for i in range(B)]
        target_ids = input_ids[range(B), pos].tolist()
        
        # Phase 1: Rollout
        trajectory_groups = generate_thought_trajectories(
            model, initial_contexts, target_ids,
            n_group=args.grpo_n_group,
            max_depth=args.max_depth,
            thinking_token_id=thinking_token_id,
            block_size=args.T,
            device="cuda",
            rag_keys=rag_keys
        )
        
        # Phase 2: Advantages
        advantages = compute_grpo_advantages(trajectory_groups, args.grpo_ponder_cost).to("cuda")
        
        # Collect rewards for logging
        rewards = torch.zeros((B, args.grpo_n_group))
        for i, group in enumerate(trajectory_groups):
            for j, t in enumerate(group):
                target = torch.tensor(t.target_id)
                logits = t.final_logits.unsqueeze(0)
                ce = F.cross_entropy(logits, target.unsqueeze(0))
                rewards[i, j] = -ce.item() - (args.grpo_ponder_cost * t.depth)

        # Phase 3: GRPO Update
        # Re-run forward only for the decision points to get gradients
        # We need the log_probs under current policy for the sampled actions.
        
        loss = torch.zeros((), device="cuda")
        
        total_trajs = B * args.grpo_n_group
        all_rewards = []
        all_depths = []
        
        # Batch re-run for efficiency
        # For each trajectory, we need the gate logits at each step.
        # This is slightly complex because trajectories have different lengths.
        
        # For this prototype, we'll do a simplified update: 
        # only backprop through the decision points.
        
        for i in range(B):
            for j in range(args.grpo_n_group):
                traj = trajectory_groups[i][j]
                adv = advantages[i, j]
                
                # Re-run with gradients
                # Construct context
                ctx = (traj.initial_context + [thinking_token_id] * traj.depth)[-args.T:]
                input_ids_grad = torch.tensor([ctx], dtype=torch.long, device="cuda")
                
                # We need all gate logits along the trajectory to compute the policy probability
                # p(traj) = product p(action_t)
                # But for Linear RNNs, we can just run the whole sequence.
                logits, gate = model(input_ids_grad, return_gate=True)
                # We only need the positions where we took actions (the last positions of each pass)
                # In our current DeltaNet implementation, one forward pass = many tokens.
                # But when thinking, it's one pass = one thought step.
                
                # Simplified: only optimize the very last decision point for now
                # (whether to Emit or Think one more time)
                # Real GRPO would optimize the whole chain.
                
                current_gate_logit = model._last_gate_logits[0, -1]
                # Prob(Emit) = sigmoid(logit), Prob(Think) = 1 - Prob(Emit)
                
                action = traj.actions[-1] # 1 for Think, 0 for Emit
                if action == 1:
                    log_prob = F.logsigmoid(-current_gate_logit)
                else:
                    log_prob = F.logsigmoid(current_gate_logit)
                    
                # Reference model for KL
                with torch.no_grad():
                    ref_logits, ref_gate = ref_model(input_ids_grad, return_gate=True)
                    ref_gate_logit = ref_model._last_gate_logits[0, -1]
                    if action == 1:
                        ref_log_prob = F.logsigmoid(-ref_gate_logit)
                    else:
                        ref_log_prob = F.logsigmoid(ref_gate_logit)
                
                ratio = torch.exp(log_prob - traj.action_logprobs[-1])
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - args.grpo_epsilon, 1.0 + args.grpo_epsilon) * adv
                
                policy_loss = -torch.min(surr1, surr2)
                kl_loss = args.grpo_kl_beta * (log_prob - ref_log_prob) # Simplified KL
                
                loss += (policy_loss + kl_loss) / total_trajs
                
                all_depths.append(traj.depth)

        opt.zero_grad()
        loss.backward()
        opt.step()
        
        # Logging
        if step % 10 == 0:
            avg_reward = rewards.mean().item()
            think_rate = sum(1 for d in all_depths if d > 0) / len(all_depths)
            avg_depth = sum(all_depths) / len(all_depths)
            
            # Compute PPL from rewards (Reward = -CE, so PPL = exp(-Reward))
            # Note: This is PPL specifically for the tokens being optimized
            current_ppl = torch.exp(torch.tensor(-avg_reward)).item()

            print(f"{step:>6}  {avg_reward:>8.4f}  {advantages.std().item():>8.4f}  {think_rate:>8.4f}  {avg_depth:>8.4f}  ppl={current_ppl:.2f}")
            
            if tb:
                tb.add_scalar("grpo/loss", loss.item(), step)
                tb.add_scalar("grpo/think_rate", think_rate, step)
                tb.add_scalar("grpo/avg_depth", avg_depth, step)
                tb.add_scalar("grpo/reward", avg_reward, step)
                tb.add_scalar("grpo/ppl", current_ppl, step)
                tb.add_histogram("grpo/depth_dist", torch.tensor(all_depths, dtype=torch.float), step)
                
        if step > 0 and step % 1000 == 0:
            torch.save({
                "step": step,
                "state_dict": model.state_dict(),
                "config": model_config,
            }, args.save_ckpt)

if __name__ == "__main__":
    main()
