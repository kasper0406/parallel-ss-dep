"""
A tiny, self-contained DeltaNet with latent thinking + working memory.

Pure PyTorch — no fla / triton / transformers / flash-attn. Everything here is
meant to be READ. The whole point of this challenge is that the mechanism is
small enough to hold in your head.

Architecture (why):
  - DeltaNet is a *linear RNN*: it carries a FIXED-SIZE recurrent state S
    (a d_head x d_head matrix per head), updated one timestep at a time with a
    delta rule. There is NO KV cache and NO attention over the input, so
    inference memory is O(1) in sequence length. That bounded-state constraint
    is the whole game — it is what makes the model cheap, and what makes
    "thinking" interesting (you cannot just attend back to re-read the prompt).

  - Latent thinking (Coconut-style): after the prompt, append R "think" slots.
    At each think slot we feed the model's OWN last hidden state back in as the
    next input embedding (through a small learned adapter), instead of a token
    embedding. This lets the model do R extra sequential compute steps before
    it has to answer — full-bandwidth (a whole d_model vector flows between
    steps, not a 1-of-vocab token). Think slots can be "state-readonly": the
    recurrent state is READ but not WRITTEN at think steps, so the prompt's
    bindings cannot be corrupted by thinking.

  - Working memory (WM): a small write-gated buffer of past hidden states,
    read by soft-attention at think positions and injected into the think-slot
    input. Bounded size, content-addressable. On by default.

Token layout convention (see tasks.py): the LAST n+3 ids of the vocab are
special: QUERY, THINK, PAD live just above the value/structure tokens.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# DeltaNet recurrent layer
# ---------------------------------------------------------------------------
class DeltaNetLayer(nn.Module):
    """One delta-rule linear-RNN layer.

    Per head h, state S_h is a (d_head x d_head) matrix. At each timestep t:
        k, v, q  = projections of the input  (L2-normalised k, q)
        beta     = sigmoid(write-gate)            in [0, 1]
        S <- S + beta * (v - S k) k^T             # delta rule (associative)
        out      = S q                            # read

    `state_readonly` (per-timestep boolean mask) forces beta -> 0, so the state
    is read but never written at those positions. We run an explicit per-step
    loop; T is tiny in this challenge (<~ 40) so this is plenty fast and maximally
    readable.
    """

    def __init__(self, d_model: int, n_heads: int, d_head: int, conv_size: int = 3):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        inner = n_heads * d_head
        self.q_proj = nn.Linear(d_model, inner, bias=False)
        self.k_proj = nn.Linear(d_model, inner, bias=False)
        self.v_proj = nn.Linear(d_model, inner, bias=False)
        self.b_proj = nn.Linear(d_model, n_heads, bias=True)   # write-gate logits
        self.o_proj = nn.Linear(inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        # Short causal depthwise conv on q/k/v. Standard in linear-attention
        # models: it lets the delta rule bind a VALUE token to the KEY token that
        # preceded it (k/v/q would otherwise depend only on the current token,
        # making associative recall from (key, value) pairs impossible). Still
        # bounded state — the conv has a tiny fixed receptive field.
        self.conv_size = conv_size
        self.q_conv = nn.Conv1d(inner, inner, conv_size, groups=inner, bias=False)
        self.k_conv = nn.Conv1d(inner, inner, conv_size, groups=inner, bias=False)
        self.v_conv = nn.Conv1d(inner, inner, conv_size, groups=inner, bias=False)

    def _causal_conv(self, conv, t):
        # t: (B, T, inner) -> causal depthwise conv -> (B, T, inner)
        x = t.transpose(1, 2)                                   # (B, inner, T)
        x = F.pad(x, (self.conv_size - 1, 0))                   # left pad => causal
        return conv(x).transpose(1, 2)

    def forward(self, x: torch.Tensor, readonly_mask: torch.Tensor | None = None):
        # x: (B, T, d_model); readonly_mask: (B, T) bool, True => beta forced 0
        B, T, _ = x.shape
        H, Dh = self.n_heads, self.d_head
        h = self.norm(x)
        q = self._causal_conv(self.q_conv, self.q_proj(h)).view(B, T, H, Dh)
        k = self._causal_conv(self.k_conv, self.k_proj(h)).view(B, T, H, Dh)
        v = self._causal_conv(self.v_conv, self.v_proj(h)).view(B, T, H, Dh)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        beta = torch.sigmoid(self.b_proj(h))                    # (B, T, H)
        if readonly_mask is not None:
            beta = beta.masked_fill(readonly_mask.unsqueeze(-1), 0.0)

        S = x.new_zeros(B, H, Dh, Dh)                           # recurrent state
        outs = []
        for t in range(T):
            k_t = k[:, t]                                       # (B, H, Dh)
            v_t = v[:, t]
            q_t = q[:, t]
            beta_t = beta[:, t].unsqueeze(-1)                   # (B, H, 1)
            # predicted value under current state, then corrected write
            Sk = torch.einsum("bhij,bhj->bhi", S, k_t)         # (B, H, Dh)
            delta = beta_t * (v_t - Sk)                         # (B, H, Dh)
            S = S + torch.einsum("bhi,bhj->bhij", delta, k_t)  # outer product
            o_t = torch.einsum("bhij,bhj->bhi", S, q_t)        # read (B, H, Dh)
            outs.append(o_t)
        o = torch.stack(outs, dim=1).reshape(B, T, H * Dh)
        return x + self.o_proj(o)                               # residual


class MLP(nn.Module):
    def __init__(self, d_model: int, mult: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, mult * d_model)
        self.fc2 = nn.Linear(mult * d_model, d_model)

    def forward(self, x):
        return x + self.fc2(F.gelu(self.fc1(self.norm(x))))


# ---------------------------------------------------------------------------
# Working memory
# ---------------------------------------------------------------------------
class WorkingMemory(nn.Module):
    """Write-gated buffer of past hidden states, read by soft-attention.

    During a forward pass we collect a per-position write-gate over the whole
    sequence and keep the top `mem_size` gated hidden states as the buffer
    (keys = values = those hiddens). At READ positions (think slots) we form a
    query, soft-attend over the buffer, and return the retrieved vector — to be
    injected into the think-slot input. Bounded size => content-addressable
    memory with no growth in T.
    """

    def __init__(self, d_model: int, mem_size: int = 32):
        super().__init__()
        self.mem_size = mem_size
        self.write_gate = nn.Linear(d_model, 1)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.scale = 1.0 / math.sqrt(d_model)

    def build_buffer(self, hiddens: torch.Tensor):
        """hiddens: (B, T, d) — pick top mem_size by write-gate -> buffer."""
        B, T, d = hiddens.shape
        gate = torch.sigmoid(self.write_gate(hiddens)).squeeze(-1)   # (B, T)
        m = min(self.mem_size, T)
        idx = gate.topk(m, dim=1).indices                            # (B, m)
        buf = torch.gather(hiddens, 1, idx.unsqueeze(-1).expand(-1, -1, d))
        gate_sel = torch.gather(gate, 1, idx)                        # (B, m)
        return buf, gate_sel

    def read(self, query_hidden: torch.Tensor, buf: torch.Tensor,
             gate_sel: torch.Tensor):
        """query_hidden: (B, d) -> retrieved (B, d) by soft-attention over buf."""
        q = self.q_proj(query_hidden).unsqueeze(1)                  # (B, 1, d)
        k = self.k_proj(buf)                                        # (B, m, d)
        v = self.v_proj(buf)                                        # (B, m, d)
        att = (q @ k.transpose(1, 2)).squeeze(1) * self.scale       # (B, m)
        att = att + torch.log(gate_sel + 1e-6)                      # gate-weighted
        w = F.softmax(att, dim=-1)                                  # (B, m)
        return (w.unsqueeze(1) @ v).squeeze(1)                      # (B, d)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    vocab_size: int
    max_T: int                    # max sequence length (for positional embedding)
    d_model: int = 128
    n_layers: int = 3
    n_heads: int = 4
    d_head: int = 32
    thinking_id: int = 0          # token id used for the think slot
    state_readonly: bool = True   # think steps READ but do not WRITE state
    use_memory: bool = True
    mem_size: int = 32


class DeltaNetLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        # Learnable absolute positional embedding. Essential here: the prompt
        # presents (key, value) pairs as *consecutive same-type tokens*, so the
        # model needs position to tell a key slot from a value slot.
        self.pos_embed = nn.Embedding(cfg.max_T, cfg.d_model)
        self.blocks = nn.ModuleList()
        for _ in range(cfg.n_layers):
            self.blocks.append(DeltaNetLayer(cfg.d_model, cfg.n_heads, cfg.d_head))
            self.blocks.append(MLP(cfg.d_model))
        self.out_norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # adapter applied to the fed-back hidden before it re-enters as an embed
        self.think_adapter = nn.Linear(cfg.d_model, cfg.d_model)
        if cfg.use_memory:
            self.memory = WorkingMemory(cfg.d_model, cfg.mem_size)
        else:
            self.memory = None

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def add_pos(self, x: torch.Tensor) -> torch.Tensor:
        """Add absolute positional embedding to (B, T, d) embeddings."""
        T = x.shape[1]
        pos = torch.arange(T, device=x.device)
        return x + self.pos_embed(pos).unsqueeze(0)

    # -- core trunk ---------------------------------------------------------
    def trunk(self, x: torch.Tensor, readonly_mask: torch.Tensor | None = None):
        """x: (B, T, d) input embeddings (positions already added) -> (logits, hidden)."""
        for block in self.blocks:
            if isinstance(block, DeltaNetLayer):
                x = block(x, readonly_mask=readonly_mask)
            else:
                x = block(x)
        h = self.out_norm(x)
        return self.lm_head(h), x          # logits from normed h; raw h for feedback

    def forward(self, input_ids: torch.Tensor):
        return self.trunk(self.add_pos(self.embed(input_ids)))[0]

    # -- latent thinking forward -------------------------------------------
    def think_forward(self, base_ids: torch.Tensor, R: int, mode: str = "latent",
                      return_steps: bool = False):
        """Append R think slots and emit from the last one.

        mode:
          'none'   — plain forward, answer from the last prompt position (R ignored).
          'latent' — feed the trunk's own hidden (through think_adapter, + WM read)
                     back as the think-slot input embedding for R steps.
          'token'  — append a think slot whose input stays embed(THINK) every step
                     (a discrete-feedback control: no bandwidth between steps).

        Returns answer logits (B, V); or per-step logits (B, R, V) if return_steps.
        """
        B, Lb = base_ids.shape
        device = base_ids.device
        if mode == "none" or R == 0:
            logits, _ = self.trunk(self.add_pos(self.embed(base_ids)))
            out = logits[:, -1, :]
            return out.unsqueeze(1) if return_steps else out

        # base prompt embeddings with positions 0..Lb-1
        base_emb = self.add_pos(self.embed(base_ids))              # (B, Lb, d)
        # positional embedding for the single appended think slot (position Lb)
        think_pos = self.pos_embed(torch.tensor(Lb, device=device))  # (d,)
        think_col = torch.full((B, 1), self.cfg.thinking_id, dtype=torch.long,
                               device=device)
        think_emb = self.embed(think_col) + think_pos              # (B, 1, d)
        # readonly mask: True only at the appended think slot (index Lb)
        if self.cfg.state_readonly:
            ro = torch.zeros(B, Lb + 1, dtype=torch.bool, device=device)
            ro[:, Lb] = True
        else:
            ro = None

        # initial latent = the prompt's last hidden
        _logits0, h0 = self.trunk(base_emb)
        z = h0[:, -1, :]                                           # (B, d)

        # build the WM buffer once from the prompt hiddens
        if self.memory is not None:
            buf, gate_sel = self.memory.build_buffer(h0)

        step_logits = []
        for _ in range(R):
            if mode == "latent":
                slot = self.think_adapter(z)                       # (B, d)
                if self.memory is not None:
                    slot = slot + self.memory.read(z, buf, gate_sel)
                slot_emb = (slot + think_pos).unsqueeze(1)         # (B, 1, d)
            elif mode == "token":
                slot_emb = think_emb
            else:
                raise ValueError(mode)
            ie = torch.cat([base_emb, slot_emb], dim=1)            # (B, Lb+1, d)
            logits, h = self.trunk(ie, readonly_mask=ro)
            z = h[:, -1, :]
            step_logits.append(logits[:, -1, :])
        if return_steps:
            return torch.stack(step_logits, dim=1)                 # (B, R, V)
        return step_logits[-1]
