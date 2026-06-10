"""
Episode Diagnostic Transformer (EDT)
Sequence model over episode trajectories that classifies failure modes
and recommends reward fixes — replacing the Gemini LLM dependency.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Label vocabularies
# ---------------------------------------------------------------------------

FAILURE_MODES = [
    "KILL",
    "STALEMATE",
    "CRASH",
    "CLOSE_RANGE_LOSS",
    "DISENGAGEMENT",
    "TIMEOUT_DRIFT",
]

FIX_TYPES = [
    "NO_FIX_NEEDED",
    "ADD_ALTITUDE_PENALTY",
    "REDUCE_PROXIMITY_RANGE",
    "ADD_FIRING_BONUS",
    "INCREASE_KILL_REWARD",
    "FIX_OPP_ADVANTAGE",
]

N_FAILURE = len(FAILURE_MODES)
N_FIX     = len(FIX_TYPES)

# ---------------------------------------------------------------------------
# Per-timestep feature layout  (7 floats, all normalised to ≈ [0, 1])
# ---------------------------------------------------------------------------
# 0  distance      / 15000
# 1  agent_health  / 100
# 2  opp_health    / 100
# 3  reward clipped to [-30, 30] then / 30  → [-1, 1]
# 4  altitude      / 2000
# 5  agent_fired   (binary: 1 when within 600m)
# 6  step_progress  t / max_steps

N_FEATURES = 7
D_MODEL    = 64
N_HEADS    = 4
N_LAYERS   = 2
FF_DIM     = 256
MAX_LEN    = 512   # trajectories longer than this are truncated


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = MAX_LEN + 1):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# ---------------------------------------------------------------------------
# EDT
# ---------------------------------------------------------------------------

class EpisodeDiagnosticTransformer(nn.Module):
    """
    Maps a padded episode trajectory (B, T, N_FEATURES) to:
      - failure_logits  (B, N_FAILURE)  — what went wrong
      - fix_logits      (B, N_FIX)      — what to fix in the reward function
      - attn_weights    (B, T)          — timestep importance for interpretability
    """

    def __init__(
        self,
        n_features: int = N_FEATURES,
        d_model: int    = D_MODEL,
        n_heads: int    = N_HEADS,
        n_layers: int   = N_LAYERS,
        ff_dim: int     = FF_DIM,
        max_len: int    = MAX_LEN,
        dropout: float  = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        self.cls_token  = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc    = PositionalEncoding(d_model, max_len + 1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model        = d_model,
            nhead          = n_heads,
            dim_feedforward= ff_dim,
            dropout        = dropout,
            batch_first    = True,
            norm_first     = True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.failure_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, N_FAILURE),
        )
        self.fix_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, N_FIX),
        )

    def forward(
        self,
        x: torch.Tensor,
        pad_mask: torch.Tensor = None,
    ):
        """
        x        : (B, T, N_FEATURES) — padded trajectories
        pad_mask : (B, T) boolean — True at padding positions (no CLS column here;
                   we prepend it internally)

        Returns failure_logits, fix_logits, attn_weights (B, T)
        """
        B, T, _ = x.shape

        x = self.input_proj(x)                            # (B, T, D)
        cls = self.cls_token.expand(B, -1, -1)            # (B, 1, D)
        x   = torch.cat([cls, x], dim=1)                  # (B, T+1, D)
        x   = self.pos_enc(x)

        # Build full padding mask including the CLS column (never masked)
        if pad_mask is not None:
            cls_col   = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
            full_mask = torch.cat([cls_col, pad_mask], dim=1)  # (B, T+1)
        else:
            full_mask = None

        out     = self.transformer(x, src_key_padding_mask=full_mask)
        cls_out = out[:, 0]                               # (B, D)

        failure_logits = self.failure_head(cls_out)
        fix_logits     = self.fix_head(cls_out)

        # Attention proxy: centred cosine similarity (subtract mean so generic
        # early timesteps don't dominate; highlights *distinctive* moments)
        seq_repr = out[:, 1:]                                # (B, T, D)
        sim = F.cosine_similarity(
            cls_out.unsqueeze(1),   # (B, 1, D)
            seq_repr,
            dim=-1,
        )                                                    # (B, T)
        # Centre per-sample before softmax
        if pad_mask is not None:
            valid = (~pad_mask).float()
            mean_sim = (sim * valid).sum(dim=1, keepdim=True) / valid.sum(dim=1, keepdim=True).clamp(min=1)
            sim = sim - mean_sim
            sim = sim.masked_fill(pad_mask, float("-inf"))
        else:
            sim = sim - sim.mean(dim=1, keepdim=True)
        attn_weights = torch.softmax(sim, dim=-1)

        return failure_logits, fix_logits, attn_weights


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_pad_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """Returns (B, max_len) bool mask — True where padded."""
    idx = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return idx >= lengths.unsqueeze(1)


def trajectory_to_tensor(
    timesteps: list,
    max_len: int = MAX_LEN,
) -> tuple:
    """
    Convert a list of replay timestep dicts to a (1, T_pad, N_FEATURES) tensor
    and a (1,) length tensor.
    """
    import numpy as np

    feats = []
    for t in timesteps[:max_len]:
        dist   = t.get("distance", 0.0)
        alt    = (t.get("agent_pos") or [0, 0, 1000])[2]
        a_h    = t.get("agent_health", 100.0)
        o_h    = t.get("opp_health",   100.0)
        reward = t.get("reward",        0.0)
        step   = t.get("t",             0)
        fired  = 1.0 if dist <= 600.0 else 0.0

        feats.append([
            dist / 15000.0,
            a_h  / 100.0,
            o_h  / 100.0,
            float(np.clip(reward / 30.0, -1.0, 1.0)),
            alt  / 2000.0,
            fired,
            step / 1000.0,
        ])

    T = len(feats)
    arr = np.zeros((max_len, N_FEATURES), dtype=np.float32)
    arr[:T] = feats
    x      = torch.from_numpy(arr).unsqueeze(0)           # (1, max_len, N_FEATURES)
    length = torch.tensor([T], dtype=torch.long)
    return x, length
