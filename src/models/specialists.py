"""
src/models/specialists.py
─────────────────────────
Five specialist deep learning models for the regime-stratified
wind power forecasting ensemble described in:

  Stergiou & Karakasidis, "Ramp-Event Directional Forecasting for
  Wind Power Integration: A Regime-Stratified Ensemble with
  Direction-Focused Training", Energies (MDPI), 2026, under review.

Each specialist is tuned to a distinct ramp-event regime:
  DailyPatternSpecialist   — sea-breeze / diurnal ramps   (4-layer BiLSTM)
  WeatherFrontSpecialist   — synoptic frontal passages     (Transformer encoder)
  MesoscaleSpecialist      — orographic channelling        (multi-scale CNN + LSTM)
  TurbulenceSpecialist     — sub-hourly turbulent gusts    (GRU + temporal attention)
  TrendSpecialist          — slow multi-hour ramps         (4-layer FeedForward)

All models share the same interface:
  __init__(lookback, forecast_steps, input_dim, **kwargs)
  forward(x) → (batch, forecast_steps)   [normalised power change]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ─────────────────────────────────────────────────────────────────────────────
# 1. DailyPatternSpecialist
#    Target regime : diurnal sea-breeze and thermal ramps
#    Architecture  : 4-layer bidirectional LSTM
# ─────────────────────────────────────────────────────────────────────────────
class DailyPatternSpecialist(nn.Module):
    """
    4-layer bidirectional LSTM for diurnal power ramp detection.
    Captures the periodic sea-breeze onset / decay pattern dominant
    at coastal Eastern Mediterranean sites.

    Args:
        lookback       : input sequence length (steps)
        forecast_steps : number of forecast steps (paper: 6 × 10 min)
        input_dim      : number of input features
        hidden         : hidden units per direction (default 128)
        dropout        : dropout probability (tuned per BEST_PARAMS)
    """
    def __init__(self, lookback: int, forecast_steps: int, input_dim: int,
                 hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.lookback       = lookback
        self.forecast_steps = forecast_steps

        # 4-layer bidirectional LSTM
        self.bilstm = nn.LSTM(
            input_size  = input_dim,
            hidden_size = hidden,
            num_layers  = 4,
            batch_first = True,
            dropout     = dropout,
            bidirectional = True,
        )
        self.norm = nn.LayerNorm(hidden * 2)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden * 2, forecast_steps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        out, _ = self.bilstm(x)          # (B, T, 2H)
        out    = self.norm(out[:, -1, :]) # last step, (B, 2H)
        out    = self.drop(out)
        return self.fc(out)              # (B, forecast_steps)


# ─────────────────────────────────────────────────────────────────────────────
# 2. WeatherFrontSpecialist
#    Target regime : synoptic frontal passages (largest-amplitude ramps)
#    Architecture  : Transformer encoder (6 layers, 8 heads)
# ─────────────────────────────────────────────────────────────────────────────
class WeatherFrontSpecialist(nn.Module):
    """
    Transformer encoder for synoptic frontal-passage ramp detection.
    Multi-head attention over the full lookback window captures the
    longer-range structure in the on-site wind-speed and power signals
    (and their engineered lag/rolling features) that tends to precede
    frontal ramps. The model uses only on-site SCADA-derived features;
    it does not ingest pressure, wind-direction, or any exogenous
    meteorological inputs.

    Args:
        lookback       : input sequence length (steps)
        forecast_steps : number of forecast steps
        input_dim      : number of input features
        hidden         : model dimension d_model (default 256)
        num_heads      : number of attention heads (default 8)
        ff_dim         : feed-forward inner dimension (default 256)
        num_layers     : number of encoder layers (default 6)
        dropout        : dropout probability
    """
    def __init__(self, lookback: int, forecast_steps: int, input_dim: int,
                 hidden: int = 256, num_heads: int = 8, ff_dim: int = 256,
                 num_layers: int = 6, dropout: float = 0.1):
        super().__init__()
        self.lookback       = lookback
        self.forecast_steps = forecast_steps

        # Project input features to d_model
        self.input_proj = nn.Linear(input_dim, hidden)

        # Sinusoidal positional encoding
        self.register_buffer(
            "pos_enc",
            self._make_pos_enc(lookback, hidden)
        )

        # Transformer encoder stack (6 layers, 8 heads)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = hidden,
            nhead           = num_heads,
            dim_feedforward = ff_dim,
            dropout         = dropout,
            batch_first     = True,
            norm_first      = True,     # pre-LN for training stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)

        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden, forecast_steps)

    @staticmethod
    def _make_pos_enc(max_len: int, d_model: int) -> torch.Tensor:
        pe  = torch.zeros(1, max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div[:d_model // 2])
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        h = self.input_proj(x) + self.pos_enc[:, :x.size(1), :]  # (B, T, H)
        h = self.encoder(h)                                        # (B, T, H)
        h = self.norm(h.mean(dim=1))                               # global avg pool
        h = self.drop(h)
        return self.fc(h)                                          # (B, forecast_steps)


# ─────────────────────────────────────────────────────────────────────────────
# 3. MesoscaleSpecialist
#    Target regime : orographic channelling / mesoscale sea-breeze interaction
#    Architecture  : multi-scale CNN (kernels 3,6,12) + 2-layer LSTM
# ─────────────────────────────────────────────────────────────────────────────
class MesoscaleSpecialist(nn.Module):
    """
    Multi-scale 1-D CNN followed by a 2-layer LSTM for mesoscale ramp detection.
    Parallel convolutional branches with kernels (3, 6, 12 steps) capture
    features at 30-min, 60-min, and 120-min scales simultaneously, matching
    the characteristic time scales of orographic gap-flow events.

    Args:
        lookback       : input sequence length (steps)
        forecast_steps : number of forecast steps
        input_dim      : number of input features
        hidden         : CNN output channels per branch; LSTM hidden units
        kernels        : tuple of 1-D kernel sizes for the CNN branches
        dropout        : dropout probability
    """
    def __init__(self, lookback: int, forecast_steps: int, input_dim: int,
                 hidden: int = 128, kernels: tuple = (3, 6, 12),
                 dropout: float = 0.2):
        super().__init__()
        self.lookback       = lookback
        self.forecast_steps = forecast_steps
        self.kernels        = kernels

        # Parallel CNN branches — one per kernel size
        self.conv_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_dim, hidden, kernel_size=k,
                          padding=k // 2),
                nn.GELU(),
                nn.BatchNorm1d(hidden),
            )
            for k in kernels
        ])

        # Merge branches → LSTM
        merged_dim = hidden * len(kernels)
        self.lstm = nn.LSTM(
            input_size  = merged_dim,
            hidden_size = hidden,
            num_layers  = 2,
            batch_first = True,
            dropout     = dropout,
        )
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden, forecast_steps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) → conv expects (B, F, T)
        xp = x.permute(0, 2, 1)

        branches = []
        for conv in self.conv_branches:
            out = conv(xp)              # (B, hidden, T')
            # Trim or pad to original T to allow concatenation
            T = x.size(1)
            out = out[:, :, :T]
            branches.append(out)

        merged = torch.cat(branches, dim=1)      # (B, hidden*n_kernels, T)
        merged = merged.permute(0, 2, 1)         # (B, T, hidden*n_kernels)

        lstm_out, _ = self.lstm(merged)          # (B, T, hidden)
        h = self.norm(lstm_out[:, -1, :])        # last step
        h = self.drop(h)
        return self.fc(h)                        # (B, forecast_steps)


# ─────────────────────────────────────────────────────────────────────────────
# 4. TurbulenceSpecialist
#    Target regime : sub-hourly turbulent gusts
#    Architecture  : 3-layer GRU + temporal self-attention
# ─────────────────────────────────────────────────────────────────────────────
class TurbulenceSpecialist(nn.Module):
    """
    3-layer GRU with temporal self-attention for turbulence-driven ramp detection.
    GRU is preferred over LSTM here for faster adaptation to rapid, high-frequency
    fluctuations. The attention layer weights timesteps by relevance, allowing the
    model to focus on the highest-turbulence intervals within the lookback window.

    Note: domain-filtered niche accuracy is below the 50% random baseline due to
    the fundamental stochasticity of turbulence at 10-minute resolution
    (Kolmogorov cascade). The specialist's primary operational role is
    prediction-interval inflation rather than directional point accuracy.

    Args:
        lookback       : input sequence length (steps)
        forecast_steps : number of forecast steps
        input_dim      : number of input features
        hidden         : GRU hidden units (default 64)
        dropout        : dropout probability
    """
    def __init__(self, lookback: int, forecast_steps: int, input_dim: int,
                 hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.lookback       = lookback
        self.forecast_steps = forecast_steps

        self.gru = nn.GRU(
            input_size  = input_dim,
            hidden_size = hidden,
            num_layers  = 3,
            batch_first = True,
            dropout     = dropout,
        )

        # Temporal self-attention: score each timestep
        self.attn_fc = nn.Linear(hidden, 1)

        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden, forecast_steps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        gru_out, _ = self.gru(x)           # (B, T, H)

        # Temporal attention weights
        scores  = self.attn_fc(gru_out)    # (B, T, 1)
        weights = torch.softmax(scores, dim=1)
        context = (weights * gru_out).sum(dim=1)  # (B, H)

        h = self.norm(context)
        h = self.drop(h)
        return self.fc(h)                  # (B, forecast_steps)


# ─────────────────────────────────────────────────────────────────────────────
# 5. TrendSpecialist
#    Target regime : slow multi-hour ramps from approaching weather systems
#    Architecture  : 4-layer feed-forward network (MLP with skip connections)
# ─────────────────────────────────────────────────────────────────────────────
class TrendSpecialist(nn.Module):
    """
    4-layer feed-forward network for slow multi-hour trend ramp detection.
    The MLP flattens the full lookback window and learns to identify slow-moving
    power trends associated with approaching weather systems over 12–24 hour
    horizons. Skip connections between adjacent layers improve gradient flow.

    Args:
        lookback       : input sequence length (steps)
        forecast_steps : number of forecast steps
        input_dim      : number of input features
        hidden         : hidden layer width (default 256)
        dropout        : dropout probability
    """
    def __init__(self, lookback: int, forecast_steps: int, input_dim: int,
                 hidden: int = 256, dropout: float = 0.15):
        super().__init__()
        self.lookback       = lookback
        self.forecast_steps = forecast_steps

        flat_dim = lookback * input_dim

        # Project flattened input to hidden
        self.input_proj = nn.Linear(flat_dim, hidden)

        # 3 residual FF blocks (input_proj + 3 = 4 total linear layers)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
            )
            for _ in range(3)
        ])

        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden, forecast_steps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) → flatten → (B, T*F)
        B = x.size(0)
        h = x.reshape(B, -1)
        h = F.gelu(self.input_proj(h))     # (B, hidden)

        for block in self.blocks:
            h = h + block(h)               # skip connection

        h = self.drop(h)
        return self.fc(h)                  # (B, forecast_steps)
