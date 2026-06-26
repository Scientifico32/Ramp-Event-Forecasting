"""
src/losses/multiobjective.py
─────────────────────────────
Multi-objective direction-focused loss for ramp-event forecasting.

    L = α·L_mag + β·L_dir + γ·L_cons + δ·L_phys

where:
  L_mag  — mean squared error on the predicted power (magnitude accuracy).
  L_dir  — margin-based directional hinge. Let
               Δ_pred   = pred_h    − pred_last
               Δ_target = target_h  − target_last
           and define the agreement product  p = Δ_pred · Δ_target.
           The term is  ReLU(margin − p)  with a fixed margin (default 0.1):
           it is zero once the predicted and observed changes share the same
           sign by at least the margin, and grows linearly otherwise. This
           penalises errors in the *direction* of the power change without the
           magnitude weighting of a regression loss.
  L_cons — temporal-consistency term: MSE between the first differences of the
           predicted and target sequences, discouraging implausible
           step-to-step oscillation in the multi-step forecast.
  L_phys — physics term, retained in the signature for API stability but
           hardcoded to zero in this configuration (δ multiplies a zero
           tensor), i.e. the model is trained without an explicit physics
           penalty.

This is a direction-focused loss: the directional component is a margin hinge
on the sign agreement of the power change, NOT a magnitude-weighted or
cross-entropy term.

Reference:
  Stergiou & Karakasidis (2026), Energies (MDPI), under review.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def multi_objective_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    pred_last: torch.Tensor,
    target_last: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 0.2,
    delta: float = 0.1,
) -> tuple:
    """
    Compute the multi-objective direction-focused loss.

    Args:
        pred         : (B, H) predicted power sequence (H = forecast steps).
        target       : (B, H) actual power sequence.
        pred_last    : (B,) last observed power value used as the reference
                       for the predicted change.
        target_last  : (B,) last observed power value used as the reference
                       for the observed change.
        alpha        : weight for the magnitude (MSE) term.
        beta         : weight for the directional hinge term.
        gamma        : weight for the temporal-consistency term.
        delta         : weight for the physics term (multiplies a zero tensor
                       in this configuration; kept for API stability).

    Returns:
        (total_loss, parts_dict) where parts_dict has keys
        'mag', 'dir', 'cons', 'phys'.
    """
    # Normalise shapes to (B, H)
    pred = pred.squeeze(-1) if pred.dim() == 3 else pred
    target = target.squeeze(-1) if target.dim() == 3 else target
    if pred.dim() == 1:
        pred = pred.unsqueeze(-1)
    if target.dim() == 1:
        target = target.unsqueeze(-1)

    pred_h = pred[:, 0]
    target_h = target[:, 0]

    # ── 1. Magnitude (MSE) ────────────────────────────────────────────────
    L_mag = F.mse_loss(pred_h, target_h)

    # ── 2. Directional hinge ──────────────────────────────────────────────
    delta_pred = pred_h - pred_last
    delta_target = target_h - target_last
    product = delta_pred * delta_target
    margin = 0.1
    L_dir = F.relu(margin - product).mean()

    # ── 3. Temporal consistency ───────────────────────────────────────────
    L_cons = torch.tensor(0.0, device=pred.device)
    if pred.size(1) > 1:
        pred_diff = pred[:, 1:] - pred[:, :-1]
        target_diff = target[:, 1:] - target[:, :-1]
        L_cons = F.mse_loss(pred_diff, target_diff)

    # ── 4. Physics term (hardcoded to zero in this configuration) ─────────
    L_phys = torch.tensor(0.0, device=pred.device)

    total = alpha * L_mag + beta * L_dir + gamma * L_cons + delta * L_phys
    return total, {
        "mag":  L_mag.item(),
        "dir":  L_dir.item(),
        "cons": L_cons.item(),
        "phys": L_phys.item(),
    }
