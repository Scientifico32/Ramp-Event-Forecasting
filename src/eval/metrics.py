"""
src/eval/metrics.py
────────────────────
All evaluation metrics reported in:

  Stergiou & Karakasidis (2026), "Ramp-Event Directional Forecasting
  for Wind Power Integration", Energies (MDPI), under review.

Primary metric (§5.2):
  directional_accuracy       — overall DA across all test steps
  ramp_directional_accuracy  — DA restricted to |Δpower| > RAMP_THRESHOLD
  stable_directional_accuracy— DA restricted to |Δpower| ≤ RAMP_THRESHOLD

Secondary metrics:
  mae                        — mean absolute error (p.u.)
  rmse                       — root mean squared error (p.u.)
  mape                       — mean absolute percentage error (%)
  pearson_r                  — Pearson correlation coefficient
  weighted_directional_accuracy (WDA) — DA weighted by |Δpower|
  trend_accuracy             — direction accuracy over 6-step (60-min) windows
  turning_point_accuracy (TPA)— fraction of turning points correctly detected

All inputs are 1-D numpy arrays of normalised power values in [0, 1].
"""

import numpy as np


# Ramp threshold: |Δpower| > 0.05 p.u. = ramp event
# Matches manuscript definition and pipeline RAMP_THRESHOLD constant
RAMP_THRESHOLD = 0.05


def compute_all_metrics(
    trues:  np.ndarray,
    preds:  np.ndarray,
    lasts:  np.ndarray,
    ramp_threshold: float = RAMP_THRESHOLD,
    tpa_tolerance:  int   = 1,
) -> dict:
    """
    Compute the full set of evaluation metrics used in the manuscript.

    Args:
        trues          : (N,) actual power at step t
        preds          : (N,) predicted power at step t
        lasts          : (N,) last observed power at step t-1 (= P(t-1))
        ramp_threshold : |Δpower| threshold for ramp-event classification
        tpa_tolerance  : ±steps tolerance for turning-point detection

    Returns:
        dict with keys matching all manuscript Table 2 columns:
          mae, rmse, mape, pearson_r,
          directional_accuracy, ramp_directional_accuracy,
          stable_directional_accuracy, weighted_directional_accuracy,
          trend_accuracy, turning_point_accuracy
    """
    trues = np.asarray(trues,  dtype=np.float64)
    preds = np.asarray(preds,  dtype=np.float64)
    lasts = np.asarray(lasts,  dtype=np.float64)

    N = len(trues)
    if N == 0:
        return _empty_metrics()

    # ── Magnitude metrics ─────────────────────────────────────────────────
    errors = np.abs(trues - preds)
    mae    = float(errors.mean())
    rmse   = float(np.sqrt(((trues - preds) ** 2).mean()))

    # MAPE: avoid division by zero for near-zero actual power
    nonzero = trues > 0.01
    mape = float(errors[nonzero].mean() / trues[nonzero].mean() * 100) if nonzero.any() else 0.0

    # Pearson r
    if trues.std() > 1e-8 and preds.std() > 1e-8:
        pearson_r = float(np.corrcoef(trues, preds)[0, 1])
    else:
        pearson_r = 0.0

    # ── Directional changes ───────────────────────────────────────────────
    delta_true = trues - lasts   # actual Δpower
    delta_pred = preds - lasts   # predicted Δpower

    sign_true  = np.sign(delta_true)
    sign_pred  = np.sign(delta_pred)

    # Correct direction: signs match AND neither is zero
    # (zero Δ = no direction; treated as incorrect to avoid inflation)
    correct_dir = (sign_true == sign_pred) & (sign_true != 0)

    # ── Overall DA ────────────────────────────────────────────────────────
    nonzero_mask = sign_true != 0
    if nonzero_mask.any():
        directional_accuracy = float(correct_dir[nonzero_mask].mean())
    else:
        directional_accuracy = 0.0

    # ── Regime-stratified DA ──────────────────────────────────────────────
    ramp_mask   = np.abs(delta_true) > ramp_threshold
    stable_mask = ~ramp_mask

    def _da_on_mask(mask):
        sub = nonzero_mask & mask
        if not sub.any():
            return float("nan")
        return float(correct_dir[sub].mean())

    ramp_directional_accuracy   = _da_on_mask(ramp_mask)
    stable_directional_accuracy = _da_on_mask(stable_mask)

    # ── Weighted DA (WDA) — each step weighted by |Δpower_true| ──────────
    weights = np.abs(delta_true)
    w_sum   = weights[nonzero_mask].sum()
    if w_sum > 1e-8:
        weighted_directional_accuracy = float(
            (weights[nonzero_mask] * correct_dir[nonzero_mask]).sum() / w_sum
        )
    else:
        weighted_directional_accuracy = 0.0

    # ── Trend accuracy (6-step windows = 60 min) ─────────────────────────
    window = 6
    trend_correct = []
    for i in range(0, N - window, window):
        seg_t = trues[i:i + window]
        seg_p = preds[i:i + window]
        seg_l = lasts[i:i + window]
        # Overall direction of the window: compare first and last
        trend_true = np.sign(seg_t[-1] - seg_l[0])
        trend_pred = np.sign(seg_p[-1] - seg_l[0])
        if trend_true != 0:
            trend_correct.append(trend_true == trend_pred)
    trend_accuracy = float(np.mean(trend_correct)) if trend_correct else 0.0

    # ── Turning-point accuracy (TPA) ─────────────────────────────────────
    # A turning point occurs where sign(Δpower) reverses.
    # TPA = fraction of actual turning points where the forecast also
    # reverses sign within ±tpa_tolerance steps.
    actual_tp  = _find_turning_points(delta_true)
    pred_tp    = _find_turning_points(delta_pred)

    if actual_tp:
        detected = 0
        for tp in actual_tp:
            # Check if any predicted turning point is within tolerance
            for ptp in pred_tp:
                if abs(tp - ptp) <= tpa_tolerance:
                    detected += 1
                    break
        turning_point_accuracy = float(detected / len(actual_tp))
    else:
        turning_point_accuracy = 0.0

    return {
        "mae":                        mae,
        "rmse":                       rmse,
        "mape":                       mape,
        "pearson_r":                  pearson_r,
        "directional_accuracy":       directional_accuracy,
        "ramp_directional_accuracy":  ramp_directional_accuracy,
        "stable_directional_accuracy":stable_directional_accuracy,
        "weighted_directional_accuracy": weighted_directional_accuracy,
        "trend_accuracy":             trend_accuracy,
        "turning_point_accuracy":     turning_point_accuracy,
        # Aliases used in some pipeline cells
        "DA":   directional_accuracy,
        "MAE":  mae,
        "RMSE": rmse,
    }


def _find_turning_points(delta: np.ndarray) -> list:
    """
    Return list of indices where sign(delta) reverses.
    Only non-zero deltas are considered (zero treated as continuation).
    """
    turning_points = []
    prev_sign = 0
    for i, d in enumerate(delta):
        s = np.sign(d)
        if s != 0:
            if prev_sign != 0 and s != prev_sign:
                turning_points.append(i)
            prev_sign = s
    return turning_points


def _empty_metrics() -> dict:
    keys = [
        "mae", "rmse", "mape", "pearson_r",
        "directional_accuracy", "ramp_directional_accuracy",
        "stable_directional_accuracy", "weighted_directional_accuracy",
        "trend_accuracy", "turning_point_accuracy",
        "DA", "MAE", "RMSE",
    ]
    return {k: float("nan") for k in keys}
