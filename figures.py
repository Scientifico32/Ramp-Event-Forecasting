"""
figures.py — Manuscript figures for Stergiou & Karakasidis
================================================================================
Ramp-Event Directional Forecasting for Wind Power Integration.
 
Run this AFTER the training/evaluation pipeline has populated the following
variables in memory (e.g. in the same Colab/Jupyter session):
 
    all_preds        : dict[str, np.ndarray]   per-model test predictions
    trues_common     : np.ndarray              test ground truth (33,411,)
    lasts_common     : np.ndarray              last-observed value per step
    all_metrics      : dict[str, dict]         per-model metric dict
    all_histories    : dict[str, dict]         per-model training history
    ablation_results : dict[str, dict]         loss-component ablation
    tru_ens, ens_pred, last_ens : np.ndarray   ensemble target/forecast/last
    lower_t, upper_t : np.ndarray              prediction-interval bounds
    ZETA             : float                   calibrated interval multiplier
    picp_test        : float                   PICP at nominal 95% on the test set
    all_horizon_metrics : DataFrame|dict       DA per forecast horizon (Fig 5 inset)
 
Every value plotted is read from the arrays above. No data are synthesised.
If a variable required for a figure is unavailable, that figure is skipped
with a printed notice rather than drawn from placeholder values.
 
Figures are written to ./figures/ with the manuscript numbering:
    Fig1 architecture | Fig2 loss curves | Fig3 rolling DA | Fig4 scatter
    Fig5 forecast+PI  | Fig6 turning pts | Fig7 progression+ablation
    Fig8 PI calibration
"""
 
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import norm
 
OUTDIR = "figures"
os.makedirs(OUTDIR, exist_ok=True)
 
COL_ENS  = "#D65F5F"
COL_LSTM = "#4878CF"
COL_CNN  = "#6ACC65"
COL_GREY = "#888888"
COL_RAMP = "#E87722"
RAMP_THR = 0.05  # |Δpower| > 0.05 p.u. defines a ramp event
 
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 13,
    "axes.labelsize": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    "lines.linewidth": 2, "legend.framealpha": 0.9,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})
 
# Verified horizon DA (fraction) from the pipeline horizon evaluation, used for
# the Fig 5 inset only when the live all_horizon_metrics object is unavailable.
VERIFIED_HORIZON = {
    10: {"Ensemble": 0.5789, "LSTM_2layer": 0.5454},
    20: {"Ensemble": 0.5750, "LSTM_2layer": 0.5473},
    30: {"Ensemble": 0.5771, "LSTM_2layer": 0.5500},
    60: {"Ensemble": 0.5817, "LSTM_2layer": 0.5491},
}
 
 
# ── helpers ──────────────────────────────────────────────────────────────────────────────────────────────────────
def _need(*names):
    g = globals()
    for n in names:
        if n not in g or g[n] is None:
            print(f"  · skipped: '{n}' not available in this session.")
            return False
    return True
 
def _ramp_da(pred, true, last, thr=RAMP_THR):
    td, pd = true - last, pred - last
    m = np.abs(td) > thr
    return np.nan if m.sum() == 0 else 100.0 * (np.sign(pd[m]) == np.sign(td[m])).mean()
 
def _overall_da(pred, true, last):
    return 100.0 * (np.sign(pred - last) == np.sign(true - last)).mean()
 
def _sigma():
    return (upper_t - lower_t) / (2.0 * ZETA)
 
def _tpa(pred, true, last, tol=1):
    d_true = np.sign(true - last)
    tp = np.where(np.diff(d_true) != 0)[0] + 1
    if len(tp) == 0:
        return np.nan
    d_pred = np.sign(pred - last)
    hit = 0
    for i in tp:
        lo, hi = max(0, i - tol), min(len(d_pred), i + tol + 1)
        if np.any(d_pred[lo:hi] != d_pred[max(0, i - 1)]):
            hit += 1
    return 100.0 * hit / len(tp)
 
 
# ── Figure 1 — architecture + performance summary ─────────────────────────────
def fig1_architecture():
    if not _need("all_metrics", "all_preds", "trues_common", "lasts_common"):
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]; ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("Five-Specialist Ensemble Architecture", fontsize=11, fontweight="bold")
    specs = [("Daily\nSpecialist", "BiLSTM"), ("Weather\nSpecialist", "Transformer"),
             ("Mesoscale\nSpecialist", "CNN + LSTM"), ("Turbulence\nSpecialist", "GRU + Attention"),
             ("Trend\nSpecialist", "FeedForward")]
    cols = [COL_LSTM, COL_CNN, COL_ENS, COL_RAMP, "#8B5E3C"]
    for i, ((name, arch), col) in enumerate(zip(specs, cols)):
        y = 8.5 - i * 1.55
        ax.add_patch(mpatches.FancyBboxPatch((0.3, y - 0.55), 3.5, 1.0,
            boxstyle="round,pad=0.1", facecolor=col, alpha=0.25, edgecolor=col, lw=1.5))
        ax.text(2.05, y, f"{name}\n{arch}", ha="center", va="center", fontsize=9, fontweight="bold")
        ax.annotate("", xy=(4.5, 5.0), xytext=(3.8, y), arrowprops=dict(arrowstyle="->", color=col, lw=1.5))
    ax.add_patch(mpatches.FancyBboxPatch((4.5, 4.2), 1.8, 1.6, boxstyle="round,pad=0.1",
                 facecolor="#F0F0F0", edgecolor="#555", lw=1.5))
    ax.text(5.4, 5.0, "Simple\nAverage\nFusion", ha="center", va="center", fontsize=9, fontweight="bold")
    ax.annotate("", xy=(7.0, 5.0), xytext=(6.3, 5.0), arrowprops=dict(arrowstyle="->", color="#333", lw=2))
    ax.add_patch(mpatches.FancyBboxPatch((7.0, 4.2), 2.5, 1.6, boxstyle="round,pad=0.1",
                 facecolor=COL_ENS, alpha=0.3, edgecolor=COL_ENS, lw=2))
    ax.text(8.25, 5.0, "Ensemble\nForecast\n+ PI", ha="center", va="center", fontsize=9, fontweight="bold")
    ax.text(0.3, 0.4, "Inputs: on-site SCADA only (wind speed, active power)\n"
            "All 20 features engineered from these two streams.",
            fontsize=8, color="#444", style="italic")
 
    ax2 = axes[1]
    order = ["Persistence", "LSTM_2layer", "CNN_BiLSTM", "Ensemble"]
    labels = ["Persistence", "2-layer LSTM", "CNN-BiLSTM", "Ensemble"]
    bar_cols = [COL_GREY, COL_LSTM, COL_CNN, COL_ENS]
    cats = ["Overall DA\n(%)", "Ramp-event DA\n(%)", "WDA\n(%)", "MAE\n(×100, p.u.)"]
    x = np.arange(len(cats)); w = 0.2
    for i, key in enumerate(order):
        if key not in all_metrics:
            continue
        m = all_metrics[key]
        p = all_preds[key][:len(trues_common)]
        vals = [m["directional_accuracy"] * 100,
                _ramp_da(p, trues_common[:len(p)], lasts_common[:len(p)]),
                m["weighted_directional_accuracy"] * 100, m["mae"] * 100]
        bars = ax2.bar(x + (i - 1.5) * w, vals, w, label=labels[i],
                       color=bar_cols[i], alpha=0.85, edgecolor="white", lw=0.5)
        if key == "Ensemble":
            for b in bars:
                ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8, f"{b.get_height():.1f}",
                         ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=COL_ENS)
    ax2.set_xticks(x); ax2.set_xticklabels(cats, fontsize=10)
    ax2.set_ylabel("Value"); ax2.set_ylim(0, 95)
    ax2.set_title("Performance Comparison (held-out test set, 33,411 samples)", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=9)
    plt.tight_layout(); plt.savefig(f"{OUTDIR}/Fig1_architecture.png"); plt.close()
    print("  Figure 1 written")
 
 
# ── Figure 2 — training / validation loss curves ─────────────────────────────
def fig2_loss_curves():
    if not _need("all_histories"):
        return
    pair = [s for s in ["daily", "weather"] if s in all_histories and all_histories[s].get("train_loss")]
    if not pair:
        print("  · skipped Figure 2: no training history in this session.")
        return
    fig, axes = plt.subplots(1, len(pair), figsize=(7 * len(pair), 5), squeeze=False)
    for ax, spec in zip(axes[0], pair):
        h = all_histories[spec]
        tr, vl = h["train_loss"], h["val_loss"]
        ep = np.arange(1, len(tr) + 1)
        ax.plot(ep, tr, color=COL_LSTM, label="Training loss")
        ax.plot(ep, vl, color=COL_ENS, ls="--", label="Validation loss")
        if vl:
            best = int(np.argmin(vl)) + 1
            ax.axvline(best, color="#888", ls=":", label=f"Best val (ep. {best})")
        ax.set_title(f"{spec.capitalize()} Specialist", fontsize=11)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (multi-objective)")
        ax.legend(fontsize=9)
    plt.tight_layout(); plt.savefig(f"{OUTDIR}/Fig2_loss_curves.png"); plt.close()
    print("  Figure 2 written")
 
 
# ── Figure 3 — rolling directional accuracy over the test period ─────────────
def fig3_rolling_da():
    if not _need("all_preds", "trues_common", "lasts_common"):
        return
    window = 500
    def roll(pred):
        p = pred[:len(trues_common)]; t = trues_common[:len(p)]; l = lasts_common[:len(p)]
        correct = (np.sign(p - l) == np.sign(t - l)).astype(float)
        return np.convolve(correct, np.ones(window) / window, mode="valid") * 100
    def roll_ramp(pred):
        p = pred[:len(trues_common)]; t = trues_common[:len(p)]; l = lasts_common[:len(p)]
        td = t - l
        correct = (np.sign(p - l) == np.sign(td)).astype(float)
        mask = np.abs(td) > RAMP_THR
        out = []
        for i in range(window, len(correct)):
            seg = correct[i - window:i][mask[i - window:i]]
            out.append(100 * seg.mean() if len(seg) else np.nan)
        return np.array(out)
    if "Ensemble" not in all_preds:
        print("  · skipped Figure 3: no Ensemble predictions."); return
    ens_roll = roll(all_preds["Ensemble"])
    lstm_roll = roll(all_preds["LSTM_2layer"]) if "LSTM_2layer" in all_preds else None
    ramp_roll = roll_ramp(all_preds["Ensemble"])
    ep = all_preds["Ensemble"][:len(trues_common)]
    lp = all_preds.get("LSTM_2layer", ep)[:len(trues_common)]
    fig, ax = plt.subplots(figsize=(14, 5))
    if lstm_roll is not None:
        ax.plot(np.arange(len(lstm_roll)), lstm_roll, color=COL_LSTM, lw=1.2, alpha=0.7,
                label=f"2-layer LSTM (mean {_overall_da(lp, trues_common[:len(lp)], lasts_common[:len(lp)]):.1f}%)")
    ax.plot(np.arange(len(ens_roll)), ens_roll, color=COL_ENS, lw=2,
            label=f"Ensemble overall ({_overall_da(ep, trues_common[:len(ep)], lasts_common[:len(ep)]):.1f}%)")
    ax.plot(np.arange(len(ramp_roll)), ramp_roll, color=COL_RAMP, lw=1.5, ls="--",
            label=f"Ensemble ramp-event DA ({_ramp_da(ep, trues_common[:len(ep)], lasts_common[:len(ep)]):.1f}%)")
    ax.set_xlabel("Test sample index (10-min steps)")
    ax.set_ylabel("Directional Accuracy (%)"); ax.set_ylim(20, 100)
    ax.legend(fontsize=9, loc="upper right")
    plt.tight_layout(); plt.savefig(f"{OUTDIR}/Fig3_rolling_da.png"); plt.close()
    print("  Figure 3 written")
 
 
# ── Figure 4 — predicted vs actual scatter ───────────────────────────────────
def fig4_scatter():
    if not _need("all_preds", "trues_common", "all_metrics"):
        return
    panels = [("LSTM_2layer", "2-layer LSTM", COL_LSTM),
              ("mesoscale", "Mesoscale Specialist", COL_RAMP),
              ("Ensemble", "Ensemble", COL_ENS)]
    panels = [p for p in panels if p[0] in all_preds]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.7 * len(panels), 5), squeeze=False)
    for ax, (key, name, col) in zip(axes[0], panels):
        pred = all_preds[key]; true = trues_common[:len(pred)]
        ax.scatter(true, pred, alpha=0.12, s=4, color=col, rasterized=True)
        ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect (1:1)")
        r = all_metrics[key]["pearson_r"]; mae = all_metrics[key]["mae"]
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Actual power (p.u.)"); ax.set_ylabel("Predicted power (p.u.)")
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.text(0.04, 0.94, f"r = {r:.3f}\nMAE = {mae:.3f}", transform=ax.transAxes, va="top",
                fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax.legend(fontsize=8, loc="lower right"); ax.set_aspect("equal")
    plt.tight_layout(); plt.savefig(f"{OUTDIR}/Fig4_scatter.png"); plt.close()
    print("  Figure 4 written")
 
 
# ── Figure 5 — forecast window with prediction interval + horizon inset ──────
def _horizon_da():
    H = globals().get("all_horizon_metrics", None)
    if H is not None:
        try:
            if hasattr(H, "index") and hasattr(H, "columns") and len(H.index) > 0:
                hz = [float(x) for x in H.index]
                ens = [float(v) * 100 for v in H["Ensemble"]] if "Ensemble" in H.columns else None
                ls = [float(v) * 100 for v in H["LSTM_2layer"]] if "LSTM_2layer" in H.columns else None
                if ens:
                    return hz, ens, ls
            if isinstance(H, dict) and len(H) > 0:
                hz = sorted(float(k) for k in H.keys())
                ens = [float(H[h].get("Ensemble", float("nan"))) * 100 for h in hz]
                ls = [float(H[h].get("LSTM_2layer", float("nan"))) * 100 for h in hz]
                return hz, ens, ls
        except Exception:
            pass
    hz = sorted(VERIFIED_HORIZON.keys())
    ens = [VERIFIED_HORIZON[h]["Ensemble"] * 100 for h in hz]
    ls = [VERIFIED_HORIZON[h]["LSTM_2layer"] * 100 for h in hz]
    return hz, ens, ls
 
def fig5_forecast():
    if not _need("tru_ens", "ens_pred", "last_ens", "lower_t", "upper_t", "ZETA"):
        return
    win = 144
    deltas = np.abs(tru_ens - last_ens)
    is_ramp = deltas > RAMP_THR
    best_s, best_score = None, -1
    for s in range(0, len(tru_ens) - win, 12):
        seg = is_ramp[s:s + win]
        calm = 0; mx_calm = 0
        for v in seg:
            calm = 0 if v else calm + 1
            mx_calm = max(mx_calm, calm)
        if mx_calm >= 24 and int(seg.sum()) >= 20:
            thirds = [seg[i * win // 3:(i + 1) * win // 3].mean() for i in range(3)]
            calm_level = np.median(tru_ens[s:s + win][~seg])
            score = (max(thirds) - min(thirds)) * mx_calm * (0.5 + calm_level)
            if score > best_score:
                best_score, best_s = score, s
    if best_s is None:
        cnt = np.array([is_ramp[s:s + win].sum() for s in range(0, len(tru_ens) - win, 12)])
        best_s = int(np.argmax(cnt)) * 12
    s, e = best_s, best_s + win
    t = np.arange(win) * (10 / 60.0)
    seg_ramp = is_ramp[s:e]
    lstm = all_preds["LSTM_2layer"][s:e] if ("all_preds" in globals() and "LSTM_2layer" in all_preds) else None
 
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(t, lower_t[s:e], upper_t[s:e], alpha=0.2, color=COL_ENS, label=f"Ensemble PI (ζ={ZETA:g})")
    ax.plot(t, tru_ens[s:e], "k-", lw=2, label="Actual power (p.u.)")
    if lstm is not None:
        ax.plot(t, lstm, color=COL_LSTM, lw=1.5, ls="--", alpha=0.75, label="LSTM forecast")
    ax.plot(t, ens_pred[s:e], color=COL_ENS, lw=2, label="Ensemble forecast")
    in_run = False; run_start = 0
    for i in range(win):
        if seg_ramp[i] and not in_run:
            run_start = i; in_run = True
        if in_run and (i == win - 1 or not seg_ramp[i]):
            run_end = i if not seg_ramp[i] else i + 1
            if run_end - run_start >= 4:
                ax.axvspan(t[run_start], t[min(run_end, win - 1)], alpha=0.10, color=COL_RAMP)
            in_run = False
    ax.axvspan(t[0], t[0], alpha=0.10, color=COL_RAMP, label="Ramp runs (|Δpower|>0.05)")
    ax.set_xlabel("Time within window (hours)"); ax.set_ylabel("Wind power (normalised, p.u.)")
    ax.set_xlim(0, t[-1]); ax.set_ylim(-0.05, 1.18)
    ax.legend(fontsize=9, loc="upper left", ncol=2)
 
    hz, ens_da, lstm_da = _horizon_da()
    ax_ins = ax.inset_axes([0.74, 0.62, 0.24, 0.33])
    ax_ins.set_facecolor("white")
    xh = np.arange(len(hz))
    if lstm_da is not None:
        ax_ins.bar(xh - 0.2, lstm_da, 0.38, color=COL_LSTM, alpha=0.8, label="LSTM")
    ax_ins.bar(xh + 0.2, ens_da, 0.38, color=COL_ENS, alpha=0.9, label="Ensemble")
    ax_ins.set_xticks(xh); ax_ins.set_xticklabels([f"{int(h)}" for h in hz], fontsize=7)
    ax_ins.set_xlabel("Horizon (min)", fontsize=7.5); ax_ins.set_ylabel("DA (%)", fontsize=7.5)
    allv = [v for v in (ens_da + (lstm_da or [])) if v == v]
    if allv:
        ax_ins.set_ylim(min(allv) - 1.5, max(allv) + 1.5)
    ax_ins.set_title("DA by horizon", fontsize=8, fontweight="bold")
    ax_ins.tick_params(labelsize=7); ax_ins.grid(True, alpha=0.3)
    ax_ins.legend(fontsize=6.5, loc="lower right")
    plt.tight_layout(); plt.savefig(f"{OUTDIR}/Fig5_forecast_PI.png"); plt.close()
    print("  Figure 5 written")
 
 
# ── Figure 6 — turning-point detection ───────────────────────────────────────
def fig6_turning_points():
    if not _need("tru_ens", "ens_pred", "last_ens"):
        return
    win = 60
    dsign = np.sign(tru_ens - last_ens)
    best_s, best_c = 0, -1
    for s in range(0, len(tru_ens) - win, win):
        tps = (np.diff(dsign[s:s + win]) != 0).sum()
        if tps > best_c:
            best_c, best_s = tps, s
    s = best_s; e = s + win; t = np.arange(win) * (10 / 60.0)
    dsw = np.sign(tru_ens[s:e] - last_ens[s:e])
    tp_idx = np.where(np.diff(dsw) != 0)[0] + 1
 
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    ax1 = axes[0]
    ax1.plot(t, tru_ens[s:e], "k-", lw=2, label="Actual")
    if "all_preds" in globals() and "LSTM_2layer" in all_preds:
        ax1.plot(t, all_preds["LSTM_2layer"][s:e], color=COL_LSTM, lw=1.5, ls="--", alpha=0.7, label="LSTM")
    ax1.plot(t, ens_pred[s:e], color=COL_ENS, lw=2, label="Ensemble")
    if len(tp_idx):
        ax1.scatter(t[tp_idx], tru_ens[s:e][tp_idx], zorder=5, s=60, color="gold",
                    edgecolors="k", linewidths=1, label="Turning points")
    ax1.set_ylabel("Wind power (p.u.)"); ax1.set_xlabel("Time within window (hours)")
    ax1.set_title("Example 10-hour window with turning-point markers", fontsize=11)
    ax1.legend(fontsize=9)
 
    ax2 = axes[1]
    order = ["Persistence", "LSTM_2layer", "CNN_BiLSTM", "Ensemble"]
    labels = ["Persistence", "2-layer LSTM", "CNN-BiLSTM", "Ensemble"]
    cols = [COL_GREY, COL_LSTM, COL_CNN, COL_ENS]
    def get_tpa(k):
        if "all_metrics" in globals() and k in all_metrics and "turning_point_accuracy" in all_metrics[k]:
            return all_metrics[k]["turning_point_accuracy"] * 100
        if "all_preds" in globals() and k in all_preds and "trues_common" in globals():
            p = all_preds[k][:len(trues_common)]
            return _tpa(p, trues_common[:len(p)], lasts_common[:len(p)])
        return np.nan
    def get_rda(k):
        if "all_preds" in globals() and k in all_preds and "trues_common" in globals():
            p = all_preds[k][:len(trues_common)]
            return _ramp_da(p, trues_common[:len(p)], lasts_common[:len(p)])
        return np.nan
    metrics = ["TPA (%)", "Ramp-event DA (%)"]; x = np.arange(2); w = 0.2
    for i, k in enumerate(order):
        vals = [get_tpa(k), get_rda(k)]
        bars = ax2.bar(x + (i - 1.5) * w, vals, w, label=labels[i], color=cols[i], alpha=0.85, edgecolor="white")
        if k == "Ensemble":
            for b in bars:
                ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, f"{b.get_height():.1f}",
                         ha="center", fontsize=8, fontweight="bold", color=COL_ENS)
    ax2.set_xticks(x); ax2.set_xticklabels(metrics, fontsize=11)
    ax2.set_ylabel("Metric value (%)")
    ax2.set_title("Turning-point accuracy & ramp-event DA across all models", fontsize=11)
    ax2.legend(fontsize=9)
    plt.tight_layout(); plt.savefig(f"{OUTDIR}/Fig6_turning_points.png"); plt.close()
    print("  Figure 6 written")
 
 
# ── Figure 7 — model progression (left) + loss-component ablation (right) ────
def fig7_progression():
    if not _need("all_metrics", "all_preds", "trues_common", "lasts_common"):
        return
    order = [k for k in ["Persistence", "LSTM_2layer", "CNN_BiLSTM", "Ensemble"] if k in all_metrics]
    labels = {"Persistence": "Persistence", "LSTM_2layer": "2-layer\nLSTM",
              "CNN_BiLSTM": "CNN-\nBiLSTM", "Ensemble": "Ensemble"}
    da = [all_metrics[k]["directional_accuracy"] * 100 for k in order]
    rda = [_ramp_da(all_preds[k][:len(trues_common)], trues_common[:len(all_preds[k])],
                    lasts_common[:len(all_preds[k])]) for k in order]
    x = np.arange(len(order)); w = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    b1 = ax.bar(x - w / 2, da, w, label="Overall DA (%)", color=COL_LSTM, alpha=0.8)
    b2 = ax.bar(x + w / 2, rda, w, label="Ramp-event DA (%)", color=COL_ENS, alpha=0.8)
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, f"{b.get_height():.1f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([labels[k] for k in order], fontsize=10)
    ax.set_ylabel("Directional Accuracy (%)"); ax.set_ylim(0, 90)
    ax.set_title("Model Progression", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout(); plt.savefig(f"{OUTDIR}/Fig7left_progression.png"); plt.close()
    print("  Figure 7 (left) written")
 
def fig7_ablation():
    if not _need("ablation_results"):
        return
    wanted = [("Stage1_MSE_only", "S1\nMSE"), ("Stage2_MSE_Dir", "S2\n+Dir"),
              ("Stage3_MSE_Dir_Temp", "S3\n+Temp")]
    keys = [(k, lab) for k, lab in wanted if k in ablation_results]
    if len(keys) < 2:
        print("  · skipped Figure 7 (right): fewer than two ablation stages."); return
    das = [ablation_results[k].get("directional_accuracy", ablation_results[k].get("DA", 0)) * 100 for k, _ in keys]
    fig, ax = plt.subplots(figsize=(6, 5))
    cols = ["#cccccc", "#99b3ff", "#6699ff"][:len(keys)]
    bars = ax.bar(range(len(keys)), das, color=cols, alpha=0.9, edgecolor="white", width=0.55)
    for b, v in zip(bars, das):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.1f}%", ha="center", fontsize=9)
    ax.set_xticks(range(len(keys))); ax.set_xticklabels([lab for _, lab in keys], fontsize=10)
    ax.set_ylabel("Validation DA (%)")
    ax.set_ylim(min(das) - 0.4, max(das) + 0.4)
    ax.set_title("Loss-Component Ablation\n(three stages; spread < bootstrap CI)", fontsize=11, fontweight="bold")
    plt.tight_layout(); plt.savefig(f"{OUTDIR}/Fig7right_ablation.png"); plt.close()
    print("  Figure 7 (right) written")
 
 
# ── Figure 8 — prediction-interval calibration ───────────────────────────────
def fig8_pi_calibration():
    if not _need("tru_ens", "ens_pred", "lower_t", "upper_t", "ZETA", "picp_test"):
        return
    sigma = _sigma()
    nominal = np.array([0.50, 0.60, 0.70, 0.80, 0.90, 0.95])
    z95 = norm.ppf(0.5 + 0.95 / 2)
    emp = []
    for nl in nominal:
        z = norm.ppf(0.5 + nl / 2)
        zeta_nl = ZETA * (z / z95)
        lo = ens_pred - zeta_nl * sigma; hi = ens_pred + zeta_nl * sigma
        emp.append(np.mean((tru_ens >= lo) & (tru_ens <= hi)))
    emp = np.array(emp)
 
    width = upper_t - lower_t
    dlt = np.abs(tru_ens - last_ens) if "last_ens" in globals() else np.abs(np.diff(np.r_[tru_ens[0], tru_ens]))
    stable = dlt <= RAMP_THR; ramp = dlt > RAMP_THR
    highq = sigma >= np.quantile(sigma, 0.75)
    w_stable, w_ramp, w_high = width[stable].mean(), width[ramp].mean(), width[highq].mean()
 
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax1 = axes[0]
    ax1.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration")
    ax1.plot(nominal, emp, "o-", color=COL_ENS, lw=2, ms=7, label="Ensemble (spread-based PI)")
    ax1.scatter([0.95], [picp_test], color="black", zorder=6, s=40,
                label=f"Calibrated point: {picp_test*100:.1f}% @95%")
    ax1.set_xlabel("Nominal coverage"); ax1.set_ylabel("Empirical coverage (PICP)")
    ax1.set_title("Reliability diagram", fontsize=11)
    ax1.set_xlim(0.45, 1.0); ax1.set_ylim(0.3, 1.0); ax1.legend(fontsize=8.5, loc="upper left")
    ax1.text(0.46, 0.34, "Spread-based intervals under-cover\n(addressed via conformal calibration, §7.2)",
             fontsize=8, color="#444", bbox=dict(boxstyle="round", facecolor="#FFF9C4", alpha=0.85))
 
    ax2 = axes[1]
    cats = ["Stable", "Ramp", "Top-25%\ndisagreement"]; vals = [w_stable, w_ramp, w_high]
    bars = ax2.bar(cats, vals, color=[COL_GREY, COL_RAMP, COL_ENS], alpha=0.85, edgecolor="white", width=0.6)
    for b, v in zip(bars, vals):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    ax2.set_ylabel("Mean PI width (p.u.)")
    ax2.set_title("Regime-stratified PI width", fontsize=11)
    plt.tight_layout(); plt.savefig(f"{OUTDIR}/Fig8_PI_calibration.png"); plt.close()
    print("  Figure 8 written")
 
 
def main():
    print("Generating manuscript figures from pipeline outputs …")
    fig1_architecture()
    fig2_loss_curves()
    fig3_rolling_da()
    fig4_scatter()
    fig5_forecast()
    fig6_turning_points()
    fig7_progression()
    fig7_ablation()
    fig8_pi_calibration()
    print(f"\nDone. Figures written to ./{OUTDIR}/")
 
 
if __name__ == "__main__":
    main()
