# ============================================================
# Wind Power Forecasting — Enhanced Full Pipeline (Colab)
# ============================================================
# Train specialists → Baselines → Ablation → Multi-Horizon
# → Niche Accuracy → Uncertainty Calibration → Statistical
# Tests → Publication Figures → Paper Snippets
#
# Runtime → Change runtime type → A100 GPU
# ============================================================

# ─────────────────────────────────────────────────────────────
# CELL 0 · Checkpoint Status Dashboard
# ─────────────────────────────────────────────────────────────
# Run this standalone BEFORE anything else to see exactly where
# training stands after a disconnection. READ-ONLY — safe to
# run at any time without affecting models or checkpoints.
# ─────────────────────────────────────────────────────────────
from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path
import torch

_DRIVE_PROJECT_PATH = "/content/drive/MyDrive/PhD/STERGIOU/PhD/wind_paper"
_OUTPUT_DIR  = Path(_DRIVE_PROJECT_PATH) / "outputs" / "colab_full"
_CKPT_DIR    = _OUTPUT_DIR / "checkpoints"

_SPECIALISTS = ["daily", "weather", "mesoscale", "turbulence", "trend"]
_BASELINES   = ["LSTM_2layer", "CNN_BiLSTM"]
_ALL_LABELS  = {s: "specialist" for s in _SPECIALISTS}
_ALL_LABELS.update({b: "baseline" for b in _BASELINES})

_EPOCHS = {"specialist": 1000, "baseline": 300}
_COL_DONE   = "\033[92m"   # green
_COL_RESUME = "\033[93m"   # yellow
_COL_NONE   = "\033[91m"   # red
_COL_RESET  = "\033[0m"

print("\n" + "=" * 62)
print("  CHECKPOINT STATUS DASHBOARD")
print("=" * 62)

if not _CKPT_DIR.exists():
    print("  checkpoints/ directory not found — no training started yet.")
else:
    for label, kind in _ALL_LABELS.items():
        resume_file = _CKPT_DIR / f"resume_{label}.pt"
        best_file   = _OUTPUT_DIR / f"best_{label}_{kind}.pt"
        total_eps   = _EPOCHS[kind]

        if resume_file.exists():
            state    = torch.load(resume_file, map_location="cpu")
            ep       = state.get("epoch", 0)
            bv       = state.get("best_val", float("inf"))
            hist     = state.get("history", {})
            da_list  = hist.get("val_da", [])
            best_da  = max(da_list) if da_list else float("nan")
            complete = state.get("complete", False)

            if complete:
                # complete=True flag → finished cleanly (early stop or full run)
                reason = "early stop" if ep < total_eps - 1 else "full epochs"
                print(f"  {_COL_DONE}✓ COMPLETE  {_COL_RESET}"
                      f"{label:<15s} | epoch {ep+1:>4d}  "
                      f"best_val={bv:.5f}  best_DA={best_da:.4f}  ({reason})")
            else:
                # complete=False → interrupted mid-training
                print(f"  {_COL_RESUME}↺ RESUMABLE {_COL_RESET}"
                      f"{label:<15s} | epoch {ep+1:>4d}/{total_eps}  "
                      f"best_val={bv:.5f}  best_DA={best_da:.4f}")
        else:
            print(f"  {_COL_NONE}✗ NOT STARTED{_COL_RESET} "
                  f"{label:<15s} | no checkpoint found")

print("=" * 62)
print("  Reconnected? Just re-run all cells (Ctrl+F9).")
print("  Finished models skip, resumable ones continue")
print("  automatically from the epoch shown above.")
print("=" * 62 + "\n")


# ─────────────────────────────────────────────────────────────
# CELL 0b · ONE-TIME Recovery — Mark corrupted checkpoint as complete
# ─────────────────────────────────────────────────────────────
# Run this cell ONCE if a specialist shows as RESUMABLE in Cell 0
# but you know it already finished (e.g. daily retrained due to the
# old bug and you want to skip it and continue from weather).
#
# After running this cell once, you can comment it out or ignore it.
# It is completely safe — it only sets the complete=True flag on the
# checkpoint file and never deletes or retrains anything.
# ─────────────────────────────────────────────────────────────
from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path
import torch

_DRIVE_PROJECT_PATH = "/content/drive/MyDrive/PhD/STERGIOU/PhD/wind_paper"
_CKPT_DIR = Path(_DRIVE_PROJECT_PATH) / "outputs" / "colab_full" / "checkpoints"

# ── List the specialists you want to force-mark as complete ───
# Remove any name from this list that should NOT be skipped.
MARK_COMPLETE = ["daily"]   # ← edit as needed

for label in MARK_COMPLETE:
    cp = _CKPT_DIR / f"resume_{label}.pt"
    if not cp.exists():
        print(f"  [{label}] No checkpoint file found — nothing to patch.")
        continue
    state = torch.load(cp, map_location="cpu")
    if state.get("complete", False):
        print(f"  [{label}] Already marked complete — no action needed.")
        continue
    state["complete"] = True
    torch.save(state, cp)
    ep      = state.get("epoch", "?")
    bv      = state.get("best_val", float("inf"))
    hist    = state.get("history", {})
    da_list = hist.get("val_da", [])
    best_da = max(da_list) if da_list else float("nan")
    print(f"  [{label}] ✓ Patched — marked complete at epoch {ep+1}, "
          f"best_val={bv:.5f}, best_DA={best_da:.4f}")
    print(f"           Daily will now be SKIPPED on the next full run.")

print("\nDone. Now run Cell 0 to verify, then Ctrl+F9 to continue training.")


# ─────────────────────────────────────────────────────────────
# CELL 1 · Setup & Paths
# ─────────────────────────────────────────────────────────────
from google.colab import drive
drive.mount("/content/drive")

import os, sys, json, yaml, csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from scipy.stats import t as t_dist

# ========== EDIT THESE ==========
DRIVE_PROJECT_PATH = "/content/drive/MyDrive/PhD/STERGIOU/PhD/wind_paper"
SCADA_PATH         = "/content/drive/MyDrive/PhD"
PREPROCESSED_DIR   = f"{DRIVE_PROJECT_PATH}/data/preprocessed"
EPOCHS             = 1000
BASELINE_EPOCHS    = 300   # shorter training for baselines

SPECIALISTS = ["daily", "weather", "mesoscale", "turbulence", "trend"]
BEST_PARAMS = {
    "daily":      {"alpha": 1.3228, "beta": 0.4797, "gamma": 0.0362, "lr": 0.000589, "dropout": 0.222},
    "weather":    {"alpha": 1.5968, "beta": 1.2484, "gamma": 0.1479, "lr": 0.000104, "dropout": 0.185},
    "mesoscale":  {"alpha": 1.1577, "beta": 1.1940, "gamma": 0.0371, "lr": 0.00329,  "dropout": 0.266},
    "turbulence": {"alpha": 1.4178, "beta": 0.4511, "gamma": 0.1461, "lr": 0.000540, "dropout": 0.137},
    "trend":      {"alpha": 1.9065, "beta": 0.6335, "gamma": 0.3045, "lr": 0.000303, "dropout": 0.162},
}

# Feature column indices — match _build_features() column order in Cell 3
# [0]=ws [1]=active_power(TARGET) [2]=hour_sin [3]=hour_cos
# [4]=month_sin [5]=month_cos [6]=ws_lag1 [7]=ws_lag6 [8]=ws_lag12
# [9]=ws_lag48 [10-13]=power_lags [14]=ws_roll_mean6
# [15]=ws_roll_std6 [16]=ws_roll_mean48 [17]=power_diff1
# [18]=turbulence_intens [19]=ws_cubed
COL_POWER       = 1   # active_power — prediction TARGET
COL_HOUR_SIN    = 2   # hour_sin
COL_TURB_INTENS = 18  # turbulence_intens
COL_PRESS_GRAD  = 7   # ws_lag6 — proxy for pressure-gradient / frontal activity

sys.path.insert(0, DRIVE_PROJECT_PATH)
os.chdir(DRIVE_PROJECT_PATH)

!pip install -q pyyaml openpyxl scipy joblib

output_dir = Path(DRIVE_PROJECT_PATH) / "outputs" / "colab_full"
output_dir.mkdir(parents=True, exist_ok=True)
fig_dir = output_dir / "figures"
fig_dir.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")
print(f"Output : {output_dir}")


# ─────────────────────────────────────────────────────────────
# CELL 2 · Matplotlib — Journal-spec global settings (Energies, MDPI)
# ─────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
# Suppress repeated font-not-found warnings
import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

COL1_W = 3.54   # single-column width in inches  (90 mm)
COL2_W = 7.48   # double-column width in inches  (190 mm)

plt.rcParams.update({
    "font.family":      "DejaVu Sans",  # Arial not in Colab; DejaVu Sans is the standard substitute
    "font.size":        9,
    "axes.titlesize":   10,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "figure.dpi":       300,
    "savefig.dpi":      300,
    "savefig.format":   "png",   # high-resolution raster for submission
    "savefig.bbox":     "tight",
})
print("Matplotlib configured for publication figures.")


# ─────────────────────────────────────────────────────────────
# CELL 3 · Load & Preprocess Data
# ─────────────────────────────────────────────────────────────
# Self-contained: reads directly from the raw Excel SCADA files on
# disk (three wind farms × active power + wind speed).
# On first run it builds feats_scaled.npy and saves it.
# On every subsequent run it loads the cache instantly.
# No external scripts. No dependencies outside numpy/pandas/sklearn.
# ─────────────────────────────────────────────────────────────

_FEATS_FILE = f"{PREPROCESSED_DIR}/feats_scaled.npy"
_META_FILE  = f"{PREPROCESSED_DIR}/meta.json"

# ── Raw SCADA file paths ───────────────────────────────────────
# The three onshore wind farms are referred to only as Farm_A / Farm_B /
# Farm_C in line with the data-confidentiality agreement; their identities
# and precise locations are withheld. Set the paths below to your own local
# SCADA files (active power + wind speed, 10-min resolution) before running.
SCADA_FILES = {
    "Farm_A": {
        "power": f"{SCADA_PATH}/Farm_A_ActivePower.xlsx",
        "wind":  f"{SCADA_PATH}/Farm_A_WindSpeed.xlsx",
    },
    "Farm_B": {
        "power": f"{SCADA_PATH}/Farm_B_ActivePower.xlsx",
        "wind":  f"{SCADA_PATH}/Farm_B_WindSpeed.xlsx",
    },
    "Farm_C": {
        "power": f"{SCADA_PATH}/Farm_C_ActivePower.xlsx",
        "wind":  f"{SCADA_PATH}/Farm_C_WindSpeed.xlsx",
    },
}

def _load_xlsx(path, value_col=None):
    """Load an Excel file, parse datetime index, return cleaned Series."""
    import pandas as pd
    df = pd.read_excel(path, engine="openpyxl")
    # Find datetime column (first column that parses as datetime)
    dt_col = df.columns[0]
    df[dt_col] = pd.to_datetime(df[dt_col], dayfirst=True, errors="coerce")
    df = df.dropna(subset=[dt_col]).set_index(dt_col).sort_index()
    # Find value column — first numeric column after datetime
    if value_col is None:
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) == 0:
            raise ValueError(f"No numeric columns found in {path}")
        value_col = num_cols[0]
    return df[value_col].rename(value_col)

def _build_features(power_series, wind_series, farm_name):
    """
    Build feature matrix for one wind farm.
    Features (20 total):
      0  wind_speed          — raw wind speed (m/s normalised)
      1  active_power        — TARGET: normalised active power [0,1]
      2  hour_sin            — sin(2π·hour/24)
      3  hour_cos            — cos(2π·hour/24)
      4  month_sin           — sin(2π·month/12)
      5  month_cos           — cos(2π·month/12)
      6  ws_lag1             — wind speed t-1
      7  ws_lag6             — wind speed t-6  (1 hour ago)
      8  ws_lag12            — wind speed t-12 (2 hours ago)
      9  ws_lag48            — wind speed t-48 (8 hours ago)
     10  power_lag1          — power t-1
     11  power_lag6          — power t-6
     12  power_lag12         — power t-12
     13  power_lag48         — power t-48
     14  ws_rolling_mean6    — 1-hour rolling mean wind speed
     15  ws_rolling_std6     — 1-hour rolling std  wind speed
     16  ws_rolling_mean48   — 8-hour rolling mean wind speed
     17  power_diff1         — power first difference (ramp indicator)
     18  turbulence_intens   — ws_rolling_std6 / (ws_rolling_mean6 + 1e-6)
     19  ws_cubed            — wind speed³ (proportional to kinetic energy)
    """
    import pandas as pd

    # ── Remove duplicate timestamps (SCADA DST/reset artefacts) ──
    power_series = power_series[~power_series.index.duplicated(keep="first")]
    wind_series  = wind_series [~wind_series.index.duplicated(keep="first")]

    # Align on 10-min grid
    freq = "10min"
    idx  = pd.date_range(
        start=max(power_series.index.min(), wind_series.index.min()),
        end  =min(power_series.index.max(), wind_series.index.max()),
        freq =freq)
    pw = power_series.reindex(idx).interpolate("time").ffill().bfill()
    ws = wind_series .reindex(idx).interpolate("time").ffill().bfill()

    df = pd.DataFrame({"ws": ws, "active_power": pw}, index=idx)

    # Clip negatives (sensor noise)
    df["ws"] = df["ws"].clip(lower=0)
    df["pw"] = df["active_power"].clip(lower=0)

    # Time features
    df["hour_sin"]   = np.sin(2 * np.pi * idx.hour / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * idx.hour / 24)
    df["month_sin"]  = np.sin(2 * np.pi * idx.month / 12)
    df["month_cos"]  = np.cos(2 * np.pi * idx.month / 12)

    # Lag features
    for lag in [1, 6, 12, 48]:
        df[f"ws_lag{lag}"]    = df["ws"].shift(lag)
        df[f"power_lag{lag}"] = df["active_power"].shift(lag)

    # Rolling features
    df["ws_rolling_mean6"]  = df["ws"].rolling(6,  min_periods=1).mean()
    df["ws_rolling_std6"]   = df["ws"].rolling(6,  min_periods=1).std().fillna(0)
    df["ws_rolling_mean48"] = df["ws"].rolling(48, min_periods=1).mean()

    # Derived physics features
    df["power_diff1"]      = df["active_power"].diff().fillna(0)
    df["turbulence_intens"]= df["ws_rolling_std6"] / (df["ws_rolling_mean6"] + 1e-6)
    df["ws_cubed"]         = df["ws"] ** 3

    df = df.dropna()

    # Column order matches feature index map above
    cols = [
        "ws", "active_power",
        "hour_sin", "hour_cos", "month_sin", "month_cos",
        "ws_lag1", "ws_lag6", "ws_lag12", "ws_lag48",
        "power_lag1", "power_lag6", "power_lag12", "power_lag48",
        "ws_rolling_mean6", "ws_rolling_std6", "ws_rolling_mean48",
        "power_diff1", "turbulence_intens", "ws_cubed",
    ]
    return df[cols].values.astype(np.float32), cols

def build_preprocessed():
    """
    Read all 6 Excel files, build feature matrices, fit MinMaxScaler
    on training split, save feats_scaled.npy + meta.json to Drive.
    Returns (feats, n, train_end, val_end, input_dim, col_names).
    """
    import pandas as pd
    print("  Building preprocessed data from raw SCADA Excel files …")
    all_feats = []
    col_names = None

    for farm, paths in SCADA_FILES.items():
        print(f"    Loading {farm} …", end=" ")
        pw = _load_xlsx(paths["power"])
        ws = _load_xlsx(paths["wind"])
        feats_farm, col_names = _build_features(pw, ws, farm)
        all_feats.append(feats_farm)
        print(f"{len(feats_farm):,} samples")

    feats = np.concatenate(all_feats, axis=0)
    n     = len(feats)
    te    = int(n * 0.70)
    ve    = int(n * 0.85)

    # Fit scaler on training split only — no data leakage
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(feats[:te])
    feats  = scaler.transform(feats).astype(np.float32)

    # Save to Drive
    os.makedirs(PREPROCESSED_DIR, exist_ok=True)
    np.save(_FEATS_FILE, feats)
    meta = {
        "n": n, "train_end": te, "val_end": ve,
        "input_dim": feats.shape[1],
        "columns": col_names,
        "farms": list(SCADA_FILES.keys()),
        "built": datetime.now().isoformat(),
    }
    with open(_META_FILE, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  ✓ Saved feats_scaled.npy ({feats.shape}) + meta.json → {PREPROCESSED_DIR}")
    return feats, n, te, ve, feats.shape[1], col_names

def load_data():
    """
    Fast path : load from Drive cache (instant).
    Slow path : build from raw Excel files and save cache to Drive.
    """
    if os.path.exists(_FEATS_FILE) and os.path.exists(_META_FILE):
        print("  Loading preprocessed data from Drive cache …")
        feats = np.load(_FEATS_FILE)
        with open(_META_FILE) as f:
            meta = json.load(f)
        col_names = meta.get("columns", None)
        print(f"  ✓ Loaded: {feats.shape}  built={meta.get('built','unknown')}")
        return feats, meta["n"], meta["train_end"], meta["val_end"], meta["input_dim"], col_names
    return build_preprocessed()

feats, n, train_end, val_end, input_dim, col_names = load_data()
print(f"  Samples : {n:,}  |  train={train_end:,}  val={val_end-train_end:,}  test={n-val_end:,}")
print(f"  Features: {input_dim}")

# ── Verify and display column map ─────────────────────────────
print("\n── Feature column map ───────────────────────────────────")
if col_names:
    for i, c in enumerate(col_names):
        tag = ""
        if i == COL_POWER:       tag = "  ← COL_POWER (TARGET)"
        elif i == COL_HOUR_SIN:  tag = "  ← COL_HOUR_SIN"
        elif i == COL_TURB_INTENS: tag = "  ← COL_TURB_INTENS"
        elif i == COL_PRESS_GRAD:  tag = "  ← COL_PRESS_GRAD (ws_lag6 proxy)"
        print(f"  [{i:2d}] {c}{tag}")
else:
    print("  (column names not available in meta.json)")

# ── Sanity check COL_POWER ─────────────────────────────────────
_pm = feats[:train_end, COL_POWER].mean()
_ps = feats[:train_end, COL_POWER].std()
print(f"\n  COL_POWER={COL_POWER} ({col_names[COL_POWER] if col_names else '?'})")
print(f"  train mean={_pm:.4f}  std={_ps:.4f}", end="  ")
if 0.05 < _pm < 0.6 and _ps > 0.05:
    print("✓ looks correct")
else:
    print("⚠ WARNING — check column map above and update COL_POWER in Cell 1")


# ─────────────────────────────────────────────────────────────
# CELL 4 · Dataset
# ─────────────────────────────────────────────────────────────

class SimpleDataset(Dataset):
    def __init__(self, feats, lookback, forecast_steps, indices):
        self.feats   = feats
        self.lb      = lookback
        self.fs      = forecast_steps
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j    = self.indices[i]
        x    = torch.from_numpy(self.feats[j - self.lb : j].astype(np.float32))
        # Guard: ensure y always has exactly fs steps (tail of array can be shorter)
        raw  = self.feats[j : j + self.fs, COL_POWER].astype(np.float32)
        if len(raw) < self.fs:
            raw = np.pad(raw, (0, self.fs - len(raw)), mode="edge")
        y    = torch.from_numpy(raw)
        last = torch.tensor(self.feats[j - 1, COL_POWER], dtype=torch.float32)
        return x, y, last

ARCH = {"daily": 48, "weather": 48, "mesoscale": 36, "turbulence": 12, "trend": 72}

max_lb   = max(ARCH.values())               # = 72
FORECAST_STEPS = 6                          # primary output width (6 × 10 min = 60 min)


# ─────────────────────────────────────────────────────────────
# CELL 5 · Specialist model imports & MODELS factory
# ─────────────────────────────────────────────────────────────
from src.models.specialists import (
    DailyPatternSpecialist, WeatherFrontSpecialist,
    MesoscaleSpecialist, TurbulenceSpecialist, TrendSpecialist,
)
from src.losses.multiobjective import multi_objective_loss

MODELS = {
    "daily":      lambda lb, idim, do: DailyPatternSpecialist(
                      lookback=lb, forecast_steps=FORECAST_STEPS, input_dim=idim, dropout=do),
    "weather":    lambda lb, idim, do: WeatherFrontSpecialist(
                      lookback=lb, forecast_steps=FORECAST_STEPS, input_dim=idim,
                      hidden=256, num_heads=8, ff_dim=256, dropout=do),
    "mesoscale":  lambda lb, idim, do: MesoscaleSpecialist(
                      lookback=lb, forecast_steps=FORECAST_STEPS, input_dim=idim,
                      hidden=128, kernels=(3, 6, 12), dropout=do),
    "turbulence": lambda lb, idim, do: TurbulenceSpecialist(
                      lookback=lb, forecast_steps=FORECAST_STEPS, input_dim=idim,
                      hidden=64, dropout=do),
    "trend":      lambda lb, idim, do: TrendSpecialist(
                      lookback=lb, forecast_steps=FORECAST_STEPS, input_dim=idim,
                      hidden=256, dropout=do),
}


# ─────────────────────────────────────────────────────────────
# CELL 6 · Baseline model definitions  [NEW]
# ─────────────────────────────────────────────────────────────

class BaselineLSTM(nn.Module):
    """Explicit 2-layer LSTM baseline."""
    def __init__(self, input_dim, hidden=128, forecast_steps=6, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=2,
                            batch_first=True, dropout=dropout)
        self.fc   = nn.Linear(hidden, forecast_steps)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class CNNBiLSTM(nn.Module):
    """Compact CNN-BiLSTM — representative of hybrid architectures cited in §2 Related Work."""
    def __init__(self, input_dim, hidden=64, forecast_steps=6, dropout=0.1):
        super().__init__()
        self.conv   = nn.Conv1d(input_dim, hidden, kernel_size=3, padding=1)
        self.bilstm = nn.LSTM(hidden, hidden, num_layers=2, batch_first=True,
                              bidirectional=True, dropout=dropout)
        self.fc     = nn.Linear(hidden * 2, forecast_steps)
        self.relu   = nn.ReLU()

    def forward(self, x):
        # x: (B, T, F) → conv expects (B, F, T)
        c      = self.relu(self.conv(x.permute(0, 2, 1))).permute(0, 2, 1)
        out, _ = self.bilstm(c)
        return self.fc(out[:, -1, :])

print("Baseline model classes defined: BaselineLSTM, CNNBiLSTM")


# ─────────────────────────────────────────────────────────────
# CELL 7 · Shared training infrastructure + Train Specialists
# ─────────────────────────────────────────────────────────────
from src.eval.metrics import compute_all_metrics

!pip install -q tqdm
from tqdm.auto import tqdm

def make_indices(start, end, lookback, fs=FORECAST_STEPS):
    # Exclude indices where the forecast window would extend beyond the array
    return [i for i in range(lookback + fs, n) if start <= i < end and i + fs <= n]

train_indices = {spec: make_indices(0,        val_end, ARCH[spec]) for spec in SPECIALISTS}
val_indices   = {spec: make_indices(train_end, val_end, ARCH[spec]) for spec in SPECIALISTS}


def compute_val_metrics(model, val_ds, device, params=None, use_multiobjective=False):
    """One validation pass; returns (val_loss, mae, da)."""
    loader = DataLoader(val_ds, batch_size=256)
    model.eval()
    total_loss, preds_v, trues_v, lasts_v = 0.0, [], [], []
    with torch.no_grad():
        for x, y, last in loader:
            x, y, last = x.to(device), y.to(device), last.to(device)
            pred = model(x)
            if use_multiobjective and params:
                loss, _ = multi_objective_loss(
                    pred, y, last, last,
                    alpha=params["alpha"], beta=params["beta"], gamma=params["gamma"])
            else:
                loss = F.mse_loss(pred, y)
            total_loss   += loss.item() * len(x)
            preds_v.append(pred.cpu().numpy()[:, 0])
            trues_v.append(y.cpu().numpy()[:, 0])
            lasts_v.append(last.cpu().numpy())
    preds_v = np.concatenate(preds_v)
    trues_v = np.concatenate(trues_v)
    lasts_v = np.concatenate(lasts_v)
    m = compute_all_metrics(trues_v, preds_v, lasts_v)
    return total_loss / len(val_ds), m["mae"], m["directional_accuracy"]


# ── Checkpoint helpers ────────────────────────────────────────
CKPT_DIR = output_dir / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)

import shutil

def ckpt_path(label):
    """Path for the resume checkpoint (never deleted — marked complete instead)."""
    return CKPT_DIR / f"resume_{label}.pt"

def best_path(label, kind="specialist"):
    """Path for the best-val-loss weights checkpoint."""
    return output_dir / f"best_{label}_{kind}.pt"

def backup_path(label, kind="specialist"):
    """Path for the automatic backup of the previous best weights."""
    return output_dir / f"best_{label}_{kind}_backup.pt"

def _read_best_val_from_disk(save_path):
    """
    Read the val_loss stored inside an existing best_*.pt file.
    Returns float('inf') if the file does not exist or has no val_loss key.
    This is the PROTECTION LAYER: a fresh run can never overwrite a
    previously saved best model unless it genuinely beats it.
    """
    if save_path is None or not Path(save_path).exists():
        return float("inf")
    try:
        ck = torch.load(save_path, map_location="cpu")
        return float(ck.get("val_loss", float("inf")))
    except Exception:
        return float("inf")

def save_best_model(save_path, model, val_loss, val_da, ep, save_meta, label):
    """
    Save best model weights to save_path.
    BEFORE overwriting, backs up the existing file to *_backup.pt
    so you always have the previous best recoverable.
    """
    save_path = Path(save_path)
    # ── Back up the current best before overwriting ────────────
    if save_path.exists():
        shutil.copy2(save_path, backup_path(label,
            "specialist" if "specialist" in save_path.name else "baseline"))
    torch.save({"model": model.state_dict(),
                "val_loss": val_loss,
                "val_da":   val_da,
                "epoch":    ep,
                **(save_meta or {})}, save_path)
    tqdm.write(f"  ✓ Best model saved  ep={ep+1}  "
               f"val_loss={val_loss:.6f}  val_DA={val_da:.4f}  "
               f"(backup kept)")

def save_checkpoint(label, epoch, model, optimizer, history,
                    best_val, patience_cnt, save_meta=None, complete=False):
    """
    Save full training state to Drive for seamless resume after disconnection.
    best_val is stored here so load_checkpoint restores the exact threshold
    and a re-run can never overwrite a better model from a previous session.
    """
    torch.save({
        "epoch":        epoch,
        "model":        model.state_dict(),
        "optimizer":    optimizer.state_dict(),
        "history":      history,
        "best_val":     best_val,
        "patience_cnt": patience_cnt,
        "complete":     complete,
        **(save_meta or {}),
    }, ckpt_path(label))

def load_checkpoint(label, model, optimizer, epochs=1000, save_path=None):
    """
    Load resume checkpoint if it exists on Drive.
    Returns (start_epoch, history, best_val, patience_cnt, already_done).

    Protection layers applied here:
      1. complete flag checked BEFORE loading any weights — a finished
         model is NEVER accidentally retrained.
      2. best_val is initialised to max(resume_checkpoint_val,
         best_model_on_disk_val) — so even if the resume checkpoint is
         stale, the threshold from the saved best model is respected and
         a re-run cannot overwrite it with worse weights.
    """
    cp    = ckpt_path(label)
    empty = {"train_loss": [], "val_loss": [], "val_mae": [], "val_da": []}

    # ── Protection layer 2: always read val_loss from best_*.pt ─
    # This is the floor — training can only save a new best if it
    # beats whatever is already on Drive, regardless of resume state.
    disk_best_val = _read_best_val_from_disk(save_path)

    if not cp.exists():
        if disk_best_val < float("inf"):
            tqdm.write(f"  ↺ [{label}] No resume checkpoint but best model exists "
                       f"on disk (val_loss={disk_best_val:.6f}). "
                       f"Starting from epoch 0 but threshold protected.")
        return 0, empty, disk_best_val, 0, False

    # Load to CPU first — safe regardless of GPU state
    state = torch.load(cp, map_location="cpu")

    # ── Protection layer 1: check complete flag BEFORE weights ──
    if state.get("complete", False):
        ep      = state.get("epoch", 0)
        bv      = state.get("best_val", float("inf"))
        hist    = state.get("history", empty)
        da_list = hist.get("val_da", [])
        best_da = max(da_list) if da_list else float("nan")
        reason  = "early stop" if ep < epochs - 1 else "full epochs"
        tqdm.write(f"  ✓ [{label}] already complete ({reason}, "
                   f"epoch {ep+1}, best_val={bv:.6f}, "
                   f"best_DA={best_da:.4f}) — skipping.")
        return ep + 1, hist, bv, state.get("patience_cnt", 0), True

    # ── Interrupted mid-training — load weights and resume ──────
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    ep  = state["epoch"]
    # Use the stricter (lower) of resume val and disk val as threshold
    bv  = min(state["best_val"], disk_best_val)
    if bv < state["best_val"]:
        tqdm.write(f"  ↺ [{label}] Resuming from epoch {ep+1}/{epochs}  "
                   f"best_val={bv:.6f} (tightened from disk — best model protected)")
    else:
        tqdm.write(f"  ↺ [{label}] Resuming from epoch {ep+1}/{epochs}  "
                   f"best_val={bv:.6f}")
    return (ep + 1,
            state.get("history", empty),
            bv,
            state.get("patience_cnt", 0),
            False)


# ── Core training function ─────────────────────────────────────
def train_model(model, train_ds, val_ds, optimizer, epochs, device,
                label="model", params=None, use_multiobjective=True,
                patience=50, save_path=None, save_meta=None,
                ckpt_every=5):
    """
    Training loop with:
      • tqdm progress bars (outer epoch bar + inner batch bar)
      • Live metrics: train_loss | val_loss | val_MAE | val_DA
      • Best-model checkpoint  → save_path  (only on val_loss improvement)
      • Resume checkpoint      → checkpoints/resume_{label}.pt
            saved every ckpt_every epochs AND always after every epoch
            so a disconnection loses at most ckpt_every epochs of work
      • Early stopping (patience epochs without val_loss improvement)
      • Automatic resume: if a resume checkpoint exists it is loaded
            before the first epoch so training continues seamlessly

    Returns history dict: {train_loss, val_loss, val_mae, val_da}.
    """
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                              pin_memory=(device.type == "cuda"))

    # ── Try to resume from a previous run ─────────────────────
    start_ep, history, best_val, patience_cnt, already_done = load_checkpoint(
        label, model, optimizer, epochs=epochs, save_path=save_path)

    if already_done:
        return history  # finished cleanly before disconnection — skip

    epoch_bar = tqdm(range(start_ep, epochs),
                     desc=f"[{label}]", unit="ep",
                     initial=start_ep, total=epochs,
                     ncols=110, colour="cyan")

    for ep in epoch_bar:
        # ── Training pass ──────────────────────────────────────
        model.train()
        batch_losses = []
        batch_bar = tqdm(train_loader, desc="  train", unit="batch",
                         leave=False, ncols=110)
        for x, y, last in batch_bar:
            x, y, last = x.to(device), y.to(device), last.to(device)
            optimizer.zero_grad()
            pred = model(x)
            if use_multiobjective and params:
                loss, _ = multi_objective_loss(
                    pred, y, last, last,
                    alpha=params["alpha"], beta=params["beta"], gamma=params["gamma"])
            else:
                loss = F.mse_loss(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_losses.append(loss.item())
            batch_bar.set_postfix({"batch_loss": f"{loss.item():.5f}"})

        train_loss = float(np.mean(batch_losses))

        # ── Validation pass ────────────────────────────────────
        val_loss, val_mae, val_da = compute_val_metrics(
            model, val_ds, device,
            params=params, use_multiobjective=use_multiobjective)

        # ── Record history ─────────────────────────────────────
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)
        history["val_da"].append(val_da)

        # ── Update epoch bar ───────────────────────────────────
        epoch_bar.set_postfix({
            "tr_loss": f"{train_loss:.5f}",
            "vl_loss": f"{val_loss:.5f}",
            "vl_MAE":  f"{val_mae:.5f}",
            "vl_DA":   f"{val_da:.4f}",
        })

        # ── Best-model checkpoint (only on genuine improvement) ──
        # Protected: can only overwrite if strictly better than both
        # the current session best AND whatever is already on Drive.
        if val_loss < best_val - 1e-6:
            best_val     = val_loss
            patience_cnt = 0
            if save_path:
                save_best_model(save_path, model, val_loss, val_da,
                                ep, save_meta, label)
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                tqdm.write(f"  Early stop at epoch {ep+1} "
                           f"(best val_loss={best_val:.6f})")
                # Mark as complete so reconnection does not restart from scratch
                save_checkpoint(label, ep, model, optimizer, history,
                                best_val, patience_cnt, save_meta, complete=True)
                tqdm.write(f"  ✓ [{label}] checkpoint marked complete (early stop).")
                break

        # ── Resume checkpoint (every ckpt_every epochs) ────────
        # Saved with complete=False so a mid-epoch disconnection resumes here
        if (ep + 1) % ckpt_every == 0:
            save_checkpoint(label, ep, model, optimizer, history,
                            best_val, patience_cnt, save_meta, complete=False)

    else:
        # Loop finished without early stop — full epochs completed
        save_checkpoint(label, ep, model, optimizer, history,
                        best_val, patience_cnt, save_meta, complete=True)
        tqdm.write(f"  ✓ [{label}] checkpoint marked complete (full epochs).")

    tqdm.write(f"  [{label}] done — best val_loss={best_val:.6f}  "
               f"best val_DA={max(history['val_da']):.4f}")
    return history


# ── Train all specialists ──────────────────────────────────────
all_histories = {}

for spec in SPECIALISTS:
    lb     = ARCH[spec]
    params = BEST_PARAMS[spec]
    tr_ds  = SimpleDataset(feats, lb, FORECAST_STEPS, train_indices[spec])
    vl_ds  = SimpleDataset(feats, lb, FORECAST_STEPS, val_indices[spec])
    model  = MODELS[spec](lb, input_dim, params["dropout"]).to(device)
    opt    = torch.optim.AdamW(model.parameters(), lr=params["lr"])

    print(f"\n{chr(9472)*60}")
    print(f"  Specialist : {spec.upper()}   "
          f"train={len(tr_ds):,}  val={len(vl_ds):,}  "
          f"lb={lb}  lr={params['lr']:.6f}")
    print(f"{chr(9472)*60}")

    hist = train_model(
        model, tr_ds, vl_ds, opt, EPOCHS, device,
        label=spec,
        params=params,
        use_multiobjective=True,
        patience=80,
        ckpt_every=5,                          # resume checkpoint every 5 epochs
        save_path=best_path(spec, "specialist"),
        save_meta={"lookback": lb, "input_dim": input_dim, "specialist": spec},
    )
    all_histories[spec] = hist

print("\nAll specialists trained.")


# ─────────────────────────────────────────────────────────────
# CELL 8 · Train Baselines  [NEW]
# ─────────────────────────────────────────────────────────────
BASELINE_LB = 48   # shared lookback for both baselines

BASELINES_DEF = {
    "LSTM_2layer": BaselineLSTM(input_dim, hidden=128, forecast_steps=FORECAST_STEPS),
    "CNN_BiLSTM":  CNNBiLSTM   (input_dim, hidden=64,  forecast_steps=FORECAST_STEPS),
}

bl_tr_ds = SimpleDataset(feats, BASELINE_LB, FORECAST_STEPS,
                         make_indices(0,        val_end,  BASELINE_LB))
bl_vl_ds = SimpleDataset(feats, BASELINE_LB, FORECAST_STEPS,
                         make_indices(train_end, val_end, BASELINE_LB))

for bname, bmodel in BASELINES_DEF.items():
    bmodel = bmodel.to(device)
    opt_b  = torch.optim.AdamW(bmodel.parameters(), lr=1e-3)

    print(f"\n{chr(9472)*60}")
    print(f"  Baseline : {bname}   "
          f"train={len(bl_tr_ds):,}  val={len(bl_vl_ds):,}")
    print(f"{chr(9472)*60}")

    hist_b = train_model(
        bmodel, bl_tr_ds, bl_vl_ds, opt_b, BASELINE_EPOCHS, device,
        label=bname,
        params=None,
        use_multiobjective=False,
        patience=30,
        ckpt_every=5,
        save_path=best_path(bname, "baseline"),
        save_meta={"lookback": BASELINE_LB, "input_dim": input_dim, "baseline": bname},
    )
    all_histories[bname] = hist_b

print("\nAll baselines trained.")



# ─────────────────────────────────────────────────────────────
# CELL 9 · Evaluate Specialists & Baselines on test set
# ─────────────────────────────────────────────────────────────
test_idx = make_indices(val_end, n, max_lb)

def run_eval(model, lookback, test_indices, step_idx=0):
    """Run a trained model on test_indices; return (metrics, preds, trues, lasts)."""
    valid  = [i for i in test_indices if i >= lookback + FORECAST_STEPS and i + FORECAST_STEPS <= n]
    ds     = SimpleDataset(feats, lookback, FORECAST_STEPS, valid)
    loader = DataLoader(ds, batch_size=64)
    model.eval()
    preds, trues, lasts = [], [], []
    with torch.no_grad():
        for x, y, last in loader:
            p = model(x.to(device))
            preds.append(p.cpu().numpy()[:, step_idx])
            trues.append(y.numpy()[:, step_idx])
            lasts.append(last.numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    lasts = np.concatenate(lasts)
    return compute_all_metrics(trues, preds, lasts), preds, trues, lasts

all_metrics = {}
all_preds   = {}
trues_common  = None
lasts_common  = None

# ── Specialists
for spec in SPECIALISTS:
    lb    = ARCH[spec]
    ckpt  = torch.load(output_dir / f"best_{spec}_specialist.pt", map_location=device)
    model = MODELS[spec](ckpt["lookback"], ckpt["input_dim"],
                         BEST_PARAMS[spec]["dropout"]).to(device)
    model.load_state_dict(ckpt["model"])
    m, pred, tru, last = run_eval(model, lb, test_idx)
    all_metrics[spec] = m
    all_preds[spec]   = pred
    if trues_common is None:
        trues_common, lasts_common = tru, last
    print(f"{spec:12s}: MAE={m['mae']:.6f}  DA={m['directional_accuracy']:.4f}")

# ── Persistence baseline
persist_pred      = lasts_common.copy()
persist_m         = compute_all_metrics(trues_common, persist_pred, lasts_common)
all_metrics["Persistence"] = persist_m
all_preds["Persistence"]   = persist_pred
print(f"{'Persistence':12s}: MAE={persist_m['mae']:.6f}  DA={persist_m['directional_accuracy']:.4f}")

# ── Trained baselines
for bname in BASELINES_DEF:
    ckpt   = torch.load(output_dir / f"best_{bname}_baseline.pt", map_location=device)
    bmodel = BASELINES_DEF[bname]
    bmodel.load_state_dict(ckpt["model"])
    bmodel = bmodel.to(device)
    m, pred, tru, last = run_eval(bmodel, BASELINE_LB, test_idx)
    all_metrics[bname] = m
    all_preds[bname]   = pred
    print(f"{bname:12s}: MAE={m['mae']:.6f}  DA={m['directional_accuracy']:.4f}")

# ── Simple-average ensemble
min_len   = min(len(all_preds[s]) for s in SPECIALISTS)
ens_pred  = np.mean([all_preds[s][:min_len] for s in SPECIALISTS], axis=0)
tru_ens   = trues_common[:min_len]
last_ens  = lasts_common[:min_len]
ens_m     = compute_all_metrics(tru_ens, ens_pred, last_ens)
all_metrics["Ensemble"] = ens_m
all_preds["Ensemble"]   = ens_pred
print(f"{'Ensemble':12s}: MAE={ens_m['mae']:.6f}  DA={ens_m['directional_accuracy']:.4f}")


# ─────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────
# CELL 9a · Fast-Resume: reload saved results from Drive
# ─────────────────────────────────────────────────────────────
# After any disconnection, Ctrl+F9 will:
#   • Skip all training (checkpoints complete)
#   • Re-run Cell 9 to rebuild all_preds / trues_common
#   • Hit THIS cell and reload all previously saved JSONs/CSVs
#     instantly — no recomputation of ablation, CI, DM tests
#   • Downstream cells detect variables already set and skip
# ─────────────────────────────────────────────────────────────
import pandas as pd

def _try_load(path, loader, label):
    p = Path(path)
    if p.exists():
        try:
            result = loader(p)
            print(f"  ✓ {label}")
            return result
        except Exception as e:
            print(f"  ✗ {label}  (error: {e})")
            return None
    print(f"  – {label}  (not yet saved)")
    return None

print("Fast-resume: loading saved results from Drive …\n")

_abl = _try_load(output_dir / "ablation_results.json",
                 lambda p: json.load(open(p)), "ablation_results.json")
if _abl is not None:
    ablation_results = _abl

_hor = _try_load(output_dir / "horizon_metrics.csv", pd.read_csv, "horizon_metrics.csv")
if _hor is not None:
    df_h = _hor

_nic = _try_load(output_dir / "niche_metrics.json",
                 lambda p: json.load(open(p)), "niche_metrics.json")
if _nic is not None:
    niche_metrics = _nic

_int = _try_load(output_dir / "interval_calibration.json",
                 lambda p: json.load(open(p)), "interval_calibration.json")
if _int is not None:
    ZETA       = _int["zeta"]
    picp_test  = _int["test_picp"]
    pinaw_test = _int["test_pinaw"]
    # Rebuild prediction intervals needed for figure generation
    _ts   = np.stack([all_preds[s][:min_len] for s in SPECIALISTS])
    _tmn  = _ts.mean(axis=0)
    _tsd  = _ts.std(axis=0) + 1e-8
    lower_t = _tmn - ZETA * _tsd
    upper_t = _tmn + ZETA * _tsd
    print(f"    ζ={ZETA}  PICP={picp_test:.4f}  PINAW={pinaw_test:.4f}")

_bci = _try_load(output_dir / "bootstrap_ci.json",
                 lambda p: json.load(open(p)), "bootstrap_ci.json")
if _bci is not None:
    ci_results = _bci

_dm = _try_load(output_dir / "dm_tests.json",
                lambda p: json.load(open(p)), "dm_tests.json")
if _dm is not None:
    dm_results = _dm

_am = _try_load(output_dir / "metrics_all.json",
                lambda p: json.load(open(p)), "metrics_all.json")
if _am is not None:
    all_metrics.update(_am)

_RESUME_FLAGS = {
    "ablation":   _abl  is not None,
    "horizon":    _hor  is not None,
    "niche":      _nic  is not None,
    "interval":   _int  is not None,
    "bootstrap":  _bci  is not None,
    "dm":         _dm   is not None,
    "metrics":    _am   is not None,
}
print("\nFast-resume flags:", {k: ("✓ loaded" if v else "– pending") for k, v in _RESUME_FLAGS.items()})


# ─────────────────────────────────────────────────────────────
# CELL 9b · Regime-Stratified Directional Accuracy
# Splits test set into ramp events vs stable periods.
# Ramp regime is the paper's headline metric.
# ─────────────────────────────────────────────────────────────
RAMP_THRESHOLD = 0.05   # |Δpower| > 5% of normalised scale = ramp event

delta_test  = np.abs(trues_common - lasts_common)
ramp_mask   = delta_test > RAMP_THRESHOLD
stable_mask = ~ramp_mask

n_ramp   = int(ramp_mask.sum())
n_stable = int(stable_mask.sum())
print(f"\n── Regime breakdown ────────────────────────────────────────")
print(f"  Total test samples : {len(trues_common):,}")
print(f"  Ramp events        : {n_ramp:,}  ({100*ramp_mask.mean():.1f}%)")
print(f"  Stable periods     : {n_stable:,}  ({100*stable_mask.mean():.1f}%)")
print(f"  Ramp threshold     : |Δpower| > {RAMP_THRESHOLD} (normalised)")
print()

regime_rows = []
for name, pred in all_preds.items():
    p = pred[:len(trues_common)]
    t = trues_common
    l = lasts_common

    def _da(pr, tr, ls):
        if len(pr) == 0:
            return float("nan")
        return float(np.mean(np.sign(pr - ls) == np.sign(tr - ls)))

    da_all    = _da(p,              t,              l)
    da_ramp   = _da(p[ramp_mask],   t[ramp_mask],   l[ramp_mask])
    da_stable = _da(p[stable_mask], t[stable_mask], l[stable_mask])
    regime_rows.append({
        "model": name,
        "DA_all": da_all,
        "DA_ramp": da_ramp,
        "DA_stable": da_stable,
        "n_ramp": n_ramp,
        "n_stable": n_stable,
    })
    print(f"  {name:15s}  DA_all={da_all:.4f}  "
          f"DA_ramp={da_ramp:.4f}  DA_stable={da_stable:.4f}")

import pandas as pd
df_regime = pd.DataFrame(regime_rows)
df_regime.to_csv(output_dir / "regime_metrics.csv", index=False)
print(f"\nRegime metrics saved → {output_dir / 'regime_metrics.csv'}")

# Headline numbers for paper
ens_ramp = df_regime.loc[df_regime.model == "Ensemble", "DA_ramp"].values[0]
per_ramp = df_regime.loc[df_regime.model == "Persistence", "DA_ramp"].values[0]
print(f"\n★ PAPER HEADLINE: Ensemble ramp DA = {ens_ramp*100:.1f}%  "
      f"vs Persistence ramp DA = {per_ramp*100:.1f}%")

# CELL 10 · Multi-Horizon Evaluation  [NEW]
# ─────────────────────────────────────────────────────────────
# NOTE: model is trained with FORECAST_STEPS=6 outputs (10–60 min).
# Horizons > 60 min are not supported — excluded for honest reporting.
HORIZONS     = [1, 2, 3, 6]          # 10, 20, 30, 60 min — within trained range
HORIZONS_MIN = [h * 10 for h in HORIZONS]

all_horizon_metrics = {}

if _RESUME_FLAGS["horizon"]:
    print("Cell 10: horizon_metrics loaded by Cell 9a — skipping.")
else:
    # Load all models once (not once per horizon — avoids redundant I/O)
    _spec_models = {}
    for spec in SPECIALISTS:
        lb   = ARCH[spec]
        ckpt = torch.load(output_dir / f"best_{spec}_specialist.pt", map_location=device)
        m    = MODELS[spec](ckpt["lookback"], ckpt["input_dim"],
                            BEST_PARAMS[spec]["dropout"]).to(device)
        m.load_state_dict(ckpt["model"])
        _spec_models[spec] = (m, lb)

    _base_models = {}
    for bname in BASELINES_DEF:
        ckpt   = torch.load(output_dir / f"best_{bname}_baseline.pt", map_location=device)
        bmodel = BASELINES_DEF[bname]
        bmodel.load_state_dict(ckpt["model"])
        _base_models[bname] = bmodel.to(device)

    print("\nMulti-horizon evaluation …")
    for h_steps in HORIZONS:
        all_horizon_metrics[h_steps] = {}
        step_idx = h_steps - 1

        for spec, (model, lb) in _spec_models.items():
            m, _, _, _ = run_eval(model, lb, test_idx, step_idx=step_idx)
            all_horizon_metrics[h_steps][spec] = m

        for bname, bmodel in _base_models.items():
            m, _, _, _ = run_eval(bmodel, BASELINE_LB, test_idx, step_idx=step_idx)
            all_horizon_metrics[h_steps][bname] = m

        # Ensemble at this horizon
        spec_preds_h = []
        tru_h = last_h = None
        for spec, (model, lb) in _spec_models.items():
            _, pred_h, tru_h, last_h = run_eval(model, lb, test_idx, step_idx=step_idx)
            spec_preds_h.append(pred_h)
        min_h = min(len(p) for p in spec_preds_h)
        ens_h = np.mean([p[:min_h] for p in spec_preds_h], axis=0)
        all_horizon_metrics[h_steps]["Ensemble"] = compute_all_metrics(
            tru_h[:min_h], ens_h, last_h[:min_h])

        print(f"  {h_steps*10:4d} min  —  Ensemble DA="
              f"{all_horizon_metrics[h_steps]['Ensemble']['directional_accuracy']:.4f}")

    # Save
    import pandas as pd
    rows_h = []
    for h_steps, mdict in all_horizon_metrics.items():
        for mname, mm in mdict.items():
            rows_h.append({"horizon_min": h_steps * 10, "model": mname,
                            "MAE": mm["mae"], "DA": mm["directional_accuracy"],
                            "RMSE": mm["rmse"]})
    df_h = pd.DataFrame(rows_h)
    df_h.to_csv(output_dir / "horizon_metrics.csv", index=False)

print("\nHorizon table (DA):")
print(df_h.pivot(index="horizon_min", columns="model", values="DA").round(4).to_string())


# ─────────────────────────────────────────────────────────────
# CELL 11 · Niche Accuracy Evaluation  [NEW]
# Sub-model domain-filtered accuracy to support §4.2 claims
# ─────────────────────────────────────────────────────────────
_run_niche = not _RESUME_FLAGS.get("niche", False)
if not _run_niche:
    print("Cell 11: niche_metrics loaded by Cell 9a — skipping.")

def get_niche_indices(base_indices, regime):
    """Return subset of base_indices matching the specialist's domain regime.

    Thresholds are set broadly so that each niche retains enough samples
    to give a statistically meaningful DA estimate (>= MIN_NICHE_SAMPLES).
    """
    subset = []
    for i in base_indices:
        f = feats[i]
        if regime == "daily":
            # Daytime hours (hour_sin > 0): captures diurnal production cycle
            if f[COL_HOUR_SIN] > 0.1 and f[COL_TURB_INTENS] < 0.4:
                subset.append(i)
        elif regime == "weather":
            # Notable wind-speed gradient (ws_lag6 proxy for frontal activity)
            if abs(f[COL_PRESS_GRAD]) > 0.3:
                subset.append(i)
        elif regime == "turbulence":
            # Higher turbulence intensity — gusty / turbulent conditions
            if f[COL_TURB_INTENS] > 0.25:
                subset.append(i)
        elif regime == "trend":
            # Low turbulence → persistent slow-moving trend
            if f[COL_TURB_INTENS] < 0.2:
                subset.append(i)
        elif regime == "mesoscale":
            # Moderate turbulence, any hour — proxy for mesoscale variability
            if 0.15 < f[COL_TURB_INTENS] < 0.55:
                subset.append(i)
    return subset

SPEC_REGIMES = {
    "daily":      "daily",
    "weather":    "weather",
    "mesoscale":  "mesoscale",
    "turbulence": "turbulence",
    "trend":      "trend",
}

niche_metrics = {}
MIN_NICHE_SAMPLES = 100

if _run_niche:
    print("\nNiche (domain-filtered) evaluation …")
    for spec in SPECIALISTS:
        regime    = SPEC_REGIMES[spec]
        niche_idx = get_niche_indices(test_idx, regime)
        if len(niche_idx) < MIN_NICHE_SAMPLES:
            print(f"  WARNING: {spec} niche has only {len(niche_idx)} samples "
                  f"— adjust threshold in get_niche_indices()")
            continue
        lb    = ARCH[spec]
        ckpt  = torch.load(output_dir / f"best_{spec}_specialist.pt", map_location=device)
        model = MODELS[spec](ckpt["lookback"], ckpt["input_dim"],
                             BEST_PARAMS[spec]["dropout"]).to(device)
        model.load_state_dict(ckpt["model"])
        m, _, _, _ = run_eval(model, lb, niche_idx)
        niche_metrics[spec] = m
        niche_metrics[spec]["n_samples"] = len(niche_idx)
        print(f"  {spec:12s} niche ({regime}, n={len(niche_idx):,}): "
              f"DA={m['directional_accuracy']:.4f}  MAE={m['mae']:.6f}")
    # Save niche metrics
    with open(output_dir / "niche_metrics.json", "w") as f:
        json.dump({k: {kk: float(vv) for kk, vv in v.items()}
                   for k, v in niche_metrics.items()}, f, indent=2)


# ─────────────────────────────────────────────────────────────
# CELL 12 · Loss-Component Ablation  [NEW]
# Trains 3 daily-specialist variants, adding one loss term at a time.
# Each config runs for ABLATION_EPOCHS (60) with early stopping.
# Results saved to Drive after EACH config — crash-safe.
# Est. time on A100: ~5–8 min total.
# ─────────────────────────────────────────────────────────────
LOSS_ABLATIONS = {
    "Stage1_MSE_only":     dict(alpha=1.0, beta=0.0, gamma=0.0),
    "Stage2_MSE_Dir":      dict(alpha=1.0, beta=0.5, gamma=0.0),
    "Stage3_MSE_Dir_Temp": dict(alpha=1.0, beta=0.5, gamma=0.2),
}
ABLATION_EPOCHS   = 60
ABLATION_PATIENCE = 12
ABLATION_LR       = 5e-4
ABLATION_BATCH    = 256
abl_path          = output_dir / "ablation_results.json"

# Load existing results — skip completed configs on re-run
if abl_path.exists():
    with open(abl_path) as f:
        ablation_results = json.load(f)
    print(f"Loaded existing ablation: {list(ablation_results.keys())}")
else:
    ablation_results = {}

ablation_train_ds = SimpleDataset(feats, 48, FORECAST_STEPS, make_indices(0,         val_end, 48))
ablation_val_ds   = SimpleDataset(feats, 48, FORECAST_STEPS, make_indices(train_end,  val_end, 48))
ablation_test_idx = make_indices(val_end, n, 48)

dl_abl_tr = DataLoader(ablation_train_ds, batch_size=ABLATION_BATCH,
                       shuffle=True, num_workers=0, pin_memory=True)
dl_abl_vl = DataLoader(ablation_val_ds,  batch_size=ABLATION_BATCH,
                       shuffle=False, num_workers=0, pin_memory=True)

print("\nLoss-component ablation (daily specialist) …")
print(f"Device: {device}  |  {ABLATION_EPOCHS} epochs × {len(LOSS_ABLATIONS)} configs\n")

for stage_name, lp in LOSS_ABLATIONS.items():
    if stage_name in ablation_results:
        r = ablation_results[stage_name]
        print(f"  [{stage_name}] already done → "
              f"DA={r.get('directional_accuracy', r.get('DA', 0)):.4f}  skipping.")
        continue

    import time as _time
    t0 = _time.time()
    print(f"  [{stage_name}] training …", end="", flush=True)

    model_a = DailyPatternSpecialist(
        lookback=48, forecast_steps=FORECAST_STEPS, input_dim=input_dim,
        dropout=0.2).to(device)
    opt_a = torch.optim.Adam(model_a.parameters(), lr=ABLATION_LR)
    sched_a = torch.optim.lr_scheduler.CosineAnnealingLR(opt_a, T_max=ABLATION_EPOCHS)

    best_val_a = float("inf")
    pat_a      = 0
    stopped_ep = ABLATION_EPOCHS

    for ep in range(1, ABLATION_EPOCHS + 1):
        model_a.train()
        for x, y, last in dl_abl_tr:
            x, y, last = x.to(device), y.to(device), last.to(device)
            opt_a.zero_grad()
            pred    = model_a(x)
            loss, _ = multi_objective_loss(
                pred, y, last, last,
                alpha=lp["alpha"], beta=lp["beta"], gamma=lp["gamma"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_a.parameters(), 1.0)
            opt_a.step()
        sched_a.step()

        # Validate every 3 epochs
        if ep % 3 != 0:
            continue
        model_a.eval()
        vl_list = []
        with torch.no_grad():
            for x, y, last in dl_abl_vl:
                x, y = x.to(device), y.to(device)
                pred = model_a(x)
                vl_list.append(nn.functional.mse_loss(pred, y).item())
        vl = float(np.mean(vl_list))
        if vl < best_val_a:
            best_val_a = vl
            pat_a = 0
        else:
            pat_a += 1
            if pat_a >= ABLATION_PATIENCE:
                stopped_ep = ep
                break

    m_a, _, _, _ = run_eval(model_a, 48, ablation_test_idx)
    elapsed = _time.time() - t0
    ablation_results[stage_name] = {k: float(v) for k, v in m_a.items()}
    ablation_results[stage_name]["epochs"]    = stopped_ep
    ablation_results[stage_name]["elapsed_s"] = round(elapsed, 1)

    # Save immediately after each config
    with open(abl_path, "w") as f:
        json.dump(ablation_results, f, indent=2)
    print(f" DA={m_a['directional_accuracy']:.4f}  MAE={m_a['mae']:.5f}"
          f"  ep={stopped_ep}  ({elapsed/60:.1f} min)  ✓ saved")

# ── Summary table ─────────────────────────────────────────────
print("\n── Ablation summary ──────────────────────────────────────────")
print(f"{'Config':<25}  {'DA':>6}  {'MAE':>8}  {'epochs':>6}")
print("-" * 52)
for k, v in ablation_results.items():
    da  = v.get("directional_accuracy", v.get("DA", 0))
    mae = v.get("mae", v.get("MAE", 0))
    ep  = v.get("epochs", "-")
    print(f"{k:<25}  {da:>6.4f}  {mae:>8.5f}  {ep:>6}")


# ─────────────────────────────────────────────────────────────
# CELL 13 · Prediction Intervals & ζ Calibration  [NEW]
# Closes the missing §3.6 reference and reports PICP / PINAW
# ─────────────────────────────────────────────────────────────
# Skip if already loaded by Cell 9a; otherwise compute and save
if _RESUME_FLAGS.get("interval", False):
    print("Cell 13: interval calibration loaded by Cell 9a — skipping.")
    # lower_t / upper_t already rebuilt in Cell 9a
else:
    # Step 1 — collect all specialist predictions on the validation set
    val_idx            = make_indices(train_end, val_end, max_lb)
    val_preds_per_spec = {}
    val_trues_ref      = None

    print("\nCollecting specialist predictions on validation set for calibration …")
    for spec in SPECIALISTS:
        lb    = ARCH[spec]
        ckpt  = torch.load(output_dir / f"best_{spec}_specialist.pt", map_location=device)
        model = MODELS[spec](ckpt["lookback"], ckpt["input_dim"],
                             BEST_PARAMS[spec]["dropout"]).to(device)
        model.load_state_dict(ckpt["model"])
        _, pred_v, tru_v, _ = run_eval(model, lb, val_idx)
        val_preds_per_spec[spec] = pred_v
        if val_trues_ref is None:
            val_trues_ref = tru_v

    # Step 2 — calibrate ζ on validation set for 95% PICP
    min_val_len = min(len(v) for v in val_preds_per_spec.values())
    val_stack   = np.stack([val_preds_per_spec[s][:min_val_len] for s in SPECIALISTS])
    val_mean    = val_stack.mean(axis=0)
    val_std     = val_stack.std(axis=0) + 1e-8
    val_trues_c = val_trues_ref[:min_val_len]

    TARGET_PICP = 0.95
    ZETA = None
    for zeta in np.arange(0.1, 10.0, 0.05):
        lower = val_mean - zeta * val_std
        upper = val_mean + zeta * val_std
        picp  = np.mean((val_trues_c >= lower) & (val_trues_c <= upper))
        if picp >= TARGET_PICP:
            ZETA = round(float(zeta), 3)
            print(f"Calibrated ζ = {ZETA:.3f}  →  Val PICP = {picp:.4f}")
            break

    if ZETA is None:
        ZETA = 5.0
        print(f"WARNING: could not reach {TARGET_PICP*100:.0f}% PICP; using ζ = {ZETA}")

    # Step 3 — test-set PICP and PINAW
    test_stack = np.stack([all_preds[s][:min_len] for s in SPECIALISTS])
    test_mean  = test_stack.mean(axis=0)
    test_std   = test_stack.std(axis=0) + 1e-8
    lower_t    = test_mean - ZETA * test_std
    upper_t    = test_mean + ZETA * test_std
    picp_test  = float(np.mean((tru_ens[:min_len] >= lower_t) & (tru_ens[:min_len] <= upper_t)))
    pinaw_test = float(np.mean(upper_t - lower_t))

    print(f"Test PICP  = {picp_test:.4f}  (nominal 95%)")
    print(f"Test PINAW = {pinaw_test:.4f}  (lower is better)")

    interval_results = {"zeta": ZETA, "val_target_picp": TARGET_PICP,
                        "test_picp": picp_test, "test_pinaw": pinaw_test}
    with open(output_dir / "interval_calibration.json", "w") as f:
        json.dump(interval_results, f, indent=2)


# ─────────────────────────────────────────────────────────────
# CELL 14 · Bootstrap Confidence Intervals  [NEW]
# ─────────────────────────────────────────────────────────────
def bootstrap_ci(trues, preds, lasts, n_boot=500, ci=0.95, metric="directional_accuracy"):
    """Return (mean, lower_bound, upper_bound) via percentile bootstrap.
    n_boot=500 is sufficient for stable 95% CI estimates on n>30,000 samples."""
    n_s   = len(trues)
    vals  = []
    for _ in range(n_boot):
        idx = np.random.randint(0, n_s, size=n_s)
        m   = compute_all_metrics(trues[idx], preds[idx], lasts[idx])
        vals.append(m[metric])
    vals  = np.array(vals)
    alpha = (1 - ci) / 2
    return float(vals.mean()), float(np.percentile(vals, alpha * 100)), \
           float(np.percentile(vals, (1 - alpha) * 100))

EVAL_ORDER = ["Persistence"] + list(BASELINES_DEF.keys()) + SPECIALISTS + ["Ensemble"]
min_len_ci = min(len(all_preds[m]) for m in EVAL_ORDER)

_bci_path = output_dir / "bootstrap_ci.json"

# Always try to load partial results from Drive (crash-safe resume)
# This works even if Cell 9a flagged bootstrap as pending —
# the file may exist with partial entries from a previous interrupted run.
if _bci_path.exists():
    try:
        with open(_bci_path) as f:
            ci_results = json.load(f)
        print(f"Loaded partial bootstrap ({len(ci_results)}/{len(EVAL_ORDER)} models): "
              f"{list(ci_results.keys())}")
    except Exception:
        ci_results = {}
else:
    ci_results = {}

if len(ci_results) == len(EVAL_ORDER):
    print("Cell 14: bootstrap_ci complete — skipping.")
else:
    print(f"\nBootstrap 95% CIs on Directional Accuracy "
          f"({len(EVAL_ORDER) - len(ci_results)} remaining) …")
    for mname in EVAL_ORDER:
        if mname in ci_results:
            r = ci_results[mname]
            print(f"  {mname:15s}: DA = {r['DA_mean']:.4f}  "
                  f"[{r['DA_lo']:.4f}, {r['DA_hi']:.4f}]  (cached)")
            continue
        pred_ci = all_preds[mname][:min_len_ci]
        tru_ci  = trues_common[:min_len_ci]
        las_ci  = lasts_common[:min_len_ci]
        mean_da, lo, hi = bootstrap_ci(tru_ci, pred_ci, las_ci)
        ci_results[mname] = {"DA_mean": mean_da, "DA_lo": lo, "DA_hi": hi}
        print(f"  {mname:15s}: DA = {mean_da:.4f}  [{lo:.4f}, {hi:.4f}]")
        # Save to Drive immediately after each model
        with open(_bci_path, "w") as f:
            json.dump(ci_results, f, indent=2)


# ─────────────────────────────────────────────────────────────
# CELL 15 · Diebold-Mariano Tests  [NEW]
# ─────────────────────────────────────────────────────────────
def diebold_mariano(e1, e2, h=1):
    """
    DM test: H0 = no difference in MSE-based forecast accuracy.
    Returns (DM statistic, two-sided p-value).
    e1 = errors of the reference model (larger = worse)
    e2 = errors of the improved model  (smaller = better)
    Significant p → e2 is significantly better than e1.
    """
    d     = e1 ** 2 - e2 ** 2        # loss differential
    n_obs = len(d)
    d_bar = d.mean()
    # Newey-West variance estimator
    nw_var = np.var(d, ddof=1)
    for lag in range(1, h):
        gamma_l = np.cov(d[lag:], d[:-lag])[0, 1]
        nw_var += 2 * (1 - lag / h) * gamma_l
    dm_stat = d_bar / np.sqrt(max(nw_var, 1e-12) / n_obs)
    p_val   = float(2 * t_dist.cdf(-abs(dm_stat), df=n_obs - 1))
    return float(dm_stat), p_val

ens_errors = np.abs(tru_ens[:min_len_ci] - ens_pred[:min_len_ci])

if _RESUME_FLAGS.get("dm", False):
    print("Cell 15: dm_tests loaded by Cell 9a — skipping.")
else:
    dm_results = {}
    print("\nDiebold-Mariano tests (Ensemble vs each baseline) …")
    for bname in ["Persistence"] + list(BASELINES_DEF.keys()):
        b_pred   = all_preds[bname][:min_len_ci]
        b_errors = np.abs(trues_common[:min_len_ci] - b_pred)
        dm, p    = diebold_mariano(b_errors, ens_errors)
        sig      = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        dm_results[bname] = {"DM": dm, "p": p, "sig": sig}
        print(f"  Ensemble vs {bname:15s}: DM={dm:+.3f}  p={p:.4f}  {sig}")
    with open(output_dir / "dm_tests.json", "w") as f:
        json.dump(dm_results, f, indent=2)


# ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────
# CELL 16 · Tables
# ─────────────────────────────────────────────────────────────
import pandas as pd

# ── Table 1: Main comparison
# Columns: Model | MAE | RMSE | r | DA (%) | 95% CI | WDA (%) | TPA (%)
rows_main = [["Model", "MAE", "RMSE", "r",
              "DA%", "DA_CI_lo%", "DA_CI_hi%",
              "DA_ramp%", "DA_stable%",
              "WDA%", "TPA%"]]

# Load regime metrics for ramp/stable DA columns
import pandas as _pd2
try:
    _df_reg = _pd2.read_csv(output_dir / "regime_metrics.csv")
    _reg_map = {row["model"]: row.to_dict() for _, row in _df_reg.iterrows()}
except Exception:
    _reg_map = {}

for name in EVAL_ORDER:
    m   = all_metrics[name]
    ci  = ci_results.get(name, {})
    reg = _reg_map.get(name, {})
    da_ramp   = f"{float(reg['DA_ramp'])*100:.1f}"   if name in _reg_map else ""
    da_stable = f"{float(reg['DA_stable'])*100:.1f}" if name in _reg_map else ""
    rows_main.append([
        name,
        f"{float(m['mae']):.4f}",
        f"{float(m['rmse']):.4f}",
        f"{float(m['pearson_r']):.4f}",
        f"{float(m['directional_accuracy'])*100:.1f}",
        f"{float(ci.get('DA_lo', 0))*100:.1f}" if ci else "",
        f"{float(ci.get('DA_hi', 0))*100:.1f}" if ci else "",
        da_ramp,
        da_stable,
        f"{float(m['weighted_directional_accuracy'])*100:.1f}",
        f"{float(m['turning_point_accuracy'])*100:.1f}",
    ])
with open(output_dir / "results_table.csv", "w", newline="") as f:
    csv.writer(f).writerows(rows_main)
df_main = pd.DataFrame(rows_main[1:], columns=rows_main[0])
print("\n=== Table 1: Main Results ===")
display(df_main)

# ── Table 2: Regime-stratified DA (paper Table 2)
rows_regime = [["Model", "DA_all%", "DA_ramp%", "DA_stable%"]]
for name in EVAL_ORDER:
    reg = _reg_map.get(name, {})
    if name in _reg_map:
        rows_regime.append([
            name,
            f"{float(reg['DA_all'])*100:.1f}",
            f"{float(reg['DA_ramp'])*100:.1f}",
            f"{float(reg['DA_stable'])*100:.1f}",
        ])
with open(output_dir / "regime_table.csv", "w", newline="") as f:
    csv.writer(f).writerows(rows_regime)
df_regime_tbl = pd.DataFrame(rows_regime[1:], columns=rows_regime[0])
print("\n=== Table 2: Regime-Stratified DA ===")
display(df_regime_tbl)

# ── Table 3: Ablation
rows_abl = [["Stage", "Loss components", "DA%", "MAE", "RMSE"]]
_abl_labels = {
    "Stage1_MSE_only":     "MSE only",
    "Stage2_MSE_Dir":      "MSE + Direction",
    "Stage3_MSE_Dir_Temp": "MSE + Direction + Temporal",
}
for stage, m in ablation_results.items():
    _da   = float(m.get("directional_accuracy", m.get("DA", 0)))
    _mae  = float(m.get("mae",  m.get("MAE",  0)))
    _rmse = float(m.get("rmse", m.get("RMSE", 0)))
    rows_abl.append([stage, _abl_labels.get(stage, stage),
                     f"{_da*100:.1f}", f"{_mae:.4f}", f"{_rmse:.4f}"])
with open(output_dir / "ablation_table.csv", "w", newline="") as f:
    csv.writer(f).writerows(rows_abl)
df_abl = pd.DataFrame(rows_abl[1:], columns=rows_abl[0])
print("\n=== Table 3: Ablation ===")
display(df_abl)

# ── Table 4: Niche accuracy
rows_niche = [["Specialist", "Domain condition", "n_samples", "DA%", "MAE"]]
_niche_desc = {
    "daily":      "Daytime, low turbulence (diurnal cycle)",
    "weather":    "High wind-speed gradient (frontal passage)",
    "mesoscale":  "Moderate turbulence (mesoscale variability)",
    "turbulence": "High turbulence intensity (gusty conditions)",
    "trend":      "Low turbulence (persistent slow trend)",
}
print(f"  Niche metrics keys: {list(niche_metrics.keys())}")
for spec, vals in niche_metrics.items():
    _spec_key = spec.lower().strip()  # normalise key
    _da  = float(vals.get("directional_accuracy", vals.get("DA", 0)))
    _mae = float(vals.get("mae", vals.get("MAE", 0)))
    _n   = int(vals.get("n_samples", 0))
    rows_niche.append([
        _spec_key.capitalize(),
        _niche_desc.get(_spec_key, ""),
        f"{_n:,}" if _n else "?",
        f"{_da*100:.1f}",
        f"{_mae:.4f}",
    ])
with open(output_dir / "niche_table.csv", "w", newline="") as f:
    csv.writer(f).writerows(rows_niche)
df_niche = pd.DataFrame(rows_niche[1:], columns=rows_niche[0])
print("\n=== Table 4: Niche Accuracy ===")
display(df_niche)

# ── Save all metrics JSON
with open(output_dir / "metrics_all.json", "w") as f:
    json.dump({k: {kk: float(vv) for kk, vv in v.items()}
               for k, v in all_metrics.items()}, f, indent=2)
print("\nAll tables saved to:", output_dir)


# ─────────────────────────────────────────────────────────────
# CELL 17 · Figures — publication-quality, journal-spec
# Energies (MDPI): 300 DPI, DejaVu Sans 9pt
# Single-col = 90 mm (3.54 in), Double-col = 190 mm (7.48 in)
# ─────────────────────────────────────────────────────────────

best_spec = min(SPECIALISTS, key=lambda s: float(all_metrics[s]["mae"]))
EVAL_ORDER_PLOT = ["Persistence", "LSTM_2layer", "CNN_BiLSTM"] + SPECIALISTS + ["Ensemble"]
_colors_base = {"Persistence": "#AAAAAA", "LSTM_2layer": "#888888",
                "CNN_BiLSTM": "#666666"}
_spec_colors = {"daily": "#5B9BD5", "weather": "#2E75B6",
                "mesoscale": "#1F4E79", "turbulence": "#70AD47",
                "trend": "#A9D18E"}
_ens_color   = "#ED7D31"

# ── Fig 1: DA comparison — overall vs ramp vs stable (grouped bar) ──────
fig, ax = plt.subplots(figsize=(COL2_W, COL2_W * 0.55))
_models_plot = EVAL_ORDER_PLOT
_x = np.arange(len(_models_plot))
_w = 0.26
_da_all    = [float(all_metrics[m]["directional_accuracy"]) for m in _models_plot]
_da_ramp   = [float(_reg_map.get(m, {}).get("DA_ramp",   0)) for m in _models_plot]
_da_stable = [float(_reg_map.get(m, {}).get("DA_stable", 0)) for m in _models_plot]
ax.bar(_x - _w, _da_all,    _w, label="Overall DA",  color="#5B9BD5", edgecolor="white")
ax.bar(_x,      _da_ramp,   _w, label="Ramp DA",     color="#ED7D31", edgecolor="white")
ax.bar(_x + _w, _da_stable, _w, label="Stable DA",   color="#A9D18E", edgecolor="white")
ax.set_xticks(_x)
ax.set_xticklabels([m.replace("_", "\n") for m in _models_plot], fontsize=7)
ax.set_ylabel("Directional Accuracy")
ax.set_ylim(0, 1.0)
ax.axhline(0.5, color="black", lw=0.6, ls=":", alpha=0.5)
ax.legend(fontsize=7, ncol=3, loc="upper left")
ax.set_title("Fig. 1 — Overall, Ramp-event and Stable-period DA by Model")
for xi, (a, r, s) in zip(_x, zip(_da_all, _da_ramp, _da_stable)):
    ax.text(xi - _w, a + 0.01, f"{a:.2f}", ha="center", va="bottom", fontsize=5.5, rotation=90)
    ax.text(xi,      r + 0.01, f"{r:.2f}", ha="center", va="bottom", fontsize=5.5, rotation=90)
    ax.text(xi + _w, s + 0.01, f"{s:.2f}", ha="center", va="bottom", fontsize=5.5, rotation=90)
plt.tight_layout()
plt.savefig(fig_dir / "fig1_da_grouped.png", dpi=300)
plt.show()
print("  ✓ fig1_da_grouped.png")

# ── Fig 2: MAE + RMSE dual bar (horizontal, sorted by MAE) ─────────────
fig, axes = plt.subplots(1, 2, figsize=(COL2_W, COL2_W * 0.45))
_sorted_models = sorted(EVAL_ORDER_PLOT, key=lambda m: float(all_metrics[m]["mae"]))
_maes  = [float(all_metrics[m]["mae"])  for m in _sorted_models]
_rmses = [float(all_metrics[m]["rmse"]) for m in _sorted_models]
_bar_colors = []
for m in _sorted_models:
    if m == "Ensemble":      _bar_colors.append(_ens_color)
    elif m in _spec_colors:  _bar_colors.append(_spec_colors[m])
    else:                    _bar_colors.append("#AAAAAA")
for ax, vals, title, xlabel in zip(
        axes, [_maes, _rmses], ["MAE", "RMSE"],
        ["MAE (normalised)", "RMSE (normalised)"]):
    bars = ax.barh(_sorted_models, vals, color=_bar_colors, edgecolor="white", height=0.6)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.invert_yaxis()
    for bar, val in zip(bars, vals):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=6.5)
    ax.set_xlim(0, max(vals) * 1.18)
plt.suptitle("Fig. 2 — MAE and RMSE: All Models (sorted by MAE)", y=1.01)
plt.tight_layout()
plt.savefig(fig_dir / "fig2_mae_rmse.png", dpi=300, bbox_inches="tight")
plt.show()
print("  ✓ fig2_mae_rmse.png")

# ── Fig 3: Scatter — best specialist predicted vs actual ────────────────
fig, ax = plt.subplots(figsize=(COL1_W, COL1_W))
bp = all_preds[best_spec]
bt = trues_common[:len(bp)]
ax.scatter(bt, bp, alpha=0.15, s=1.5, color="#1F4E79", rasterized=True)
mn, mx = float(bt.min()), float(bt.max())
ax.plot([mn, mx], [mn, mx], "r-", lw=1.0, label="1:1 line")
r_val = float(all_metrics[best_spec]["pearson_r"])
r2    = r_val ** 2
ax.set_xlabel("Measured power (p.u.)")
ax.set_ylabel("Forecast power (p.u.)")
ax.set_title(f"Fig. 3 — {best_spec.capitalize()} specialist\nr = {r_val:.3f},  R² = {r2:.3f}")
ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(fig_dir / "fig3_scatter.png", dpi=300)
plt.show()
print("  ✓ fig3_scatter.png")

# ── Fig 4: Time-series sample with 95% prediction interval ─────────────
L = min(288, len(tru_ens))   # 288 × 10 min = 48 h
t_axis = np.arange(L) * 10 / 60   # hours
fig, ax = plt.subplots(figsize=(COL2_W, COL2_W * 0.4))
ax.fill_between(t_axis, lower_t[:L], upper_t[:L],
                alpha=0.2, color=_ens_color, label=f"95% PI (ζ={ZETA})")
ax.plot(t_axis, tru_ens[:L],  color="black",    lw=1.0, label="Measured", zorder=5)
ax.plot(t_axis, ens_pred[:L], color=_ens_color, lw=0.9, ls="--",
        alpha=0.9, label="Ensemble forecast", zorder=4)
ax.set_xlabel("Time (hours)")
ax.set_ylabel("Active power (p.u.)")
ax.set_title("Fig. 4 — Ensemble forecast vs measured: 48-hour test window")
ax.legend(fontsize=7, loc="upper right")
plt.tight_layout()
plt.savefig(fig_dir / "fig4_forecast_sample.png", dpi=300)
plt.show()
print("  ✓ fig4_forecast_sample.png")

# ── Fig 5: Rolling DA over test period — ramp periods highlighted ───────
window = 144   # 144 × 10 min = 24 h rolling window
da_roll  = []
ramp_ens = np.abs(tru_ens - last_ens) > 0.05
for i in range(window, len(tru_ens)):
    seg_t  = tru_ens[i-window:i]
    seg_p  = ens_pred[i-window:i]
    seg_l  = last_ens[i-window:i]
    da_roll.append(float((np.sign(seg_p-seg_l)==np.sign(seg_t-seg_l)).mean()))
t_roll = np.arange(len(da_roll))
fig, ax = plt.subplots(figsize=(COL2_W, COL2_W * 0.4))
# Shade ramp-heavy windows
ramp_frac = np.array([ramp_ens[i-window:i].mean() for i in range(window, len(tru_ens))])
ax.fill_between(t_roll, 0, 1, where=ramp_frac > 0.5,
                alpha=0.12, color="#ED7D31", label="Ramp-dominated window")
ax.plot(t_roll, da_roll, color="#1F4E79", lw=0.8, label="Ensemble DA (24 h rolling)")
ax.axhline(float(all_metrics["Ensemble"]["directional_accuracy"]),
           color="black", lw=0.8, ls="--", label=f"Mean DA={float(all_metrics['Ensemble']['directional_accuracy'])*100:.1f}%")
ax.set_ylim(0, 1.0)
ax.set_xlabel("Test sample index")
ax.set_ylabel("Directional Accuracy")
ax.set_title("Fig. 5 — Rolling directional accuracy (24 h window) with ramp periods highlighted")
ax.legend(fontsize=7, loc="lower right")
plt.tight_layout()
plt.savefig(fig_dir / "fig5_da_rolling.png", dpi=300)
plt.show()
print("  ✓ fig5_da_rolling.png")

# ── Fig 6: Ramp vs stable DA — specialist comparison ───────────────────
fig, ax = plt.subplots(figsize=(COL1_W * 1.5, COL1_W * 1.3))
_spec_names = SPECIALISTS + ["Ensemble"]
_ramp_das   = [float(_reg_map.get(s, {}).get("DA_ramp",   0)) for s in _spec_names]
_stable_das = [float(_reg_map.get(s, {}).get("DA_stable", 0)) for s in _spec_names]
_sc_colors  = [_spec_colors.get(s, _ens_color) for s in _spec_names]
for i, (s, rd, sd) in enumerate(zip(_spec_names, _ramp_das, _stable_das)):
    ax.scatter(sd, rd, s=90, color=_sc_colors[i], zorder=3,
               edgecolors="white", linewidths=0.5)
    ax.annotate(s.capitalize(), (sd, rd),
                textcoords="offset points", xytext=(5, 3), fontsize=7)
# Add baselines
for bname in ["LSTM_2layer", "CNN_BiLSTM", "Persistence"]:
    _rd = float(_reg_map.get(bname, {}).get("DA_ramp",   0))
    _sd = float(_reg_map.get(bname, {}).get("DA_stable", 0))
    ax.scatter(_sd, _rd, s=60, color="#AAAAAA", zorder=3,
               marker="s", edgecolors="white", linewidths=0.5)
    ax.annotate(bname.replace("_","\n"), (_sd, _rd),
                textcoords="offset points", xytext=(5, 3), fontsize=6)
ax.axline((0,0), slope=1, color="black", lw=0.7, ls=":", alpha=0.5, label="Ramp=Stable")
ax.set_xlabel("Stable-period DA")
ax.set_ylabel("Ramp-event DA")
ax.set_title("Fig. 6 — Ramp vs stable DA per model\n(above diagonal = ramp specialisation)")
ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(fig_dir / "fig6_ramp_vs_stable.png", dpi=300)
plt.show()
print("  ✓ fig6_ramp_vs_stable.png")

# ── Fig 7: Horizon DA line plot ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(COL1_W * 1.4, COL1_W * 1.1))
_h_models = SPECIALISTS + ["Ensemble", "LSTM_2layer"]
_h_colors = [_spec_colors.get(m, "#888888") for m in SPECIALISTS] + [_ens_color, "#888888"]
_h_styles = ["-"]*len(SPECIALISTS) + ["-", "--"]
for mname, col, ls in zip(_h_models, _h_colors, _h_styles):
    _sub = df_h[df_h["model"] == mname].sort_values("horizon_min")
    ax.plot(_sub["horizon_min"], _sub["DA"], color=col, ls=ls, lw=1.2,
            marker="o", ms=4, label=mname.capitalize())
ax.set_xlabel("Forecast horizon (min)")
ax.set_ylabel("Directional Accuracy")
ax.set_title("Fig. 7 — DA vs forecast horizon (10–60 min)")
ax.legend(fontsize=7, ncol=2)
ax.set_xticks([10, 20, 30, 60])
plt.tight_layout()
plt.savefig(fig_dir / "fig7_horizon_da.png", dpi=300)
plt.show()
print("  ✓ fig7_horizon_da.png")

# ── Fig 8: Ablation — overall DA across the three loss stages ──────────
abl_labels_clean = [_abl_labels.get(s, s) for s in ablation_results.keys()]
abl_das_pct = [float(ablation_results[s].get("directional_accuracy",
               ablation_results[s].get("DA", 0)))*100
               for s in ablation_results.keys()]
fig, ax = plt.subplots(figsize=(COL1_W * 1.5, COL1_W))
_bar_cols_abl = ["#CCCCCC", "#5B9BD5", "#2E75B6"]
bars = ax.bar(range(len(abl_labels_clean)), abl_das_pct,
              color=_bar_cols_abl[:len(abl_labels_clean)], edgecolor="white", width=0.6)
ax.set_xticks(range(len(abl_labels_clean)))
ax.set_xticklabels(abl_labels_clean, rotation=15, ha="right", fontsize=7)
ax.set_ylabel("Directional Accuracy (%)")
ax.set_title("Fig. 8 — Loss-component ablation (daily specialist)")
_abl_min = min(abl_das_pct) - 0.5
_abl_max = max(abl_das_pct) + 0.5
ax.set_ylim(_abl_min, _abl_max)
for bar, val in zip(bars, abl_das_pct):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=7)
plt.tight_layout()
plt.savefig(fig_dir / "fig8_ablation.png", dpi=300)
plt.show()
print("  ✓ fig8_ablation.png")

# ── Fig 9: Uncertainty calibration — PICP vs zeta curve ────────────────
try:
    # Rebuild val predictions if Cell 13 was skipped (interval loaded from JSON)
    if "val_preds_per_spec" not in dir() or not val_preds_per_spec:
        print("  Rebuilding val predictions for calibration curve …")
        val_preds_per_spec = {}
        val_trues_ref      = None
        _val_idx_fig9      = make_indices(train_end, val_end, max_lb)
        for _spec9 in SPECIALISTS:
            _lb9   = ARCH[_spec9]
            _ck9   = torch.load(output_dir / f"best_{_spec9}_specialist.pt", map_location=device)
            _m9    = MODELS[_spec9](_ck9["lookback"], _ck9["input_dim"],
                                    BEST_PARAMS[_spec9]["dropout"]).to(device)
            _m9.load_state_dict(_ck9["model"])
            _, _pv9, _tv9, _ = run_eval(_m9, _lb9, _val_idx_fig9)
            val_preds_per_spec[_spec9] = _pv9
            if val_trues_ref is None: val_trues_ref = _tv9
        min_val_len = min(len(v) for v in val_preds_per_spec.values())
        val_trues_c = val_trues_ref[:min_val_len]
    _val_stack = np.stack([val_preds_per_spec[s][:min_val_len] for s in SPECIALISTS])
    _val_mean  = _val_stack.mean(axis=0)
    _val_std   = _val_stack.std(axis=0) + 1e-8
    _zeta_range = np.arange(0.1, 6.0, 0.1)
    _picps = []
    for _z in _zeta_range:
        _lo = _val_mean - _z * _val_std
        _hi = _val_mean + _z * _val_std
        _picps.append(float(np.mean((val_trues_c >= _lo) & (val_trues_c <= _hi))))
    fig, ax = plt.subplots(figsize=(COL1_W * 1.3, COL1_W))
    ax.plot(_zeta_range, [p*100 for p in _picps], color="#1F4E79", lw=1.2, label="Val PICP")
    ax.axhline(95, color="red", lw=0.8, ls="--", label="95% target")
    ax.axvline(ZETA, color="#ED7D31", lw=0.8, ls="--", label=f"ζ={ZETA}")
    ax.scatter([ZETA], [picp_test*100], color="#ED7D31", s=50, zorder=5,
               label=f"Test PICP={picp_test*100:.1f}%")
    ax.set_xlabel("ζ (interval width multiplier)")
    ax.set_ylabel("PICP (%)")
    ax.set_title("Fig. 9 — Prediction interval calibration curve")
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig9_calibration.png", dpi=300)
    plt.show()
    print("  ✓ fig9_calibration.png")
except Exception as _e:
    print(f"  – fig9_calibration skipped ({_e})")

print(f"\nAll figures saved to {fig_dir}")


# ─────────────────────────────────────────────────────────────
# CELL 18 · Auto-Generate Paper Snippets  [NEW]
# Pre-formatted numbers for direct paste into manuscript
# ─────────────────────────────────────────────────────────────

def _fmt_niche(nm):
    """Format niche_metrics for paper snippets — handles both live and JSON-loaded dicts."""
    lines = []
    for spec, vals in nm.items():
        da  = float(vals.get("directional_accuracy", vals.get("DA", 0))) * 100
        mae = float(vals.get("mae", vals.get("MAE", 0)))
        lines.append(f"  {spec:12s}: DA={da:.2f}%  MAE={mae:.4f}")
    return chr(10).join(lines) if lines else "  (none computed)"

def _fmt_ablation(ar):
    """Format ablation_results for paper snippets — handles both live and JSON-loaded dicts."""
    lines = []
    for s, vals in ar.items():
        da = float(vals.get("directional_accuracy", vals.get("DA", 0))) * 100
        lines.append(f"  {s:30s}: DA={da:.2f}%")
    return chr(10).join(lines)

def generate_paper_snippets():
    ens  = all_metrics["Ensemble"]
    lstm = all_metrics.get("LSTM_2layer", all_metrics["Persistence"])
    da_ens   = ens["directional_accuracy"]  * 100
    da_base  = lstm["directional_accuracy"] * 100
    improv   = da_ens - da_base
    mae_red  = (1 - ens["mae"] / lstm["mae"]) * 100
    rmse_red = (1 - ens["rmse"] / lstm["rmse"]) * 100
    ci_ens   = ci_results.get("Ensemble", {})
    dm_lstm  = dm_results.get("LSTM_2layer", dm_results.get("Persistence", {}))
    # Regime metrics
    import pandas as _pd
    try:
        _df_reg = _pd.read_csv(output_dir / "regime_metrics.csv")
        _ens_row = _df_reg[_df_reg.model == "Ensemble"].iloc[0]
        _per_row = _df_reg[_df_reg.model == "Persistence"].iloc[0]
        ens_ramp_pct    = float(_ens_row["DA_ramp"])   * 100
        ens_stable_pct  = float(_ens_row["DA_stable"]) * 100
        per_ramp_pct    = float(_per_row["DA_ramp"])   * 100
        n_ramp_pct      = float(_df_reg["n_ramp"].iloc[0]) / (float(_df_reg["n_ramp"].iloc[0]) + float(_df_reg["n_stable"].iloc[0])) * 100
    except Exception:
        ens_ramp_pct = ens_stable_pct = per_ramp_pct = n_ramp_pct = float("nan")

    snippet = f"""
{'='*65}
PAPER SNIPPETS  —  generated {datetime.now().strftime('%Y-%m-%d %H:%M')}
{'='*65}

── ABSTRACT (key numbers) ──────────────────────────────────────
Directional Accuracy (Ensemble):  {da_ens:.2f}%
Improvement over LSTM baseline:   {improv:.1f} pp
MAE (normalised):                  {ens['mae']:.3f}
MAE reduction:                     {mae_red:.1f}%
RMSE reduction:                    {rmse_red:.1f}%
Pearson r:                         {ens['pearson_r']:.3f}
Turning-point accuracy:            {ens['turning_point_accuracy']*100:.1f}%
Weighted DA:                       {ens['weighted_directional_accuracy']*100:.2f}%

── TABLE 1 (Ensemble row) ─────────────────────────────────────
DA   = {ens['directional_accuracy']*100:.2f}%  [{ci_ens.get('DA_lo',0)*100:.2f}%, {ci_ens.get('DA_hi',0)*100:.2f}%]  (95% bootstrap CI)
WDA  = {ens['weighted_directional_accuracy']*100:.2f}%
Trend= {ens['trend_accuracy']*100:.2f}%
TPA  = {ens['turning_point_accuracy']*100:.2f}%
MAE  = {ens['mae']:.3f}
RMSE = {ens['rmse']:.3f}
r    = {ens['pearson_r']:.4f}

── UNCERTAINTY / SECTION 3.6 ──────────────────────────────────
Calibrated ζ:      {ZETA}
Test PICP (95%):   {picp_test:.4f}
Test PINAW:        {pinaw_test:.4f}

── STATISTICAL SIGNIFICANCE ──────────────────────────────────
Diebold-Mariano vs LSTM_2layer:  DM={dm_lstm.get('DM',float('nan')):+.3f}, p={dm_lstm.get('p',float('nan')):.4f} {dm_lstm.get('sig','')}
(Positive DM = ensemble significantly better)

── NICHE ACCURACY (§4.2) ─────────────────────────────────────
{_fmt_niche(niche_metrics)}

── ABLATION TABLE (§6.4) ─────────────────────────────────────
{_fmt_ablation(ablation_results)}

── REGIME-STRATIFIED DA (paper headline) ──────────────────────
Ensemble ramp DA    = {ens_ramp_pct:.1f}%   (vs Persistence = {per_ramp_pct:.1f}%)
Ensemble stable DA  = {ens_stable_pct:.1f}%
Ramp events in test = {n_ramp_pct:.1f}% of samples

── AI TOOL DECLARATION ────────────────────────────────────────
During the preparation of this work, the authors used AI-assisted
tools in order to support code development and manuscript drafting.
After using these tools, the authors reviewed and edited all content
as needed and take full responsibility for the published article.

{'='*65}
"""
    print(snippet)
    with open(output_dir / "paper_snippets.txt", "w") as f:
        f.write(snippet)
    print(f"Saved → {output_dir / 'paper_snippets.txt'}")

generate_paper_snippets()


# ─────────────────────────────────────────────────────────────
# CELL 19 · Final report
# ─────────────────────────────────────────────────────────────
report = f"""# Wind Forecast — Enhanced Pipeline Results
**Generated:** {datetime.now().isoformat()}

## Outputs
| File | Description |
|------|-------------|
| results_table.csv         | Main Table 1 with bootstrap CIs |
| ablation_table.csv        | Loss-component ablation stages |
| horizon_metrics.csv       | DA / MAE per forecast horizon (10–60 min, within trained range) |
| niche_metrics.json        | Per-specialist domain-filtered accuracy |
| ablation_results.json     | Full metrics per ablation stage |
| interval_calibration.json | ζ={ZETA}, PICP={picp_test:.4f}, PINAW={pinaw_test:.4f} |
| bootstrap_ci.json         | 95% confidence intervals on DA |
| dm_tests.json             | Diebold-Mariano test results |
| metrics_all.json          | All model metrics |
| paper_snippets.txt        | Pre-formatted numbers for manuscript |
| figures/*.png             | Journal-spec 300 DPI figures |

## Best Specialist
- {best_spec}: MAE={all_metrics[best_spec]['mae']:.6f}  DA={all_metrics[best_spec]['directional_accuracy']:.4f}

## Ensemble
- MAE={ens_m['mae']:.6f}  DA={ens_m['directional_accuracy']:.4f}
- 95% PI: PICP={picp_test:.4f}  PINAW={pinaw_test:.4f}  ζ={ZETA}
"""
with open(output_dir / "REPORT.md", "w") as f:
    f.write(report)
print(report)
print(f"\nAll outputs saved to: {output_dir}")
