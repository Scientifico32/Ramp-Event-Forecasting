# Ramp-Event Directional Forecasting for Wind Power Integration

**A Regime-Stratified Ensemble with Direction-Focused Training**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-orange.svg)](https://pytorch.org/)

---

## Paper

> **Ramp-Event Directional Forecasting for Wind Power Integration:
> A Regime-Stratified Ensemble with Direction-Focused Training**
>
> Konstantinos Stergiou¹, Theodoros E. Karakasidis²
>
> ¹ University of Thessaly, Department of Civil Engineering · ² University of Thessaly, Department of Physics
>
> Submitted to *Energies* (MDPI), 2026 — under review
>
> **DOI:** [to be assigned upon acceptance]

### Abstract

Wind power ramp events — sudden large changes in output driven by frontal
passages, sea-breeze transitions, and turbulence — are a principal challenge
for grid operators integrating variable renewable energy. This work proposes a
regime-stratified evaluation framework that separately quantifies ramp-event
directional accuracy (ramp-DA) and stable-period DA, and a specialist ensemble
of five regime-specialised models trained with a direction-focused loss that
explicitly penalises errors in the sign of the forecast power change, using only
on-site SCADA wind-speed and active-power signals. On 33,411 held-out samples
from three Greek onshore wind farms, the ensemble achieves **76.5% ramp-event
directional accuracy** versus 0.0% for persistence and 70.1% for a 2-layer LSTM
baseline (Diebold–Mariano p < 0.001). Overall DA is 57.9% — modest by design, as
the loss deliberately concentrates gains on the 46.1% of test steps that are
ramp events.

---

## Key Results

| Model | Overall DA (%) | Ramp-DA (%) | Stable-DA (%) | MAE (p.u.) | Pearson r |
|-------|:--------------:|:-----------:|:-------------:|:----------:|:---------:|
| Persistence | 13.9 | **0.0** ★ | 25.7 | 0.096 | 0.792 |
| 2-layer LSTM | 54.5 | 70.1 | 41.3 | 0.110 | 0.774 |
| CNN-BiLSTM | 55.7 | 71.4 | 42.3 | 0.108 | 0.769 |
| **Ensemble** | **57.9** [57.4, 58.4] | **76.5** [76.0, 77.0] | 42.0 | **0.089** | **0.848** |

★ Persistence ramp-DA = 0.0% by construction (always predicts Δpower = 0).
Bootstrap 95% CIs in brackets. All ensemble vs. baseline differences: DM p < 0.001.

---

## Repository Structure

```
├── wind_pipeline.py          Main training & evaluation pipeline (Colab)
├── make_figures.py           Generate the eight manuscript figures from pipeline outputs
├── src/
│   ├── models/
│   │   └── specialists.py    Five specialist nn.Module architectures
│   ├── losses/
│   │   └── multiobjective.py Direction-focused multi-objective loss
│   └── eval/
│       └── metrics.py        All evaluation metrics (DA, ramp-DA, WDA, TPA, ...)
├── data/
│   └── README.md             Data availability statement
├── notebooks/
│   └── quickstart.ipynb      Short Colab demo on synthetic data
├── requirements.txt
└── LICENSE
```

---

## Installation

```bash
git clone https://github.com/USERNAME/REPO.git
cd REPO
pip install -r requirements.txt
```

*(Replace `USERNAME/REPO` with the actual repository path.)*

---

## Usage

### Training (Google Colab — A100 GPU recommended)

1. Clone the repository in Colab:
```python
!git clone https://github.com/USERNAME/REPO.git
%cd REPO
```

2. Edit the data paths in `wind_pipeline.py`:
```python
DRIVE_PROJECT_PATH = "/content/drive/MyDrive/your/path"
SCADA_PATH         = "/content/drive/MyDrive/your/SCADA/files"
```

3. Run all cells (`Runtime → Run all`). Training checkpoints so it can resume
   after a Colab disconnection.

### Generating the figures

The figures are produced by `make_figures.py`, which reads the arrays produced
by the evaluation pipeline (predictions, targets, per-model metrics, training
histories, ablation results, and prediction-interval bounds) and renders the
eight manuscript figures directly from those outputs. It does not synthesise
data; it requires the pipeline outputs, which in turn depend on the proprietary
SCADA records.

```python
# after the pipeline has run in the same session:
%run make_figures.py
# → writes Fig1_architecture.png … Fig8_PI_calibration.png to ./figures/
```

### Using the src/ modules independently

```python
import sys
sys.path.insert(0, ".")

from src.models.specialists import (
    DailyPatternSpecialist, WeatherFrontSpecialist,
    MesoscaleSpecialist, TurbulenceSpecialist, TrendSpecialist,
)
from src.losses.multiobjective import multi_objective_loss
from src.eval.metrics import compute_all_metrics

import torch

# Instantiate the weather-front specialist
model = WeatherFrontSpecialist(
    lookback=48, forecast_steps=6, input_dim=20,
    hidden=256, num_heads=8, ff_dim=256, dropout=0.185
)

# Forward pass
x   = torch.randn(32, 48, 20)   # (batch, lookback, features)
out = model(x)                  # (32, 6) — 6-step forecast
```

---

## Specialist Architectures

| Specialist | Architecture | Target Regime | Lookback |
|------------|-------------|---------------|----------|
| Daily | 4-layer BiLSTM | Sea-breeze / diurnal ramps | 48 steps (8 h) |
| Weather-Front | Transformer encoder (6L, 8H) | Synoptic frontal passages | 48 steps (8 h) |
| Mesoscale | Multi-scale CNN (k=3,6,12) + LSTM | Orographic channelling | 36 steps (6 h) |
| Turbulence | 3-layer GRU + temporal attention | Sub-hourly turbulent gusts | 12 steps (2 h) |
| Trend | 4-layer FeedForward + skip connections | Slow multi-hour ramps | 72 steps (12 h) |

All five specialists are configured to capture distinct ramp regimes. The
specialist names denote the ramp dynamics each sub-model targets, **not**
external meteorological inputs — every feature is engineered from the two
on-site SCADA streams (wind speed and active power).

---

## Multi-Objective Direction-Focused Loss

Each specialist is trained with a three-term loss:

```
L = α·L_mag + β·L_dir + γ·L_temp
```

- **L_mag** — mean squared error on the predicted power (anchors magnitude).
- **L_dir** — a margin-based penalty on the agreement between the predicted and
  observed *direction* of the power change (each taken relative to the last
  observed value). It penalises predictions whose sign disagrees with the
  observed movement by at least a fixed margin, concentrating learning pressure
  on correct ramp direction.
- **L_temp** — a temporal-smoothness penalty on the first difference of
  successive predicted steps, discouraging implausible step-to-step oscillation.

The three weights (α, β, γ) are tuned per specialist by grid search on the
validation set:

| Specialist | α | β | γ | LR |
|------------|---|---|---|----|
| daily | 1.32 | 0.48 | 0.036 | 5.89e-4 |
| weather | 1.60 | 1.25 | 0.148 | 1.04e-4 |
| mesoscale | 1.16 | 1.19 | 0.037 | 3.29e-3 |
| turbulence | 1.42 | 0.45 | 0.146 | 5.40e-4 |
| trend | 1.91 | 0.63 | 0.305 | 3.03e-4 |

---

## Data Availability

The SCADA data used in this study are proprietary to **TERNA Energy S.A.**
(three onshore wind farms, Peloponnese, Greece, **January 2017 – May 2018**) and
are subject to commercial confidentiality; they cannot be publicly deposited.
To respect data-confidentiality obligations, the three sites are referred to
only as Farms A–C and their precise locations are withheld. Requests for data
access should be directed to the corresponding author and are subject to the
permission of TERNA Energy S.A.

The model relies **only** on on-site SCADA wind-speed and active-power signals;
it does not ingest any exogenous meteorological context (numerical-weather-
prediction fields, reanalysis data, or upstream observations). The model code,
training configuration, and figure-generation scripts are provided in this
repository. See `data/README.md` for details.

---

## Citation

```bibtex
@article{stergiou2026ramp,
  title   = {Ramp-Event Directional Forecasting for Wind Power Integration:
             A Regime-Stratified Ensemble with Direction-Focused Training},
  author  = {Stergiou, Konstantinos and Karakasidis, Theodoros E.},
  journal = {Energies},
  year    = {2026},
  publisher = {MDPI},
  note    = {Under review},
}
```


---

## License

MIT License — see [LICENSE](LICENSE) for details.

