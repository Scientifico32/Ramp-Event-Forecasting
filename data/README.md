# Ramp-Event Directional Forecasting for Wind Power Integration

Code and reproducibility materials for *"Ramp-Event Directional Forecasting for
Wind Power Integration: A Regime-Stratified Ensemble with Direction-Focused
Training"* (Stergiou & Karakasidis).

---

## Data Availability

### SCADA Data

The SCADA data used in this study are proprietary to **TERNA Energy S.A.**
and were provided under a research data-use agreement for academic purposes.
The data are subject to commercial confidentiality and cannot be publicly
deposited.

**Three onshore wind farms, Peloponnese, Greece (January 2017 – May 2018):**

| Site | Terrain type | Primary ramp driver |
|------|-------------|---------------------|
| Farm A | Coastal | Sea-breeze diurnal cycles |
| Farm B | Inland valley | Thermally driven valley winds |
| Farm C | Orographic | Gap-flow acceleration |

To respect data-confidentiality obligations, the three sites are referred to
only as Farms A–C and their precise locations are withheld.

**Resolution:** 10-minute SCADA intervals
**Input signals:** wind speed and active power (the two on-site SCADA streams
from which all 20 model features are engineered)

The model uses **only** on-site SCADA wind-speed and active-power signals. It does
**not** ingest any exogenous meteorological context (numerical-weather-prediction
fields, reanalysis data, or upstream observations); every feature is computed from
past on-site measurements available in real time.

**Requests for data access** should be directed to the corresponding author and
are subject to the permission of TERNA Energy S.A.

---

## What IS Provided

| Item | Location | Description |
|------|----------|-------------|
| Preprocessing code | `wind_pipeline.py` | Full feature-engineering pipeline |
| Model architectures | `src/models/specialists.py` | All five specialist `nn.Module` classes |
| Training pipeline | `wind_pipeline.py` | Complete training with checkpointing |
| Evaluation metrics | `src/eval/metrics.py` | All manuscript metrics |
| Figure generation | `make_real_figures.py`, `make_fig5_option_b.py`, `make_real_figures_5_6_10.py` | Regenerate the paper figures from real pipeline outputs |

The figure-generation scripts read the actual arrays produced by the evaluation
pipeline (predictions, targets, per-model metrics, training histories, ablation
results) and render the eight figures in the paper directly from those outputs.
They do not synthesise data; running them requires the pipeline outputs, which in
turn depend on the proprietary SCADA records.

---

## Preprocessing Pipeline

The preprocessing pipeline reads raw Excel SCADA files (wind speed and active
power) and produces a 20-feature matrix. All twenty features are derived
exclusively from the two on-site SCADA signals:

```
[0]  wind_speed              [10] power_lag1
[1]  active_power (TARGET)   [11] power_lag6
[2]  hour_sin                [12] power_lag12
[3]  hour_cos                [13] power_lag48
[4]  month_sin               [14] ws_rolling_mean6
[5]  month_cos               [15] ws_rolling_std6
[6]  ws_lag1                 [16] ws_rolling_mean48
[7]  ws_lag6                 [17] power_diff1
[8]  ws_lag12                [18] ws_variability_ratio
[9]  ws_lag48                [19] ws_cubed
```

Quality control (duplicate-timestamp removal, alignment to a common 10-minute
grid, time-based interpolation of gaps, and clipping of physically invalid
negatives) retained 98.7% of raw timesteps.

**Train / validation / test split:** 70% / 15% / 15%, chronological. The three
farms are concatenated before the split; the validation and test sets fall
entirely within the later, unseen period of Farm C, giving a test set of 33,411
samples. A small early-overlap of the evaluation farm (4.8% of training samples)
is disclosed in the manuscript for full transparency.

---

## Feature Notes

- `power_diff1` — first difference of active power (a ramp indicator).
- `ws_variability_ratio` — wind-speed rolling standard deviation divided by
  rolling mean.
- `ws_cubed` — wind speed cubed (proportional to the kinetic energy of the flow).
- The specialist names used in the code and paper (e.g. *weather-front*,
  *mesoscale*) denote the **ramp regimes** each sub-model is tuned to capture,
  **not** external meteorological inputs.

All features are scaled to [0, 1] using a min-max scaler fitted on the training
split only, to avoid information leakage.
