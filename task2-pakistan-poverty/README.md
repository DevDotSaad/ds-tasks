
# Task 2 — Pakistan Poverty (2000–2023)

**Pipeline order:** setup → EDA → modeling/forecasting → insights → visuals → artifacts → inference script → README.

## Quick start
```bash
pip install -r requirements.txt
python run_all.py --data data/Pakistan_Poverty_Dataset_2000_2023.csv --forecast 5
```

## Files
- `data/` — place `Pakistan_Poverty_Dataset_2000_2023.csv` here.
- `src/` — code (`data_utils.py`, `eda.py`, `train.py`, `forecast.py`).
- `run_all.py` — end-to-end runner (trains + EDA + forecast).
- `outputs/` — `plots/` and `forecasts.csv`.
- `artifacts/` — `model.pkl`, `metrics.json`, `columns.json`.
- `inference_script.py` — CLI predictions for custom years.

## Model (upgraded)
- `RidgeCV` with lag features (`t-1`, `t-2`), time trend `t`, macros: unemployment, population, GDP growth, inflation, and `real_growth_proxy`.
- Recursive 5-year forecasting.

## Inference (ad‑hoc)
```bash
python inference_script.py --years 2024 2025 2026 --artifacts_dir artifacts
```
