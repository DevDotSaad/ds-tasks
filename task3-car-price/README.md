# Task 3 — Car Price Prediction (Modular Version)

## Project Structure
```
Task3_CarPricePrediction_Modular/
 ├── main.py
 ├── preprocessing.py
 ├── models.py
 ├── utils.py
 ├── requirements.txt
 ├── README.md
 ├── data/
 │    └── cars.csv
 └── artifacts/
```

## How to Run
1. Put your dataset as `data/cars.csv` (must include a `price` column).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run training:
   ```bash
   python main.py
   ```
4. Outputs will appear in `artifacts/`:
   - Metrics (`metrics.json`)
   - Residual plots
   - Feature importance plot
   - Trained models & preprocessor
