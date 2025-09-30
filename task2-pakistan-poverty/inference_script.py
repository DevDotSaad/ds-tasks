import argparse, os, pickle
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', nargs='*', type=int, required=True, help='Years to predict')
    ap.add_argument('--artifacts_dir', type=str, default='artifacts')
    args = ap.parse_args()

    with open(os.path.join(args.artifacts_dir, 'model.pkl'), 'rb') as f:
        bundle = pickle.load(f)
    model = bundle['model']
    feat_names = bundle['features']
    nlags = bundle['nlags']

    years = sorted(args.years)
    df = pd.DataFrame({'year': years})
   
    for c in ['t','unemployment_rate_pct','population_millions','gdp_growth_rate_pct','inflation_rate_pct','real_growth_proxy']:
        if c not in df.columns:
            df[c] = 0.0
    for L in range(1, nlags+1):
        df[f'headcount_ratio_national_lag{L}'] = 0.0
    X = df[feat_names]
    preds = model.predict(X)
    out = pd.DataFrame({'year': years, 'pred_headcount_ratio_national': preds})
    print(out.to_csv(index=False))

if __name__ == '__main__':
    main()
