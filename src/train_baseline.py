import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

from config import TRAIN_DATA_PATH, TEST_DATA_PATH, MODELS_DIR, FEATURE_COLUMNS, TARGET_COLUMN, SEASONS_ORDER


def run_baseline_training():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("1. Wczytywanie zbiorów treningowego i testowego...")
    df_train = pd.read_parquet(TRAIN_DATA_PATH)
    df_test = pd.read_parquet(TEST_DATA_PATH)

    X_train, y_train = df_train[FEATURE_COLUMNS], df_train[TARGET_COLUMN]
    X_test, y_test = df_test[FEATURE_COLUMNS], df_test[TARGET_COLUMN]

    # Baseline 0: Surowe ERA5-Land
    y_era5 = df_test['temp_era5']
    mae_era5 = mean_absolute_error(y_test, y_era5)
    rmse_era5 = np.sqrt(mean_squared_error(y_test, y_era5))
    r2_era5 = r2_score(y_test, y_era5)
    bias_era5 = (y_era5 - y_test).mean()

    print("\n" + "=" * 65)
    print("📊 BASELINE 0 (Surowe ERA5-Land na zbiorze testowym)")
    print("=" * 65)
    print(f"MAE:   {mae_era5:.3f} °C")
    print(f"RMSE:  {rmse_era5:.3f} °C")
    print(f"R²:    {r2_era5:.4f}")
    print(f"BIAS:  {bias_era5:.3f} °C")

    # Scikit-Learn Pipeline: Random Forest
    rf_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(
            n_estimators=100,
            max_depth=16,
            min_samples_leaf=4,
            n_jobs=-1,
            random_state=42
        ))
    ])

    print("\n2. Trenowanie modelu Random Forest...")
    rf_pipeline.fit(X_train, y_train)

    y_pred = rf_pipeline.predict(X_test)
    mae_rf = mean_absolute_error(y_test, y_pred)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred))
    r2_rf = r2_score(y_test, y_pred)
    bias_rf = (y_pred - y_test).mean()

    print("\n" + "=" * 65)
    print("🌲 RANDOM FOREST PIPELINE (Zbiór testowy - 42 dni)")
    print("=" * 65)
    print(f"MAE:   {mae_rf:.3f} °C")
    print(f"RMSE:  {rmse_rf:.3f} °C")
    print(f"R²:    {r2_rf:.4f}")
    print(f"BIAS:  {bias_rf:.3f} °C")

    # Rozbicie na pory roku Gumińskiego
    df_eval = df_test.copy()
    df_eval['pred_rf'] = y_pred

    print("\nWyniki Random Forest w rozbiciu na pory roku:")
    season_rows = []
    for s in SEASONS_ORDER:
        sub = df_eval[df_eval['guminski_season'] == s]
        if len(sub) > 0:
            m_s = mean_absolute_error(sub[TARGET_COLUMN], sub['pred_rf'])
            rm_s = np.sqrt(mean_squared_error(sub[TARGET_COLUMN], sub['pred_rf']))
            r2_s = r2_score(sub[TARGET_COLUMN], sub['pred_rf'])
            b_s = (sub['pred_rf'] - sub[TARGET_COLUMN]).mean()
            season_rows.append({
                'Pora roku': s,
                'Próbek': len(sub),
                'MAE [°C]': round(m_s, 3),
                'RMSE [°C]': round(rm_s, 3),
                'R²': round(r2_s, 4),
                'BIAS [°C]': round(b_s, 3)
            })

    print(pd.DataFrame(season_rows).to_string(index=False))
    print("=" * 65)

    out_path = MODELS_DIR / "rf_pipeline.joblib"
    joblib.dump(rf_pipeline, out_path)
    print(f"\n✅ Zapisano model: {out_path}")


if __name__ == '__main__':
    run_baseline_training()