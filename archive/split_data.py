import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import DATASET_2022_PATH, TRAIN_DATA_PATH, TEST_DATA_PATH, SEASONS_ORDER


def create_train_test_split():
    print(f"1. Wczytywanie zbioru danych: {DATASET_2022_PATH}...")
    df = pd.read_parquet(DATASET_2022_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date

    test_days = []
    print("\n--- WYBRANE TYGODNIE TESTOWE DLA PÓR ROKU GUMIŃSKIEGO ---")

    for season in SEASONS_ORDER:
        season_df = df[df['guminski_season'] == season]

        if season == 'zima':
            zima_early = season_df[season_df['timestamp'].dt.month <= 3]
            days = sorted(zima_early['date'].unique())
            last_7 = days[-7:]
            print(f"- {'Zima (luty)':<18}: od {last_7[0]} do {last_7[-1]} (7 dni - koniec zimy 2021/22)")
        else:
            days = sorted(season_df['date'].unique())
            last_7 = days[-7:]
            print(f"- {season.capitalize():<18}: od {last_7[0]} do {last_7[-1]} (7 dni)")

        test_days.extend(last_7)

    is_test = df['date'].isin(test_days)
    df_train = df[~is_test].drop(columns=['date']).copy().reset_index(drop=True)
    df_test = df[is_test].drop(columns=['date']).copy().reset_index(drop=True)

    print("\n--- BILANS PODZIAŁU DANYCH ---")
    print(f"Liczba wierszy łącznie: {len(df):,}")
    print(f"Zbiór TRENINGOWY:       {len(df_train):,} ({len(df_train) / len(df) * 100:.1f}%)")
    print(f"Zbiór TESTOWY (42 dni): {len(df_test):,} ({len(df_test) / len(df) * 100:.1f}%)")

    df_train.to_parquet(TRAIN_DATA_PATH, index=False)
    df_test.to_parquet(TEST_DATA_PATH, index=False)
    print(f"\n✅ Zapisano: {TRAIN_DATA_PATH}")
    print(f"✅ Zapisano: {TEST_DATA_PATH}")

    # Baseline 0 (Surowe ERA5-Land vs pomiary naziemne)
    y_true = df_test['temp_ground']
    y_era5 = df_test['temp_era5']

    mae_total = mean_absolute_error(y_true, y_era5)
    rmse_total = np.sqrt(mean_squared_error(y_true, y_era5))
    r2_total = r2_score(y_true, y_era5)
    bias_total = (y_era5 - y_true).mean()

    print("\n" + "=" * 65)
    print("📊 BASELINE 0: SUROWE ERA5-LAND (Zbiór testowy - 42 dni)")
    print("=" * 65)
    print(f"MAE:   {mae_total:.3f} °C")
    print(f"RMSE:  {rmse_total:.3f} °C")
    print(f"R²:    {r2_total:.4f}")
    print(f"BIAS:  {bias_total:.3f} °C")

    print("\nRozbicie błędu surowego ERA5 na poszczególne pory roku:")
    season_rows = []
    for s in SEASONS_ORDER:
        sub = df_test[df_test['guminski_season'] == s]
        if len(sub) > 0:
            mae_s = mean_absolute_error(sub['temp_ground'], sub['temp_era5'])
            rmse_s = np.sqrt(mean_squared_error(sub['temp_ground'], sub['temp_era5']))
            r2_s = r2_score(sub['temp_ground'], sub['temp_era5'])
            bias_s = (sub['temp_era5'] - sub['temp_ground']).mean()
            season_rows.append({
                'Pora roku': s,
                'Próbek': len(sub),
                'MAE [°C]': round(mae_s, 3),
                'RMSE [°C]': round(rmse_s, 3),
                'R²': round(r2_s, 4),
                'BIAS [°C]': round(bias_s, 3)
            })

    print(pd.DataFrame(season_rows).to_string(index=False))
    print("=" * 65)


if __name__ == '__main__':
    create_train_test_split()