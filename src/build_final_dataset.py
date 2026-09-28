import pandas as pd
from config import PROCESSED_DATA_DIR, DATASET_2022_PATH
from climatology import add_guminski_seasons_harmonic


def build_final_dataset():
    imgw_path = PROCESSED_DATA_DIR / "imgw_dataset.parquet"
    lcs_path = PROCESSED_DATA_DIR / "lcs_dataset.parquet"

    # Fallback na format CSV w razie potrzeby
    if not imgw_path.exists():
        imgw_path = PROCESSED_DATA_DIR / "imgw_dataset.csv"
    if not lcs_path.exists():
        lcs_path = PROCESSED_DATA_DIR / "lcs_dataset.csv"

    print("1. Wczytywanie przetworzonych zbiorów IMGW i LCS...")
    df_imgw = pd.read_parquet(imgw_path) if str(imgw_path).endswith('.parquet') else pd.read_csv(imgw_path)
    df_lcs = pd.read_parquet(lcs_path) if str(lcs_path).endswith('.parquet') else pd.read_csv(lcs_path)

    # Oznaczenie pochodzenia danych
    df_imgw['data_source'] = 'imgw'
    df_lcs['data_source'] = 'lcs'

    base_columns = [
        'timestamp', 'station', 'data_source', 'lat', 'lon',
        'altitude', 'urban_fraction_500m',
        'sin_hour', 'cos_hour', 'sin_doy', 'cos_doy',
        'temp_era5', 'temp_ground'
    ]

    # Bezpieczne połączenie
    df_merged = pd.concat([df_imgw[base_columns], df_lcs[base_columns]], ignore_index=True)
    df_merged['timestamp'] = pd.to_datetime(df_merged['timestamp'])
    df_merged.sort_values(by=['timestamp', 'station'], inplace=True)
    df_merged.reset_index(drop=True, inplace=True)

    # 2. Klasyfikacja termicznych pór roku Gumińskiego (analiza harmoniczna fourierowska)
    print("2. Wyznaczanie pór roku Gumińskiego (model harmoniczny)...")
    df_merged = add_guminski_seasons_harmonic(
        df_merged,
        timestamp_col='timestamp',
        temp_col='temp_era5'
    )

    # Porządkowanie kolumn wyjściowych
    target_columns = [
        'timestamp', 'station', 'data_source', 'lat', 'lon',
        'altitude', 'urban_fraction_500m',
        'sin_hour', 'cos_hour', 'sin_doy', 'cos_doy',
        'guminski_season', 'temp_era5', 'temp_ground'
    ]
    df_merged = df_merged[target_columns].copy()

    # 3. Kontrola jakości i braków danych
    print("\n--- RAPORT INTEGRALNOŚCI ZBIORU ---")
    print(f"Łączna liczba obserwacji: {len(df_merged):,}")
    print(f"Unikalne stacje ({df_merged['station'].nunique()}): {list(df_merged['station'].unique())}")

    null_sum = df_merged.isna().sum().sum()
    if null_sum > 0:
        print(f"Usunięto wiersze z wartościami NaN ({null_sum}).")
        df_merged.dropna(inplace=True)
        df_merged.reset_index(drop=True, inplace=True)
    else:
        print("Kompletność danych: 100% (brak wartości NaN).")

    # 4. Zapis zbioru pełnego (2000–2025)
    out_all_parquet = PROCESSED_DATA_DIR / "final_dataset.parquet"
    out_all_csv = PROCESSED_DATA_DIR / "final_dataset.csv"
    df_merged.to_parquet(out_all_parquet, index=False)
    df_merged.to_csv(out_all_csv, index=False)

    # 5. Zapis wspólnego zbioru eksperymentalnego dla roku 2022 (LCS + IMGW)
    df_2022 = df_merged[df_merged['timestamp'].dt.year == 2022].copy().reset_index(drop=True)
    out_2022_csv = PROCESSED_DATA_DIR / "final_dataset_2022.csv"
    df_2022.to_parquet(DATASET_2022_PATH, index=False)
    df_2022.to_csv(out_2022_csv, index=False)

    print("\n" + "=" * 60)
    print("✅ POMYŚLNIE ZBUDOWANO FINALNE ZBIORY:")
    print(f"1. Pełny zbiór (2000–2025): {out_all_parquet} ({len(df_merged):,} wierszy)")
    print(f"2. Zbiór 2022 roku:         {DATASET_2022_PATH} ({len(df_2022):,} wierszy)")
    print("=" * 60)
    print("\nRozkład obserwacji w porach roku (2022):")
    print(df_2022['guminski_season'].value_counts())


if __name__ == '__main__':
    build_final_dataset()