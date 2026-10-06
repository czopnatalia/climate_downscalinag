import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer
from scipy.spatial import cKDTree

from config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR,
    DEM_RASTER_PATH, LAND_COVER_PATH, ERA5_NETCDF_PATH,
    KRAKOW_BBOX
)
from utils_geo import sample_dem, compute_urban_fraction
from utils_time import localize_to_utc, add_cyclic_temporal_features


def process_lcs_data():
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    excel_path = RAW_DATA_DIR / "lcs" / "lokalizacja_nazwy_miast_LCS.xlsx"
    parquet_path = RAW_DATA_DIR / "lcs" / "dane_z_lokalizacja_godzinne.parquet"

    # 1. Wczytanie i standaryzacja geometrii stacji LCS
    print("1. Wczytywanie metadanych stacji LCS...")
    xl = pd.ExcelFile(excel_path)
    meta_dfs = [pd.read_excel(excel_path, sheet_name=s) for s in ['KRK', 'ALL_sensors'] if s in xl.sheet_names]
    stations_df = pd.concat(meta_dfs, ignore_index=True).dropna(subset=['X', 'Y', 'Localization'])
    stations_df.rename(columns={'Localization': 'station'}, inplace=True)
    stations_df = stations_df.drop_duplicates(subset=['station']).copy()

    # Transformacja współrzędnych: UTM 34N -> WGS84
    tr_utm = Transformer.from_crs("EPSG:32634", "EPSG:4326", always_xy=True)
    stations_df['lon'], stations_df['lat'] = tr_utm.transform(stations_df['X'].values, stations_df['Y'].values)

    # Filtracja przestrzenna do obszaru Krakowa
    stations_df = stations_df[
        (stations_df['lon'] >= KRAKOW_BBOX['west']) & (stations_df['lon'] <= KRAKOW_BBOX['east']) &
        (stations_df['lat'] >= KRAKOW_BBOX['south']) & (stations_df['lat'] <= KRAKOW_BBOX['north'])
    ].copy()

    # 2. Próbkowanie cech ze wspólnego modułu utils_geo
    print("2. Próbkowanie DEM i wyznaczanie Urban Fraction (500m)...")
    stations_df['altitude'] = sample_dem(DEM_RASTER_PATH, stations_df['lat'].values, stations_df['lon'].values)
    stations_df['urban_fraction_500m'] = compute_urban_fraction(LAND_COVER_PATH, stations_df['lat'].values, stations_df['lon'].values)

    stations_df = stations_df.dropna(subset=['altitude', 'urban_fraction_500m']).copy()
    stations_df = stations_df[stations_df['altitude'] > 0].reset_index(drop=True)
    print(f"Stacje zakwalifikowane ({len(stations_df)}): {list(stations_df['station'].unique())}")

    # 3. Dopasowanie pomiarów z pliku parquet za pomocą KDTree (promień 100m)
    print("3. Łączenie pomiarów sensorów (KDTree)...")
    df_raw = pd.read_parquet(parquet_path)
    unique_pq = df_raw[['X', 'Y']].drop_duplicates().copy().reset_index(drop=True)

    tree = cKDTree(stations_df[['X', 'Y']].values)
    dists, indices = tree.query(unique_pq[['X', 'Y']].values)
    unique_pq['station_idx'] = indices
    unique_pq['dist_m'] = dists

    valid_pq = unique_pq[unique_pq['dist_m'] <= 100.0].copy()
    for col in ['station', 'lat', 'lon', 'altitude', 'urban_fraction_500m']:
        valid_pq[col] = stations_df.iloc[valid_pq['station_idx']][col].values

    df_lcs = pd.merge(df_raw, valid_pq.drop(columns=['station_idx']), on=['X', 'Y'], how='inner')

    # 4. Synchronizacja czasu do UTC (ze wspólnego modułu utils_time)
    print("4. Synchronizacja czasu do UTC...")
    dt_local = pd.to_datetime(df_lcs['date'].astype(str) + ' ' + df_lcs['hour'].astype(str) + ':00:00')
    df_lcs['timestamp'] = localize_to_utc(dt_local)
    df_lcs.dropna(subset=['timestamp'], inplace=True)
    df_lcs.rename(columns={'temperature_2m': 'temp_ground'}, inplace=True)

    # 5. Dołączenie temperatury z siatki ERA5-Land
    print("5. Próbkowanie czasoprzestrzenne ERA5-Land...")
    ds_era5 = xr.open_dataset(ERA5_NETCDF_PATH)
    time_coord = 'valid_time' if 'valid_time' in ds_era5.coords else 'time'

    era5_pts = []
    for _, r in stations_df[['station', 'lat', 'lon']].drop_duplicates().iterrows():
        pt = ds_era5['t2m'].sel(latitude=r['lat'], longitude=r['lon'], method='nearest').to_dataframe().reset_index()
        pt['temp_era5'] = pt['t2m'] - 273.15
        pt['timestamp'] = pd.to_datetime(pt[time_coord])
        pt['station'] = r['station']
        era5_pts.append(pt[['timestamp', 'station', 'temp_era5']])

    df_era5_all = pd.concat(era5_pts, ignore_index=True)
    df_merged = pd.merge(df_lcs, df_era5_all, on=['timestamp', 'station'], how='inner')

    # 6. Dodanie cech harmonicznych czasu ze wspólnego modułu
    print("6. Dodawanie cech cyklicznych...")
    df_final = add_cyclic_temporal_features(df_merged)

    final_cols = [
        'timestamp', 'station', 'lat', 'lon', 'altitude', 'urban_fraction_500m',
        'sin_hour', 'cos_hour', 'sin_doy', 'cos_doy',
        'temp_era5', 'temp_ground'
    ]
    df_final = df_final[final_cols].dropna().sort_values(by=['timestamp', 'station']).reset_index(drop=True)

    # Zapis
    out_parquet = PROCESSED_DATA_DIR / "lcs_dataset.parquet"
    out_csv = PROCESSED_DATA_DIR / "lcs_dataset.csv"
    df_final.to_parquet(out_parquet, index=False)
    df_final.to_csv(out_csv, index=False)

    print(f"✅ Zapisano zbiór LCS: {out_parquet} ({len(df_final):,} wierszy)")


if __name__ == '__main__':
    process_lcs_data()