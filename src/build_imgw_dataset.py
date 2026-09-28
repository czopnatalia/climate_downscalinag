import pandas as pd
import xarray as xr
from config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR,
    DEM_RASTER_PATH, LAND_COVER_PATH, ERA5_NETCDF_PATH
)
from utils_geo import sample_dem, compute_urban_fraction
from utils_time import localize_to_utc, add_cyclic_temporal_features


def process_imgw_data():
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    metadata = pd.DataFrame([
        {
            'station': 'Krakow-Balice',
            'lat': 50.0777,
            'lon': 19.7848,
            'file_path': RAW_DATA_DIR / "krakow_balice_imgw_2000_2025.csv"
        },
        {
            'station': 'Krakow-Obserwatorium',
            'lat': 50.0614,
            'lon': 19.9594,
            'file_path': RAW_DATA_DIR / "krakow_obserwatorium_imgw_2000_2025.csv"
        }
    ])

    print("1. Pobieranie wysokości i urban fraction dla stacji IMGW...")
    metadata['altitude'] = sample_dem(DEM_RASTER_PATH, metadata['lat'].values, metadata['lon'].values)
    metadata['urban_fraction_500m'] = compute_urban_fraction(LAND_COVER_PATH, metadata['lat'].values, metadata['lon'].values)

    print("2. Wczytywanie pomiarów naziemnych i konwersja do UTC...")
    station_dfs = []
    for _, meta in metadata.iterrows():
        df_raw = pd.read_csv(meta['file_path'])
        
        # Budowa daty i konwersja do UTC za pomocą współdzielonej funkcji
        dt_local = pd.to_datetime(
            df_raw['year'].astype(str) + '-' +
            df_raw['month'].astype(str).str.zfill(2) + '-' +
            df_raw['day'].astype(str).str.zfill(2) + ' ' +
            df_raw['hour'].astype(str).str.zfill(2) + ':00:00'
        )
        df_raw['timestamp'] = localize_to_utc(dt_local)
        df_raw['station'] = meta['station']
        df_raw['lat'] = meta['lat']
        df_raw['lon'] = meta['lon']
        df_raw['altitude'] = meta['altitude']
        df_raw['urban_fraction_500m'] = meta['urban_fraction_500m']
        df_raw.rename(columns={'temp': 'temp_ground'}, inplace=True)
        
        clean = df_raw[['timestamp', 'station', 'lat', 'lon', 'altitude', 'urban_fraction_500m', 'temp_ground']].dropna()
        station_dfs.append(clean)

    df_imgw = pd.concat(station_dfs, ignore_index=True)

    print("3. Ekstrakcja temperatur ERA5-Land (NetCDF)...")
    ds = xr.open_dataset(ERA5_NETCDF_PATH)
    time_coord = 'valid_time' if 'valid_time' in ds.coords else 'time'

    era_series = []
    for _, r in metadata.iterrows():
        pt = ds['t2m'].sel(latitude=r['lat'], longitude=r['lon'], method='nearest').to_dataframe().reset_index()
        pt['temp_era5'] = pt['t2m'] - 273.15
        pt['timestamp'] = pd.to_datetime(pt[time_coord])
        pt['station'] = r['station']
        era_series.append(pt[['timestamp', 'station', 'temp_era5']])

    df_era = pd.concat(era_series, ignore_index=True)
    df_merged = pd.merge(df_imgw, df_era, on=['timestamp', 'station'], how='inner')

    print("4. Dodawanie cech cyklicznych...")
    df_final = add_cyclic_temporal_features(df_merged)

    target_cols = [
        'timestamp', 'station', 'lat', 'lon', 'altitude', 'urban_fraction_500m',
        'sin_hour', 'cos_hour', 'sin_doy', 'cos_doy',
        'temp_era5', 'temp_ground'
    ]
    df_final = df_final[target_cols].dropna().sort_values(['timestamp', 'station']).reset_index(drop=True)

    out_parquet = PROCESSED_DATA_DIR / "imgw_dataset.parquet"
    out_csv = PROCESSED_DATA_DIR / "imgw_dataset.csv"
    df_final.to_parquet(out_parquet, index=False)
    df_final.to_csv(out_csv, index=False)
    print(f"✅ Zapisano gotowy zbiór IMGW: {out_parquet} ({len(df_final):,} wierszy)")


if __name__ == '__main__':
    process_imgw_data()