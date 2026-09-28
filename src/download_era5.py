import os
import glob
from pathlib import Path
import cdsapi
import xarray as xr
from config import RAW_DATA_DIR, ERA5_NETCDF_PATH, KRAKOW_BBOX


def download_and_merge_era5(start_year=2000, end_year=2025):
    era5_dir = RAW_DATA_DIR / "era5"
    era5_dir.mkdir(parents=True, exist_ok=True)

    # client = cdsapi.Client(url="https://cds.climate.copernicus.eu/api",key="f85a23bc-7b23-4cea-92fa-ec62fa5810fc")
    client = cdsapi.Client()
    area = [
        KRAKOW_BBOX['north'], KRAKOW_BBOX['west'],
        KRAKOW_BBOX['south'], KRAKOW_BBOX['east']
    ]

    print("--- 1. Sprawdzanie i pobieranie paczek miesięcznych ERA5-Land ---")
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            filename = era5_dir / f"krakow_temp_{year}_{month:02d}.nc"
            if filename.exists():
                continue

            print(f"Pobieranie: {year}-{month:02d}...")
            request = {
                "variable": ["2m_temperature"],
                "year": str(year),
                "month": f"{month:02d}",
                "day": [f"{d:02d}" for d in range(1, 32)],
                "time": [f"{h:02d}:00" for h in range(24)],
                "data_format": "netcdf",
                "download_format": "unarchived",
                "area": area,
            }
            try:
                client.retrieve("reanalysis-era5-land", request).download(str(filename))
            except Exception as e:
                print(f"Pominięto {year}-{month:02d}: {e}")
                if year == end_year:
                    break

    print("\n--- 2. Scalanie plików NetCDF do jednego zbioru wieloletniego ---")
    nc_files = sorted(glob.glob(str(era5_dir / "krakow_temp_*.nc")))
    if not nc_files:
        print("Brak plików do scalenia!")
        return

    # Otwieramy pliki jako zbiór wieloczęściowy
    with xr.open_mfdataset(nc_files, combine='by_coords') as ds:
        ds.to_netcdf(ERA5_NETCDF_PATH)
        print(f"✅ Pomyślnie utworzono: {ERA5_NETCDF_PATH}")


if __name__ == '__main__':
    download_and_merge_era5()