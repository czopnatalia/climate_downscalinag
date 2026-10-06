import pandas as pd
import xarray as xr
from datetime import datetime
from config import ERA5_NETCDF_PATH, KRAKOW_BBOX

def get_era5_temperature(target_time: datetime) -> float:
    """
    Zwraca uśrednioną temperaturę tła z ERA5-Land (°C) dla Krakowa w danym terminie.
    """
    target_str = target_time.strftime("%Y-%m-%d %H:00:00")
    
    # 1. Próba odczytu z lokalnego pliku NetCDF
    if ERA5_NETCDF_PATH.exists():
        try:
            with xr.open_dataset(ERA5_NETCDF_PATH) as ds:
                # Sprawdzenie obecności daty w pliku
                time_slice = ds.sel(time=target_str, method='nearest')
                
                # Zabezpieczenie przed zbyt dużą różnicą czasu
                file_time = pd.to_datetime(time_slice.time.values)
                if abs((file_time - target_time).total_seconds()) <= 3600:
                    t_kelvin = time_slice['t2m'].mean().item()
                    return t_kelvin - 273.15
        except Exception:
            pass

    # 2. Fallback: Jeśli daty brak w pliku lokalnym, użyjemy zapytania API
    print(f"Pobieranie terminu {target_str} przez Copernicus CDS API...")
    import cdsapi
    import tempfile
    
    c = cdsapi.Client()
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        c.retrieve(
            'reanalysis-era5-land',
            {
                'variable': '2m_temperature',
                'year': str(target_time.year),
                'month': f"{target_time.month:02d}",
                'day': f"{target_time.day:02d}",
                'time': f"{target_time.hour:02d}:00",
                'area': [
                    KRAKOW_BBOX['north'], KRAKOW_BBOX['west'], 
                    KRAKOW_BBOX['south'], KRAKOW_BBOX['east']
                ],
                'format': 'netcdf',
            },
            tmp.name
        )
        with xr.open_dataset(tmp.name) as ds:
            t_kelvin = ds['t2m'].mean().item()
            return t_kelvin - 273.15