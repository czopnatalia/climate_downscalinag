from pathlib import Path

# Główne katalogi projektu
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Pliki danych wejściowych (raw)
ERA5_NETCDF_PATH = RAW_DATA_DIR / "krakow_era5_2000_2025.nc"
DEM_RASTER_PATH = RAW_DATA_DIR / "krakow_dem.tif"
LAND_COVER_PATH = RAW_DATA_DIR / "krakow_land_cover.tif"

# Pliki danych przetworzonych (processed)
DATASET_FILE = PROCESSED_DATA_DIR / "final_dataset.parquet"
DATASET_2022_PATH = PROCESSED_DATA_DIR / "final_dataset_2022.parquet"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train_2022.parquet"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test_2022.parquet"

# Cechy wejściowe i zmienna celu
FEATURE_COLUMNS = [
    'temp_era5',
    'altitude',
    'urban_fraction_500m',
    'sin_hour',
    'cos_hour',
    'sin_doy',
    'cos_doy'
]
TARGET_COLUMN = 'temp_ground'

# Bounding box aglomeracji Krakowa (WGS84)
KRAKOW_BBOX = {
    'north': 50.20,
    'south': 49.95,
    'west': 19.75,
    'east': 20.22
}

# Kolejność pór roku Gumińskiego
SEASONS_ORDER = ['zima', 'przedwiosnie', 'wiosna', 'lato', 'jesien', 'przedzimie']