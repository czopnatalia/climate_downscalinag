import os
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import xarray as xr

os.makedirs("reports/figures", exist_ok=True)

# 1. Scaler
df_train = pd.read_parquet("data/processed/train_2022.parquet")
features = [
    "temp_era5",
    "altitude",
    "urban_fraction_500m",
    "sin_hour",
    "cos_hour",
    "sin_doy",
    "cos_doy",
]
scaler = StandardScaler()
scaler.fit(df_train[features].values)

# 2. DEM i Land Cover
granice = gpd.read_file("data/raw/krakow_granice.geojson")
with rasterio.open("data/raw/krakow_dem.tif") as src_dem:
  dem = src_dem.read(1).astype(float)
  nodata_dem = src_dem.nodata
  bounds = src_dem.bounds
  dem_transform = src_dem.transform
  dem_crs = src_dem.crs
  rows, cols = dem.shape

lc = np.empty((rows, cols), dtype=np.uint8)
with rasterio.open("data/raw/krakow_land_cover.tif") as src_lc:
  reproject(
      source=rasterio.band(src_lc, 1),
      destination=lc,
      src_transform=src_lc.transform,
      src_crs=src_lc.crs,
      dst_transform=dem_transform,
      dst_crs=dem_crs,
      resampling=Resampling.nearest,
  )

valid = (dem != nodata_dem) & (~np.isnan(dem)) & (dem > 150) & (dem < 600)

lons_1d = np.linspace(bounds.left, bounds.right, cols)
lats_1d = np.linspace(bounds.top, bounds.bottom, rows)
xs, ys = np.meshgrid(lons_1d, lats_1d)

lons = xs[valid]
lats = ys[valid]
altitudes = dem[valid]
urban_fractions = (lc[valid] == 1).astype(float)

# 3. ERA5
chosen_time = "2022-07-21 02:00:00"
ds_era5 = xr.open_dataset("data/raw/krakow_era5_2000_2025.nc")
time_col = "valid_time" if "valid_time" in ds_era5.coords else "time"
snap = ds_era5.sel({time_col: chosen_time})

era5_t2m = (
    snap["t2m"]
    .interp(
        latitude=xr.DataArray(lats),
        longitude=xr.DataArray(lons),
        method="cubic",
        kwargs={"fill_value": None},
    )
    .values
    - 273.15
)

hour, doy = 2, 202
sin_hour = np.full_like(era5_t2m, np.sin(2 * np.pi * hour / 24.0))
cos_hour = np.full_like(era5_t2m, np.cos(2 * np.pi * hour / 24.0))
sin_doy = np.full_like(era5_t2m, np.sin(2 * np.pi * doy / 365.25))
cos_doy = np.full_like(era5_t2m, np.cos(2 * np.pi * doy / 365.25))

# 4. Predykcja samej delty (wkładu sieci)
X_grid = np.column_stack(
    [era5_t2m, altitudes, urban_fractions, sin_hour, cos_hour, sin_doy, cos_doy]
)
X_grid_scaled = scaler.transform(X_grid)
model = tf.keras.models.load_model("models/keras_mlp_model.keras")
delta_pred = model.predict(X_grid_scaled, batch_size=8192).flatten()

grid_delta = np.full((rows, cols), np.nan)
grid_delta[valid] = delta_pred

# 5. Rysowanie samej poprawki mikroklimatycznej
fig, ax = plt.subplots(figsize=(8, 6))
extent_dem = [bounds.left, bounds.right, bounds.bottom, bounds.top]

# Symetryczna skala barw skupiona na 0
vlim = np.nanpercentile(np.abs(delta_pred), 98)
im = ax.imshow(
    grid_delta,
    extent=extent_dem,
    cmap="coolwarm",
    vmin=-vlim,
    vmax=vlim,
    origin="upper",
)

granice.boundary.plot(
    ax=ax, color="black", linewidth=1.5, linestyle="--", label="Granice Krakowa"
)

ax.set_xlim(19.75, 20.22)
ax.set_ylim(49.95, 50.15)
ax.set_aspect(1.0 / np.cos(np.radians(50.0)))

ax.set_title(
    f"Efekt downscalingu: Poprawka mikroklimatyczna ($\Delta T$)\n{chosen_time} UTC",
    fontsize=11,
    fontweight="bold",
)
ax.set_xlabel("Długość geograficzna [°E]")
ax.set_ylabel("Szerokość geograficzna [°N]")
ax.legend(loc="upper left")

cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Poprawka temperatury $\Delta T$ [°C]")

plt.tight_layout()
out_file = "reports/figures/3_delta_mikroklimat_krakow.png"
plt.savefig(out_file, dpi=300)
plt.close()

print(f"✅ Zapisano mapę samej poprawki mikroklimatycznej: {out_file}")