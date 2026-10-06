import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import xarray as xr
import geopandas as gpd

os.makedirs("reports/figures", exist_ok=True)

# 1. Odtworzenie skalera
print("1. Odtwarzanie skalera...")
df_train = pd.read_parquet("data/processed/train_2022.parquet")
features = ['temp_era5', 'altitude', 'urban_fraction_500m', 'sin_hour', 'cos_hour', 'sin_doy', 'cos_doy']
scaler = StandardScaler()
scaler.fit(df_train[features].values)

# 2. Wczytanie rastrów i granic
print("2. Wczytywanie rastrów i granic Krakowa...")
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
        resampling=Resampling.nearest
    )

valid = (dem != nodata_dem) & (~np.isnan(dem)) & (dem > 150) & (dem < 600)

lons_1d = np.linspace(bounds.left, bounds.right, cols)
lats_1d = np.linspace(bounds.top, bounds.bottom, rows)
xs, ys = np.meshgrid(lons_1d, lats_1d)

lons = xs[valid]
lats = ys[valid]
altitudes = dem[valid]
urban_fractions = (lc[valid] == 1).astype(float)

# 3. Płynne tło mezoskalowe ERA5
chosen_time = "2022-07-21 02:00:00"
print(f"3. Pobieranie i wygładzanie tła ERA5 ({chosen_time})...")
ds_era5 = xr.open_dataset("data/raw/krakow_era5_2000_2025.nc")
time_col = 'valid_time' if 'valid_time' in ds_era5.coords else 'time'
snap = ds_era5.sel({time_col: chosen_time})

era5_t2m = snap["t2m"].interp(
    latitude=xr.DataArray(lats),
    longitude=xr.DataArray(lons),
    method="cubic",
    kwargs={"fill_value": None}
).values - 273.15

hour, doy = 2, 202
sin_hour = np.full_like(era5_t2m, np.sin(2 * np.pi * hour / 24.0))
cos_hour = np.full_like(era5_t2m, np.cos(2 * np.pi * hour / 24.0))
sin_doy = np.full_like(era5_t2m, np.sin(2 * np.pi * doy / 365.25))
cos_doy = np.full_like(era5_t2m, np.cos(2 * np.pi * doy / 365.25))

# 4. Predykcja modelem (downscaling)
print("4. Predykcja temperatur w wysokiej rozdzielczości...")
X_grid = np.column_stack([
    era5_t2m,
    altitudes,
    urban_fractions,
    sin_hour,
    cos_hour,
    sin_doy,
    cos_doy
])

X_grid_scaled = scaler.transform(X_grid)
model = tf.keras.models.load_model("models/keras_mlp_model.keras")
delta_pred = model.predict(X_grid_scaled, batch_size=8192).flatten()
t_downscaled = era5_t2m + delta_pred

grid_downscaled = np.full((rows, cols), np.nan)
grid_downscaled[valid] = t_downscaled

# 5. Rysowanie mapy wynikowej
print("5. Generowanie mapy po downscalingu...")
fig, ax = plt.subplots(figsize=(8, 6))
extent_dem = [bounds.left, bounds.right, bounds.bottom, bounds.top]

im = ax.imshow(
    grid_downscaled,
    extent=extent_dem,
    cmap="turbo",
    origin="upper"
)

# Naniesienie granic administracyjnych Krakowa
granice.boundary.plot(ax=ax, color="black", linewidth=2.0, linestyle="--", label="Granice Krakowa")

# Identyczne limity osi i proporcje jak na wykresie 1
ax.set_xlim(19.75, 20.22)
ax.set_ylim(49.95, 50.15)
ax.set_aspect(1.0 / np.cos(np.radians(50.0)))

ax.set_title(f"Model po downscalingu (~100 m)\n{chosen_time} UTC", fontsize=12, fontweight="bold")
ax.set_xlabel("Długość geograficzna [°E]")
ax.set_ylabel("Szerokość geograficzna [°N]")
ax.legend(loc="upper left")

cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Temperatura powietrza [°C]")

plt.tight_layout()
out_file = "reports/figures/2_downscaled_krakow.png"
plt.savefig(out_file, dpi=300)
plt.close()

print(f"✅ Zapisano mapę po downscalingu: {out_file}")