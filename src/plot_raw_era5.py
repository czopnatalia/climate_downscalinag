import os
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import geopandas as gpd

os.makedirs("reports/figures", exist_ok=True)

# 1. Wczytanie ERA5 i granic Krakowa
print("1. Wczytywanie danych...")
ds = xr.open_dataset("data/raw/krakow_era5_2000_2025.nc")
granice = gpd.read_file("data/raw/krakow_granice.geojson")

time_col = "valid_time" if "valid_time" in ds.coords else "time"
snap = ds.sel({time_col: "2022-07-21 02:00:00"})

temp_c = snap["t2m"].values - 273.15
lats = snap["latitude"].values
lons = snap["longitude"].values

# 2. Ustalenie zakresu (dopasowanego do obszaru aglomeracji Krakowa)
extent = [lons.min() - 0.05, lons.max() + 0.05, lats.min() - 0.05, lats.max() + 0.05]

fig, ax = plt.subplots(figsize=(8, 6))

# Rysowanie oczek ERA5
im = ax.imshow(temp_c, cmap="turbo", extent=extent, origin="upper")

# Wypisanie temperatur w każdym oczku
for r, lat in enumerate(lats):
    for c, lon in enumerate(lons):
        wartosc = temp_c[r, c]
        ax.text(
            lon, lat, f"{wartosc:.1f} °C",
            color="white", ha="center", va="center",
            fontweight="bold", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.45)
        )

# 3. Naniesienie granic administracyjnych Krakowa
granice.boundary.plot(ax=ax, color="black", linewidth=2.0, linestyle="--", label="Granice Krakowa")

# 4. Formatowanie osi i proporcji geograficznej
ax.set_xlim(19.75, 20.22)
ax.set_ylim(49.95, 50.15)
ax.set_aspect(1.0 / np.cos(np.radians(50.0)))

ax.set_title("Surowe dane ERA5-Land (~9 km)\n2022-07-21 02:00:00 UTC", fontsize=12, fontweight="bold")
ax.set_xlabel("Długość geograficzna [°E]")
ax.set_ylabel("Szerokość geograficzna [°N]")
ax.legend(loc="upper left")

cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Temperatura powietrza [°C]")

plt.tight_layout()
out_file = "reports/figures/1_raw_era5_krakow.png"
plt.savefig(out_file, dpi=300)
plt.close()

print(f"✅ Zapisano mapę surowego ERA5: {out_file}")