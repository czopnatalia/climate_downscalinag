import os
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from rasterio.warp import reproject, Resampling
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import xarray as xr
import geopandas as gpd

# =========================================================
# KONFIGURACJA STRONY
# =========================================================
st.set_page_config(
    page_title="Downscaling Temperatury - Kraków",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ Przestrzenny Downscaling Temperatury dla Krakowa")
st.markdown("Aplikacja demonstracyjna: Rekonstrukcja pola temperatury (~100m) z reanalizy mezoskalowej ERA5-Land.")

# =========================================================
# WCZYTYWANIE ZASOBÓW Z CACHE
# =========================================================
@st.cache_resource
def load_model_and_scaler():
    df_train = pd.read_parquet("data/processed/train_2022.parquet")
    features = ['temp_era5', 'altitude', 'urban_fraction_500m', 'sin_hour', 'cos_hour', 'sin_doy', 'cos_doy']
    scaler = StandardScaler()
    scaler.fit(df_train[features].values)
    model = tf.keras.models.load_model("models/keras_mlp_model.keras")
    return model, scaler

@st.cache_data
def load_base_rasters():
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

    # Krok co 2 piksele dla natychmiastowej responsywności w przeglądarce
    step = 2
    dem_sub = dem[::step, ::step]
    lc_sub = lc[::step, ::step]
    rows_sub, cols_sub = dem_sub.shape

    valid = (dem_sub != nodata_dem) & (~np.isnan(dem_sub)) & (dem_sub > 150) & (dem_sub < 600)
    lons_1d = np.linspace(bounds.left, bounds.right, cols_sub)
    lats_1d = np.linspace(bounds.top, bounds.bottom, rows_sub)
    xs, ys = np.meshgrid(lons_1d, lats_1d)

    return bounds, granice, valid, xs[valid], ys[valid], dem_sub[valid], (lc_sub[valid] == 1).astype(float), rows_sub, cols_sub

@st.cache_resource
def load_era5_dataset():
    return xr.open_dataset("data/raw/krakow_era5_2000_2025.nc")

@st.cache_data
def load_all_station_data():
    # Łączymy dane treningowe i testowe, aby mieć pełne pokrycie 2022 roku
    parts = []
    if os.path.exists("data/processed/train_2022.parquet"):
        parts.append(pd.read_parquet("data/processed/train_2022.parquet"))
    if os.path.exists("data/processed/test_2022.parquet"):
        parts.append(pd.read_parquet("data/processed/test_2022.parquet"))
    
    if parts:
        df_all = pd.concat(parts, ignore_index=True)
    else:
        df_all = pd.DataFrame()

    # Automatyczna detekcja kolumny czasu
    time_cols = ['timestamp', 'time', 'date', 'datetime', 'valid_time']
    c_time = next((c for c in time_cols if c in df_all.columns), None)
    if c_time:
        # Usuwamy ewentualne strefy czasowe do czystego UTC (tz-naive)
        s_dt = pd.to_datetime(df_all[c_time])
        if s_dt.dt.tz is not None:
            s_dt = s_dt.dt.tz_convert('UTC').dt.tz_localize(None)
        df_all['clean_datetime'] = s_dt

    return df_all

model, scaler = load_model_and_scaler()
bounds, granice, valid_mask, lons, lats, altitudes, urban_fracs, r_sub, c_sub = load_base_rasters()
ds_era5 = load_era5_dataset()
df_stations = load_all_station_data()

# =========================================================
# PASEK BOCZNY
# =========================================================
st.sidebar.header("⚙️ Wybór sytuacji synoptycznej")

scenariusze = {
    "Letnia noc (Miejska Wyspa Ciepła)": "2022-07-21 02:00:00",
    "Letnie popołudnie (Szczyt insolacji)": "2022-07-21 14:00:00",
    "Zimowa noc inwersyjna": "2022-01-12 04:00:00",
    "Wiosenne południe": "2022-04-15 12:00:00"
}

tryb = st.sidebar.radio("Tryb wyboru daty:", ["Gotowe scenariusze", "Własna data (2000-2025)"])

if tryb == "Gotowe scenariusze":
    wybrany_opis = st.sidebar.selectbox("Scenariusz:", list(scenariusze.keys()))
    chosen_time = scenariusze[wybrany_opis]
else:
    wybrana_data = st.sidebar.date_input("Dzień:", value=pd.to_datetime("2022-07-21"))
    wybrana_godzina = st.sidebar.slider("Godzina (UTC):", min_value=0, max_value=23, value=2)
    chosen_time = f"{wybrana_data} {wybrana_godzina:02d}:00:00"

st.sidebar.markdown(f"**Wybrany termin symulacji:**\n`{chosen_time}`")

# =========================================================
# PRZELICZENIE MODELU
# =========================================================
time_col = 'valid_time' if 'valid_time' in ds_era5.coords else 'time'

try:
    snap = ds_era5.sel({time_col: chosen_time})
except Exception:
    st.error(f"Nie znaleziono terminu {chosen_time} w pliku ERA5.")
    st.stop()

# Płynne tło mezoskalowe z ekstrapolacją brzegów
era5_t2m = snap["t2m"].interp(
    latitude=xr.DataArray(lats),
    longitude=xr.DataArray(lons),
    method="cubic",
    kwargs={"fill_value": None}
).values - 273.15

dt = pd.to_datetime(chosen_time)
hour = dt.hour
doy = dt.dayofyear

sin_hour = np.full_like(era5_t2m, np.sin(2 * np.pi * hour / 24.0))
cos_hour = np.full_like(era5_t2m, np.cos(2 * np.pi * hour / 24.0))
sin_doy = np.full_like(era5_t2m, np.sin(2 * np.pi * doy / 365.25))
cos_doy = np.full_like(era5_t2m, np.cos(2 * np.pi * doy / 365.25))

X_grid = np.column_stack([era5_t2m, altitudes, urban_fracs, sin_hour, cos_hour, sin_doy, cos_doy])
X_scaled = scaler.transform(X_grid)
delta_pred = model.predict(X_scaled, batch_size=8192, verbose=0).flatten()
t_downscaled = era5_t2m + delta_pred

# Matryce rastrów
grid_downscaled = np.full((r_sub, c_sub), np.nan)
grid_downscaled[valid_mask] = t_downscaled

grid_diff = np.full((r_sub, c_sub), np.nan)
grid_diff[valid_mask] = delta_pred

# =========================================================
# WIZUALIZACJA GŁÓWNA (2 PANELE)
# =========================================================
col1, col2 = st.columns(2)
aspect_ratio = 1.0 / np.cos(np.radians(50.0))
extent_dem = [bounds.left, bounds.right, bounds.bottom, bounds.top]

with col1:
    st.subheader("Pole temperatury po downscalingu")
    fig1, ax1 = plt.subplots(figsize=(6, 4.5))
    im1 = ax1.imshow(grid_downscaled, extent=extent_dem, cmap="turbo", origin="upper")
    granice.boundary.plot(ax=ax1, color="black", linewidth=1.2, linestyle="--")
    ax1.set_xlim(19.75, 20.22)
    ax1.set_ylim(49.95, 50.15)
    ax1.set_aspect(aspect_ratio)
    ax1.set_xlabel("Długość [°E]")
    ax1.set_ylabel("Szerokość [°N]")
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("Temperatura [°C]")
    st.pyplot(fig1)

with col2:
    st.subheader("Poprawka mikroklimatyczna ($\Delta T$)")
    fig2, ax2 = plt.subplots(figsize=(6, 4.5))
    vlim = float(np.nanpercentile(np.abs(delta_pred), 98))
    im2 = ax2.imshow(grid_diff, extent=extent_dem, cmap="coolwarm", vmin=-vlim, vmax=vlim, origin="upper")
    granice.boundary.plot(ax=ax2, color="black", linewidth=1.2, linestyle="--")
    ax2.set_xlim(19.75, 20.22)
    ax2.set_ylim(49.95, 50.15)
    ax2.set_aspect(aspect_ratio)
    ax2.set_xlabel("Długość [°E]")
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label("$\Delta T$ [°C]")
    st.pyplot(fig2)

# =========================================================
# INSPEKTOR PUNKTOWY (STACJE POMIAROWE)
# =========================================================
st.markdown("---")
st.subheader("📍 Walidacja ze stacją pomiarową")

potential_station_cols = ['station_name', 'station_id', 'station', 'sensor_id', 'id', 'nazwa_stacji']
col_stacja = next((c for c in potential_station_cols if c in df_stations.columns), None)

if col_stacja is None or 'clean_datetime' not in df_stations.columns:
    st.info("Brak tabeli stacji pomiarowych lub nierozpoznany format kolumn.")
else:
    stacje = sorted(df_stations[col_stacja].dropna().unique().tolist())
    wybrana_stacja = st.selectbox("Wybierz stację pomiarową:", stacje)

    target_dt = pd.to_datetime(chosen_time)
    sub_st = df_stations[(df_stations[col_stacja] == wybrana_stacja) & (df_stations['clean_datetime'] == target_dt)]

    if not sub_st.empty:
        wiersz = sub_st.iloc[0]
        col_ground = next((c for c in ['temp_ground', 'target', 'temp', 'temperature'] if c in df_stations.columns), None)
        
        t_ground = float(wiersz[col_ground]) if col_ground else np.nan
        t_era = float(wiersz['temp_era5'])

        feat_vals = wiersz[['temp_era5', 'altitude', 'urban_fraction_500m', 'sin_hour', 'cos_hour', 'sin_doy', 'cos_doy']].values.reshape(1, -1)
        x_pt = scaler.transform(feat_vals)
        t_model_pt = t_era + float(model.predict(x_pt, verbose=0)[0][0])

        err_era = abs(t_era - t_ground) if not np.isnan(t_ground) else np.nan
        err_model = abs(t_model_pt - t_ground) if not np.isnan(t_ground) else np.nan
        zysk = err_era - err_model if not np.isnan(t_ground) else np.nan

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Surowe ERA5", f"{t_era:.2f} °C")
        m2.metric("Nasz Model (Downscaling)", f"{t_model_pt:.2f} °C", delta=f"{t_model_pt - t_era:+.2f} °C")
        m3.metric("Rzeczywisty pomiar stacji", f"{t_ground:.2f} °C" if not np.isnan(t_ground) else "Brak danych")
        if not np.isnan(err_model):
            m4.metric("Błąd modelu (|Błąd|)", f"{err_model:.2f} °C", delta=f"{zysk:+.2f} °C poprawy" if zysk >= 0 else f"{zysk:+.2f} °C gorszy")
        else:
            m4.metric("Błąd modelu", "N/A")
    else:
        st.info(f"Dla stacji **{wybrana_stacja}** brak bezpośredniego pomiaru w terminie `{chosen_time}` (pomiary stacyjne są dostępne dla roku 2022). Wybierz inny termin lub stację.")