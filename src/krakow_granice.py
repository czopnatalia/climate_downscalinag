import json
import os
import geopandas as gpd
import requests

url = "https://nominatim.openstreetmap.org/search"
params = {
    "q": "Kraków, Poland",
    "format": "geojson",
    "polygon_geojson": 1,
}

# Nominatim wymaga nagłówka User-Agent identyfikującego skrypt
headers = {"User-Agent": "krakow_downscaling_student_project"}

print("Pobieranie granic Krakowa...")
response = requests.get(url, params=params, headers=headers)
response.raise_for_status()

# Przekształcenie odpowiedzi do formatu GeoDataFrame
data = response.json()
gdf = gpd.GeoDataFrame.from_features(data["features"], crs="EPSG:4326")

# Wybór obrysu miasta (admin_level=8 lub addresstype='city')
krakow = gdf[gdf["addresstype"] == "city"]

# Zabezpieczenie: jeśli filtr zwróci pustą ramkę, bierzemy pierwszy rekord z poligonem
if krakow.empty:
    krakow = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].head(
        1
    )

output_path = "data/raw/krakow_granice.geojson"
krakow.to_file(output_path, driver="GeoJSON")

print(f"✅ Sukces! Granice zapisane w: {output_path}")