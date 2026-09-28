import numpy as np
import rasterio
from rasterio.mask import mask
from shapely.geometry import Point, Polygon, mapping
from pyproj import Transformer


def sample_dem(raster_path, lats, lons):
    """Próbkuje wysokość n.p.m. z rastra DEM dla zadanych współrzędnych WGS84."""
    with rasterio.open(raster_path) as src:
        crs = src.crs
        nodata = src.nodata
        if crs and crs.to_string() != "EPSG:4326":
            tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
            xs, ys = tr.transform(lons, lats)
        else:
            xs, ys = lons, lats
            
        coords = list(zip(xs, ys))
        sampled = [val[0] for val in src.sample(coords)]
        return np.array([np.nan if (v == nodata or v <= 0) else float(v) for v in sampled])


def compute_urban_fraction(raster_path, lats, lons, radius_m=500.0):
    """Oblicza wskaźnik powierzchni zurbanizowanej w buforze wokół punktu."""
    with rasterio.open(raster_path) as src:
        crs = src.crs
        nodata = src.nodata

        sample_win = src.read(1, window=((0, 100), (0, 100)))
        urban_classes = list(range(100, 143)) if np.nanmax(sample_win) > 60 else [50]

        to_metric = Transformer.from_crs("EPSG:4326", "EPSG:32634", always_xy=True)
        to_raster = Transformer.from_crs("EPSG:32634", crs, always_xy=True)

        fractions = []
        for lat, lon in zip(lats, lons):
            xm, ym = to_metric.transform(lon, lat)
            circle = Point(xm, ym).buffer(radius_m)
            poly_coords = [to_raster.transform(x, y) for x, y in circle.exterior.coords]
            geom = Polygon(poly_coords)
            try:
                out_img, _ = mask(src, [mapping(geom)], crop=True)
                px = out_img[0].flatten()
                valid = px[px != nodata] if nodata is not None else px[px > 0]
                if len(valid) == 0:
                    fractions.append(np.nan)
                else:
                    fractions.append(round(float(np.isin(valid, urban_classes).sum() / len(valid)), 4))
            except Exception:
                fractions.append(np.nan)
        return np.array(fractions)