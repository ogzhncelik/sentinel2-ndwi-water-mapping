# sentinel2-ndwi-water-mapping

An end-to-end Python pipeline that turns raw Sentinel-2 Level-2A scenes into clean water-body polygons.
It computes the Normalized Difference Water Index (NDWI), converts water pixels into vector polygons, smooths and filters them, and exports a shapefile with area information. It also creates an RGB GeoTIFF of each scene for visual checks.

## Pipeline

For every `.SAFE.zip` scene in the input folder:

1. **Organize** – scenes are sorted by acquisition date (newest first) and moved into numbered folders.
2. **Extract** – each archive is unzipped in its own folder.
3. **NDWI** – computed from the 10 m Green (B03) and NIR (B08) bands:
   `NDWI = (Green − NIR) / (Green + NIR)`
4. **Water mask** – pixels with NDWI > 0 are labeled as water.
5. **Sieve** – small isolated pixel groups (noise) are removed.
6. **Polygonize** – the water mask is converted to vector polygons.
7. **Clean-up** – holes are removed, edges are smoothed with Chaikin's algorithm and geometries are simplified.
8. **Area filter** – an `area` field (hectares) is added and polygons ≤ 10 ha are dropped.
9. **Reproject** – NDWI raster and final shapefile are written in the CRS of the input bands.
10. **RGB** – a contrast-stretched natural-color GeoTIFF (B04/B03/B02) is created.

## Outputs (per scene)

| File | Description |
|---|---|
| `<scene>_NDWI.tif` | NDWI raster (float32, nodata = −9999) |
| `<scene>_water_final.shp` | Smoothed water polygons with an `area` field in hectares |
| `<scene>.tif` | 8-bit RGB GeoTIFF |

## Installation

```bash
pip install -r requirements.txt
```

GDAL can be hard to install with pip on Windows. Using conda is recommended:

```bash
conda install -c conda-forge gdal rasterio
```

## Usage

Set `ZIP_SOURCE_DIR`, `OUTPUT_BASE_DIR` and `START_NUMBER` at the top of the script, then run:

```bash
python sentinel2_ndwi_water_polygons.py
```

Processing parameters (NDWI threshold, minimum object size, smoothing, area limit) can be adjusted in the script.

## License

MIT
