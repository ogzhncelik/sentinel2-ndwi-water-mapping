import os, pyproj
proj_dir = pyproj.datadir.get_data_dir()
os.environ["PROJ_LIB"] = proj_dir
os.environ["PROJ_DATA"] = proj_dir
import rasterio
from osgeo import osr
from rasterio.enums import Resampling, ColorInterp
import glob
from osgeo import gdal
from osgeo import ogr
from rasterio.warp import calculate_default_transform, reproject
import numpy as np
import shutil
import re
import zipfile
import traceback
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
gdal.PushErrorHandler("CPLQuietErrorHandler")
# =====================================================================
# SETTINGS
# =====================================================================
ZIP_SOURCE_DIR  = r"path\to\input"    # folder containing the .SAFE.zip files
OUTPUT_BASE_DIR = r"path\to\output"   # parent folder where the numbered folders will be created
START_NUMBER    = 800                   # starting folder number


# =====================================================================
# STEP 1: MOVE ZIP FILES INTO NUMBERED FOLDERS IN DATE ORDER
# =====================================================================

def organize_sentinel_zips(source_dir, output_base, start_num):
    """
    Finds the S2*_MSIL2A_*.SAFE.zip files in source_dir,
    sorts them by date from NEWEST -> OLDEST,
    creates numbered folders starting from start_num and
    moves each zip into its corresponding folder.
    """
    zip_files = glob.glob(os.path.join(source_dir, "S2*_MSIL2A_*.SAFE.zip"))

    if not zip_files:
        print("[Organize] WARNING: No S2*_MSIL2A_*.SAFE.zip files found in the given folder!")
        print(f"           Searched folder: {source_dir}")
        return []

    print(f"[Organize] {len(zip_files)} .SAFE.zip files found.")

    # Extract the date from each zip
    zip_date_pairs = []
    for zf in zip_files:
        basename = os.path.basename(zf)
        match = re.search(r"_MSIL2A_(\d{8}T\d{6})_", basename)
        if match:
            date_str = match.group(1)
            dt = datetime.strptime(date_str, "%Y%m%dT%H%M%S")
            zip_date_pairs.append((zf, dt, basename))
        else:
            print(f"  [Organize] WARNING: Could not parse the date, skipping -> {basename}")

    if not zip_date_pairs:
        print("[Organize] ERROR: No date could be extracted from any file!")
        return []

    # Sort from the newest date to the oldest date (descending)
    zip_date_pairs.sort(key=lambda x: x[1], reverse=True)

    print("\n[Organize] In date order (newest -> oldest):")
    for i, (zf, dt, bn) in enumerate(zip_date_pairs):
        folder_num = start_num + i
        print(f"  {folder_num} <- {bn}  (Date: {dt.strftime('%Y-%m-%d %H:%M:%S')})")

    # Create the folders and move the zips
    created_folders = []
    for i, (zf, dt, bn) in enumerate(zip_date_pairs):
        folder_num = start_num + i
        folder_path = os.path.join(output_base, str(folder_num))
        os.makedirs(folder_path, exist_ok=True)

        dest_path = os.path.join(folder_path, bn)
        if os.path.abspath(zf) != os.path.abspath(dest_path):
            shutil.move(zf, dest_path)
            print(f"  [Moved] {bn} -> {folder_path}")
        else:
            print(f"  [Already in place] {bn} -> {folder_path}")

        created_folders.append(folder_path)

    print(f"\n[Organize] Done! {len(created_folders)} folders created: "
          f"{start_num} - {start_num + len(created_folders) - 1}")

    return created_folders


# =====================================================================
# STEP 2: EXTRACT THE ZIP IN EACH FOLDER
# =====================================================================

def extract_zips_in_folder(folder_path):
    """
    Finds the .SAFE.zip file in the folder and extracts it into the same folder.
    Structure inside the zip: *.SAFE/GRANULE/L2A_.../IMG_DATA/R10m/...jp2
    """
    zip_files = glob.glob(os.path.join(folder_path, "*.SAFE.zip"))
    if not zip_files:
        print(f"  [Extract] No .SAFE.zip found in {folder_path}, skipping.")
        return False

    for zf in zip_files:
        print(f"  [Extract] Extracting: {os.path.basename(zf)} -> {folder_path}")
        try:
            with zipfile.ZipFile(zf, 'r') as z:
                z.extractall(folder_path)
            print(f"  [Extract] Extracted successfully.")
        except Exception as e:
            print(f"  [Extract] ERROR: {e}")
            return False

    return True


# =====================================================================
# STEP 3: RUN ALL PROCESSES FOR A SINGLE FOLDER (NDWI + SHP + RGB)
# =====================================================================

def process_single_folder(INPUT_DIR):
    """
    Finds the Sentinel-2 bands in the given folder and:
      1) Calculates NDWI
      2) Water mask -> polygon -> shapefile
      3) Area calculation + filtering
      4) CRS reprojection
      5) Creates an RGB GeoTIFF
    """
    print(f"\n{'='*70}")
    print(f"  PROCESSING: {INPUT_DIR}")
    print(f"{'='*70}")

    OUTPUT_NAME = os.path.basename(INPUT_DIR) + "_NDWI.tif"
    OUTPUT_PATH = os.path.join(INPUT_DIR, OUTPUT_NAME)

    # --- Create a Virtual Raster (B1..B7) ---
    def build_virtual_raster(EXTRACT_DIR, INPUT_DIR):
        def find_band(b):
            hits = glob.glob(os.path.join(EXTRACT_DIR, "**", f"*B{b}.TIF"), recursive=True)
            return hits[0] if hits else None

        b1, b2, b3, b4 = find_band(1), find_band(2), find_band(3), find_band(4)
        b5, b6, b7 = find_band(5), find_band(6), find_band(7)

        if not (b4 and b3 and b2):
            print("ERROR: B4/B3/B2 not found; RGB order could not be set.")
            return

        ordered_files = [b4, b3, b2] + [p for p in (b1, b5, b6, b7) if p]

        base = os.path.basename(INPUT_DIR)
        vrt_path = os.path.join(INPUT_DIR, f"{base}.vrt")
        out_tif  = os.path.join(INPUT_DIR, f"{base}.tif")

        vrt = gdal.BuildVRT(vrt_path, ordered_files, separate=True)
        if vrt is None:
            raise RuntimeError("VRT could not be created.")
        vrt = None

        opts = gdal.TranslateOptions(creationOptions=["COMPRESS=LZW", "PHOTOMETRIC=RGB"])
        gdal.Translate(destName=out_tif, srcDS=vrt_path, options=opts)

        ds = gdal.Open(out_tif, gdal.GA_Update)
        if ds:
            if ds.RasterCount >= 1:
                b = ds.GetRasterBand(1)
                b.SetColorInterpretation(gdal.GCI_RedBand)
                b.SetDescription("B4")
            if ds.RasterCount >= 2:
                b = ds.GetRasterBand(2)
                b.SetColorInterpretation(gdal.GCI_GreenBand)
                b.SetDescription("B3")
            if ds.RasterCount >= 3:
                b = ds.GetRasterBand(3)
                b.SetColorInterpretation(gdal.GCI_BlueBand)
                b.SetDescription("B2")
            names = ["B1", "B5", "B6", "B7"]
            for idx, name in enumerate(names, start=4):
                if ds.RasterCount >= idx:
                    ds.GetRasterBand(idx).SetDescription(name)
            ds = None

        try:
            os.remove(vrt_path)
        except OSError:
            pass

        print("Multi-band GeoTIFF written in RGB order (B4,B3,B2):", out_tif)

    # --- Find the B03 (Green) and B08 (NIR) bands for Sentinel-2 ---
    green_candidates = glob.glob(os.path.join(INPUT_DIR, "**", "*B03*10m.jp2"), recursive=True)
    nir_candidates   = glob.glob(os.path.join(INPUT_DIR, "**", "*B08*10m.jp2"), recursive=True)

    if not green_candidates:
        print(f"[SKIP] {INPUT_DIR} -> *B03*10m.jp2 (Green) band not found.")
        return
    if not nir_candidates:
        print(f"[SKIP] {INPUT_DIR} -> *B08*10m.jp2 (NIR) band not found.")
        return

    green_path = green_candidates[0]
    nir_path   = nir_candidates[0]

    # CRS of the input bands
    with rasterio.open(green_path) as _src_ref:
        target_crs = _src_ref.crs

    epsg_code = None
    target_srs_param = None

    if target_crs:
        try:
            epsg_code = target_crs.to_epsg()
        except Exception:
            epsg_code = None

    if epsg_code is None:
        try:
            wkt = gdal.Open(green_path).GetProjection()
            if wkt:
                srs = osr.SpatialReference()
                if srs.ImportFromWkt(wkt) == 0:
                    auth = srs.GetAuthorityCode(None)
                    if auth:
                        epsg_code = int(auth)
                        target_crs = None
                        target_srs_param = f"EPSG:{epsg_code}"
        except Exception:
            pass

    if epsg_code is None:
        m = re.search(r"T(\d{2})([C-X])", os.path.basename(green_path))
        if m:
            zone = int(m.group(1))
            lat_band = m.group(2)
            north = lat_band >= "N"
            epsg_code = (32600 if north else 32700) + zone
            target_crs = rasterio.crs.CRS.from_epsg(epsg_code)

    if epsg_code is not None:
        target_srs_param = f"EPSG:{epsg_code}"
    elif target_crs:
        target_srs_param = target_crs.to_wkt()
    else:
        target_srs_param = None

    print("Green (B03, 10m):", green_path)
    print("NIR   (B08, 10m):", nir_path)

    # --- Calculate NDWI ---
    with rasterio.open(green_path) as src_g, rasterio.open(nir_path) as src_n:
        if (src_g.width, src_g.height) != (src_n.width, src_n.height) or src_g.transform != src_n.transform:
            data_n = src_n.read(
                1,
                out_shape=(src_g.height, src_g.width),
                resampling=Resampling.bilinear,
            )
            transform_to_use = src_g.transform
            crs_to_use = src_g.crs
            meta_src = src_g
        else:
            data_n = src_n.read(1)
            transform_to_use = src_g.transform
            crs_to_use = src_g.crs
            meta_src = src_g

        data_g = src_g.read(1)

        nodata_g = src_g.nodata
        nodata_n = src_n.nodata

        mask = np.zeros_like(data_g, dtype=bool)
        if nodata_g is not None:
            mask |= (data_g == nodata_g)
        if nodata_n is not None:
            mask |= (data_n == nodata_n)

        g = data_g.astype("float32")
        n = data_n.astype("float32")

        denom = (g + n)
        zero_denom = denom == 0
        ndwi = np.full(g.shape, np.nan, dtype="float32")
        valid = ~mask & ~zero_denom
        ndwi[valid] = (g[valid] - n[valid]) / denom[valid]

        out_nodata = -9999.0
        ndwi_filled = ndwi.copy()
        ndwi_filled[~np.isfinite(ndwi_filled)] = out_nodata

        profile = meta_src.profile.copy()
        profile.pop("driver", None)
        profile.update(
            driver="GTiff",
            dtype="float32",
            count=1,
            compress="LZW",
            nodata=out_nodata,
            transform=transform_to_use,
            crs=crs_to_use,
        )

        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        with rasterio.open(OUTPUT_PATH, "w", **profile) as dst:
            dst.write(ndwi_filled, 1)

    print("NDWI output saved:", OUTPUT_PATH)

    # === NDWI -> Water mask -> Polygonize -> DeleteHoles -> Smooth -> FINAL SHP ===
    def water_polygonize_from_ndwi(
                                    *,
                                    OUTPUT_PATH,
                                    INPUT_DIR,
                                    threshold_ndwi,
                                    MIN_OBJ_PIX,
                                    CONNECTEDNESS,
                                    WATER_VALUE,
                                    MASK_NODATA,
                                    DELETE_HOLES_MAX_AREA,
                                    SMOOTH_ITER,
                                    SMOOTH_OFFSET,
                                    SIMPLIFY_PX,
                                    KEEP_INTERMEDIATES
                                ):
        def _remove_if_exists_shp(path_wo_ext):
            for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
                p = path_wo_ext + ext
                if os.path.exists(p):
                    try: os.remove(p)
                    except Exception: pass

        def _ring_to_coords(ring):
            pts = []
            for i in range(ring.GetPointCount()):
                x, y, _ = ring.GetPoint(i)
                pts.append((x, y))
            return pts

        def _coords_to_ring(coords):
            if coords[0] != coords[-1]:
                coords = coords + [coords[0]]
            ring = ogr.Geometry(ogr.wkbLinearRing)
            for x, y in coords:
                ring.AddPoint(x, y)
            return ring

        def _chaikin_once(coords, offset):
            if coords[0] != coords[-1]:
                coords = coords + [coords[0]]
            newc = []
            for i in range(len(coords) - 1):
                x0, y0 = coords[i]
                x1, y1 = coords[i+1]
                Q = ( (1 - offset) * x0 + offset * x1, (1 - offset) * y0 + offset * y1 )
                R = ( offset * x0 + (1 - offset) * x1, offset * y0 + (1 - offset) * y1 )
                newc.extend([Q, R])
            if newc[0] != newc[-1]:
                newc.append(newc[0])
            return newc

        def _chaikin(coords, iterations, offset):
            out = coords
            for _ in range(max(0, iterations)):
                out = _chaikin_once(out, offset)
            return out

        def _ring_area(ring):
            poly = ogr.Geometry(ogr.wkbPolygon)
            poly.AddGeometry(ring.Clone())
            a = poly.GetArea()
            poly = None
            return a

        def _delete_holes_from_polygon(poly, max_area):
            ext = poly.GetGeometryRef(0)
            ext_coords = _ring_to_coords(ext)
            ext_ring = _coords_to_ring(ext_coords)

            new_poly = ogr.Geometry(ogr.wkbPolygon)
            new_poly.AddGeometry(ext_ring)

            for i in range(1, poly.GetGeometryCount()):
                hole = poly.GetGeometryRef(i)
                if hole is None:
                    continue
                if max_area <= 0.0:
                    continue
                area = _ring_area(hole)
                if area > max_area:
                    new_poly.AddGeometry(hole.Clone())

            return new_poly

        def _smooth_polygon(poly, iterations, offset):
            ext = poly.GetGeometryRef(0)
            ext_coords = _ring_to_coords(ext)
            ext_smooth = _chaikin(ext_coords, iterations, offset)
            new_ext = _coords_to_ring(ext_smooth)

            out = ogr.Geometry(ogr.wkbPolygon)
            out.AddGeometry(new_ext)

            for i in range(1, poly.GetGeometryCount()):
                hole = poly.GetGeometryRef(i)
                hole_coords = _ring_to_coords(hole)
                hole_smooth = _chaikin(hole_coords, iterations, offset)
                out.AddGeometry(_coords_to_ring(hole_smooth))
            return out

        # ---------- Start ----------
        ndwi_ds = gdal.Open(OUTPUT_PATH, gdal.GA_ReadOnly)
        gt = ndwi_ds.GetGeoTransform()
        px = abs(gt[1])
        simplify_tol = px * float(SIMPLIFY_PX)
        ndwi_ds = None

        if not os.path.exists(OUTPUT_PATH):
            print("[Polygonize] Skipped: NDWI not found ->", OUTPUT_PATH)
            return

        base_name = os.path.basename(INPUT_DIR)
        water_mask_path = os.path.join(INPUT_DIR, f"{base_name}_water_mask.tif")
        sieved_path     = os.path.join(INPUT_DIR, f"{base_name}_water_mask_sieved.tif")
        shp_base        = os.path.join(INPUT_DIR, f"{base_name}_water")
        final_base      = os.path.join(INPUT_DIR, f"{base_name}_water_final")

        shp_path   = shp_base + ".shp"
        final_path = final_base + ".shp"

        layer_name = (base_name[:10] or "water")

        print("\n[Polygonize] NDWI -> water mask -> polygon (Shapefile)...")
        print("NDWI used:", OUTPUT_PATH)

        # 1) Water mask
        with rasterio.open(OUTPUT_PATH) as src_ndwi:
            ndwi_arr = src_ndwi.read(1)
            profile  = src_ndwi.profile.copy()

            out = np.full(ndwi_arr.shape, MASK_NODATA, dtype="int16")
            valid = np.isfinite(ndwi_arr)
            water = valid & (ndwi_arr > threshold_ndwi)
            out[water] = WATER_VALUE

            profile.update(dtype="int16", nodata=MASK_NODATA, compress="LZW", count=1)
            with rasterio.open(water_mask_path, "w", **profile) as dst:
                dst.write(out, 1)
        print("Water mask written:", water_mask_path)

        # 2) Sieve
        if MIN_OBJ_PIX and MIN_OBJ_PIX > 0:
            src_ds  = gdal.Open(water_mask_path, gdal.GA_ReadOnly)
            drv_tif = gdal.GetDriverByName("GTiff")
            _remove_if_exists_shp(os.path.splitext(sieved_path)[0])
            out_ds  = drv_tif.CreateCopy(sieved_path, src_ds, strict=0)
            band    = out_ds.GetRasterBand(1)
            band.SetNoDataValue(MASK_NODATA)
            gdal.SieveFilter(srcBand=band, maskBand=None, dstBand=band,
                             threshold=int(MIN_OBJ_PIX), connectedness=int(CONNECTEDNESS))
            out_ds = None
            src_ds = None
            sieve_input_path = sieved_path
            print(f"Sieve applied (min {MIN_OBJ_PIX} px), output:", sieved_path)
        else:
            sieve_input_path = water_mask_path
            print("Sieve skipped (MIN_OBJ_PIX=0).")

        # 3) Polygonize
        src_ds   = gdal.Open(sieve_input_path, gdal.GA_ReadOnly)
        band     = src_ds.GetRasterBand(1)
        band.SetNoDataValue(MASK_NODATA)
        maskBand = band.GetMaskBand()

        drv = ogr.GetDriverByName("ESRI Shapefile")
        _remove_if_exists_shp(shp_base)
        dst_ds = drv.CreateDataSource(shp_path)
        if dst_ds is None:
            raise RuntimeError(f"Shapefile could not be created: {shp_path}")

        srs_wkt = src_ds.GetProjection()
        spref = None
        if srs_wkt:
            spref = osr.SpatialReference()
            try: spref.ImportFromWkt(srs_wkt)
            except Exception: spref = None

        layer = dst_ds.CreateLayer(layer_name, srs=spref, geom_type=ogr.wkbPolygon)
        layer.CreateField(ogr.FieldDefn("value", ogr.OFTInteger))
        gdal.Polygonize(band, maskBand, layer, 0, [], None)
        layer.SyncToDisk()
        dst_ds.SyncToDisk()

        src_chk = ogr.Open(shp_path)
        if src_chk is None:
            raise RuntimeError(f"Intermediate SHP could not be opened: {shp_path}")
        fc = src_chk.GetLayer(0).GetFeatureCount()
        print("Intermediate SHP feature count:", fc)
        src_chk = None

        layer = None
        dst_ds = None
        src_ds = None
        print("Polygonize (intermediate) written:", shp_path)

        # 4) Delete holes -> Smooth -> FINAL shapefile
        src = ogr.Open(shp_path, update=0)
        in_lyr = src.GetLayer(0)

        _remove_if_exists_shp(final_base)
        dst = drv.CreateDataSource(final_path)
        if dst is None:
            raise RuntimeError(f"Final shapefile could not be opened: {final_path}")
        out_lyr = dst.CreateLayer(layer_name, srs=spref, geom_type=ogr.wkbPolygon)
        out_lyr.CreateField(ogr.FieldDefn("value", ogr.OFTInteger))

        for feat in in_lyr:
            geom = feat.GetGeometryRef()
            val = feat.GetField("value")
            if val != 1:
                continue

            def process_polygon(p):
                nohole = _delete_holes_from_polygon(p, DELETE_HOLES_MAX_AREA)
                smooth = _smooth_polygon(nohole, SMOOTH_ITER, SMOOTH_OFFSET)
                simplified = smooth.SimplifyPreserveTopology(simplify_tol)
                return simplified

            out_geom = None
            if geom.GetGeometryType() in (ogr.wkbPolygon, ogr.wkbPolygon25D):
                out_geom = process_polygon(geom)
            elif geom.GetGeometryType() in (ogr.wkbMultiPolygon, ogr.wkbMultiPolygon25D):
                mp = ogr.Geometry(ogr.wkbMultiPolygon)
                for i in range(geom.GetGeometryCount()):
                    p = geom.GetGeometryRef(i)
                    mp.AddGeometry(process_polygon(p))
                for i in range(mp.GetGeometryCount()):
                    g = mp.GetGeometryRef(i)
                    of = ogr.Feature(out_lyr.GetLayerDefn())
                    of.SetGeometry(g.Clone())
                    of.SetField("value", 1)
                    out_lyr.CreateFeature(of)
                    of = None
                mp = None
                continue
            else:
                continue

            of = ogr.Feature(out_lyr.GetLayerDefn())
            of.SetGeometry(out_geom)
            of.SetField("value", 1)
            out_lyr.CreateFeature(of)
            of = None

        out_lyr = None
        dst = None
        in_lyr = None
        src = None
        print("FINAL shapefile written:", final_path)

        # 5) Delete the intermediate files
        if not KEEP_INTERMEDIATES:
            _remove_if_exists_shp(shp_base)
            print("Deleted (intermediate shapefile):", shp_path)
            for p in (sieved_path if os.path.exists(sieved_path) else None, water_mask_path):
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                        print("Deleted:", p)
                    except Exception:
                        pass

        print("[Post-process] Delete holes + Smooth completed.\n")

    # --- Call Polygonize ---
    try:
        water_polygonize_from_ndwi(
            OUTPUT_PATH=OUTPUT_PATH,
            INPUT_DIR=INPUT_DIR,
            threshold_ndwi=0.0,
            MIN_OBJ_PIX=9,
            CONNECTEDNESS=8,
            WATER_VALUE=1,
            MASK_NODATA=-9999,
            DELETE_HOLES_MAX_AREA=0.0,
            SMOOTH_ITER=5,
            SMOOTH_OFFSET=0.35,
            SIMPLIFY_PX=1.2,
            KEEP_INTERMEDIATES=False
        )
    except Exception as e:
        print("[Polygonize] Error:", e)

    # === ADD AN AREA (HECTARE) COLUMN TO THE FINAL SHAPEFILE ===
    final_shp = os.path.join(INPUT_DIR, os.path.basename(INPUT_DIR) + "_water_final.shp")

    if os.path.exists(final_shp):
        print("[Area Calculation] Calculating area in hectares:", final_shp)
        ds = ogr.Open(final_shp, update=1)
        layer = ds.GetLayer()

        field_names = [f.GetName() for f in layer.schema]
        if "area" not in field_names:
            field_defn = ogr.FieldDefn("area", ogr.OFTReal)
            field_defn.SetWidth(32)
            field_defn.SetPrecision(6)
            layer.CreateField(field_defn)

        for feat in layer:
            geom = feat.GetGeometryRef()
            if geom is not None:
                area_m2 = geom.GetArea()
                area_ha = area_m2 / 10000.0
                feat.SetField("area", area_ha)
                layer.SetFeature(feat)

        ds = None
        print("[Area Calculation] Done. 'area' column added in hectares.")
    else:
        print("[Area Calculation] Final shapefile not found, step skipped.")

    # === FINAL SHAPEFILE: delete polygons with area <= 10 ha ===
    final_shp = os.path.join(INPUT_DIR, os.path.basename(INPUT_DIR) + "_water_final.shp")

    if os.path.exists(final_shp):
        print("[Filter] Polygons with area <= 10 ha will be deleted:", final_shp)
        ds = ogr.Open(final_shp, update=1)
        lyr = ds.GetLayer()

        field_names = [f.GetName() for f in lyr.schema]
        if "area" not in field_names:
            fd = ogr.FieldDefn("area", ogr.OFTReal)
            fd.SetWidth(32); fd.SetPrecision(6)
            lyr.CreateField(fd)

        to_delete = []
        for feat in lyr:
            geom = feat.GetGeometryRef()
            if geom is None:
                to_delete.append(feat.GetFID())
                continue
            area_ha = geom.GetArea() / 10000.0
            feat.SetField("area", float(area_ha))
            lyr.SetFeature(feat)

            if (not np.isfinite(area_ha)) or (area_ha <= 10.0):
                to_delete.append(feat.GetFID())

        for fid in to_delete:
            lyr.DeleteFeature(fid)

        deleted = len(to_delete)
        ds = None
        print(f"[Filter] Number of deleted polygons: {deleted} (area <= 10 ha).")
    else:
        print("[Filter] Final shapefile not found, step skipped.")

    # === LAST STEP: reproject NDWI.tif and the final shapefile to the CRS of the input bands ===

    def _remove_shp_family(base_wo_ext):
        for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
            p = base_wo_ext + ext
            if os.path.exists(p):
                try: os.remove(p)
                except Exception: pass

    # 1) NDWI.tif -> target CRS
    if os.path.exists(OUTPUT_PATH) and (target_crs or target_srs_param):
        tmp_ndwi = OUTPUT_PATH.replace(".tif", "_reproj_tmp.tif")
        dst_crs = target_crs if target_crs else target_srs_param

        with rasterio.open(OUTPUT_PATH) as src:
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds
            )
            kwargs = src.meta.copy()
            kwargs.update(
                driver="GTiff",
                crs=dst_crs,
                transform=transform,
                width=width,
                height=height,
                dtype="float32",
                count=1,
                compress="LZW",
                nodata=-9999.0,
            )
            src_data = src.read(1)
            with rasterio.open(tmp_ndwi, "w", **kwargs) as dst:
                out = np.full((height, width), -9999.0, dtype="float32")
                reproject(
                    source=src_data,
                    destination=out,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    src_nodata=src.nodata,
                    dst_nodata=-9999.0,
                    resampling=Resampling.bilinear,
                )
                dst.write(out, 1)

        try:
            os.remove(OUTPUT_PATH)
        except Exception:
            pass
        os.replace(tmp_ndwi, OUTPUT_PATH)
        print("[Reproject] NDWI converted to the input CRS ->", OUTPUT_PATH)
    else:
        print("[Reproject] NDWI not found; skipped:", OUTPUT_PATH)

    # 2) FINAL SHP -> target CRS
    base_name   = os.path.basename(INPUT_DIR)
    final_base  = os.path.join(INPUT_DIR, base_name + "_water_final")
    final_shp   = final_base + ".shp"
    tmp_base    = final_base + "_reproj_tmp"
    tmp_shp     = tmp_base + ".shp"

    if os.path.exists(final_shp):
        dst_srs = None
        if target_srs_param:
            dst_srs = target_srs_param
        elif target_crs:
            try:
                dst_srs = target_crs.to_wkt()
            except Exception:
                dst_srs = None

        if not dst_srs:
            print("[Reproject][SHP] Warning: Target CRS could not be determined. Reprojection skipped ->", final_shp)
        else:
            _remove_shp_family(tmp_base)
            vt_opts = gdal.VectorTranslateOptions(dstSRS=dst_srs, format="ESRI Shapefile")
            res = gdal.VectorTranslate(tmp_shp, final_shp, options=vt_opts)

            if (res is None) or (not os.path.exists(tmp_shp)):
                print("[Reproject][SHP] ERROR: VectorTranslate failed ->", final_shp)
                _remove_shp_family(tmp_base)
            else:
                _remove_shp_family(final_base)
                for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
                    src_f = tmp_base + ext
                    if os.path.exists(src_f):
                        os.replace(src_f, final_base + ext)
                print("[Reproject][SHP] FINAL shapefile converted to the input CRS ->", final_shp)
    else:
        print("[Reproject][SHP] Final shapefile not found; skipped:", final_shp)

    # === RGB (B04,B03,B02) -> Create a color GeoTIFF ===
    try:
        print("\n[RGB Creation] Scanning the folder for the B04,B03,B02 bands...")

        b2_list = glob.glob(os.path.join(INPUT_DIR, "**", "*B02*10m.jp2"), recursive=True)
        b3_list = glob.glob(os.path.join(INPUT_DIR, "**", "*B03*10m.jp2"), recursive=True)
        b4_list = glob.glob(os.path.join(INPUT_DIR, "**", "*B04*10m.jp2"), recursive=True)

        if b2_list and b3_list and b4_list:
            b2 = b2_list[0]
            b3 = b3_list[0]
            b4 = b4_list[0]

            base_name = os.path.basename(INPUT_DIR)
            output_rgb = os.path.join(INPUT_DIR, f"{base_name}.tif")
            tmp_rgb = os.path.join(INPUT_DIR, f"{base_name}_tmp.tif")

            print("[RGB Creation] Bands found:")
            print("  B02 (Blue): ", b2)
            print("  B03 (Green):", b3)
            print("  B04 (Red):  ", b4)



            with rasterio.open(b4) as src_ref, \
                 rasterio.open(b3) as src_g, \
                 rasterio.open(b2) as src_b:

                ref_profile = src_ref.profile.copy()
                ref_profile.update(
                    driver="GTiff",
                    count=3,
                    compress="LZW",
                    nodata=None,
                    dtype=rasterio.uint8
                )

                R = np.empty((src_ref.height, src_ref.width), dtype=src_ref.dtypes[0])
                G = np.empty_like(R, dtype=src_g.dtypes[0])
                B = np.empty_like(R, dtype=src_b.dtypes[0])

                R[:, :] = src_ref.read(1)

                reproject(
                    source=rasterio.band(src_g, 1),
                    destination=G,
                    src_transform   = src_g.transform,
                    src_crs         = src_g.crs,
                    dst_transform   = src_ref.transform,
                    dst_crs         = src_ref.crs,
                    resampling      = Resampling.bilinear
                )
                reproject(
                    source=rasterio.band(src_b, 1),
                    destination=B,
                    src_transform   = src_b.transform,
                    src_crs         = src_b.crs,
                    dst_transform   = src_ref.transform,
                    dst_crs         = src_ref.crs,
                    resampling      = Resampling.bilinear
                )

                def to_uint8(band):
                    b = band.astype("float32")
                    vmin = np.nanpercentile(b, 2)
                    vmax = np.nanpercentile(b, 98)
                    scaled = (b - vmin) / (vmax - vmin + 1e-6)
                    scaled = (scaled * 255.0).clip(0, 255).astype("uint8")
                    return scaled

                R8 = to_uint8(R)
                G8 = to_uint8(G)
                B8 = to_uint8(B)

                with rasterio.open(tmp_rgb, "w", **ref_profile) as dst:
                    dst.write(R8, 1)
                    dst.write(G8, 2)
                    dst.write(B8, 3)
                    dst.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)

            translate_opts = gdal.TranslateOptions(creationOptions=["COMPRESS=LZW", "PHOTOMETRIC=RGB"])
            gdal.Translate(destName=output_rgb, srcDS=tmp_rgb, options=translate_opts)
            try:
                os.remove(tmp_rgb)
            except Exception:
                pass

            print("[RGB Creation] Color RGB GeoTIFF written ->", output_rgb)
        else:
            print("[RGB Creation] One of the required bands (B02,B03,B04) was not found.")
    except Exception as e:
        print("[RGB Creation] Error:", e)

    print(f"\n[DONE] {INPUT_DIR}")


# =====================================================================
# MAIN EXECUTION
# =====================================================================

if __name__ == "__main__":

    # --- STEP 1: Move the zips into numbered folders in date order ---
    created_folders = organize_sentinel_zips(ZIP_SOURCE_DIR, OUTPUT_BASE_DIR, START_NUMBER)

    if not created_folders:
        # If no zip was found, scan the existing numbered folders
        print("\n[INFO] No zip found, existing numbered folders will be scanned...")
        all_items = os.listdir(OUTPUT_BASE_DIR)
        created_folders = sorted([
            os.path.join(OUTPUT_BASE_DIR, d)
            for d in all_items
            if os.path.isdir(os.path.join(OUTPUT_BASE_DIR, d)) and d.isdigit()
        ], key=lambda x: int(os.path.basename(x)))

        if not created_folders:
            print("[ERROR] No numbered folders found either, stopping.")
        else:
            print(f"[INFO] {len(created_folders)} numbered folders found.")

    if created_folders:
        # --- STEP 2 & 3: Extract and process the zip in each folder ---
        for folder in created_folders:
            print(f"\n{'=' * 70}")
            print(f"  FOLDER: {folder}")
            print(f"{'=' * 70}")


            # Extract the zip (skips if there is none, but processing continues)

            extracted = extract_zips_in_folder(folder)
            if not extracted:
                print(f"  [INFO] Zip not found or could not be extracted, continuing with the existing files: {folder}")

            # Run all processes (NDWI, SHP, RGB, etc.)
            try:
                process_single_folder(folder)
            except Exception as e:
                print(f"  [ERROR] Error while processing {folder}: {e}")


                traceback.print_exc()
                continue

        print(f"\n{'=' * 70}")
        print(f"  ALL PROCESSES COMPLETED!")
        print(f"  Number of processed folders: {len(created_folders)}")
        print(f"{'=' * 70}")