import os
import sys

import geopandas as gpd
import numpy as np
import rasterio.mask
from rasterio.features import rasterize
from shapely.geometry import Polygon
from shapely.ops import cascaded_union

raster_path = "/mnt/data/train/"

# load raster
with rasterio.open(raster_path, "r") as src:
    raster_img = src.read()
    raster_meta = src.meta

# load a shapefile or GeoJson
train_df = gpd.read_file(shape_path)

# Verify crs
if train_df.crs != src.crs:
    print(
        " Raster crs : {}, Vector crs : {}.\n Convert vector and raster to the same CRS.".format(src.crs, train_df.crs))
    sys.exit("crs error")


# Function that generates the mask
def poly_from_utm(polygon, transform):
    poly_pts = []

    poly = cascaded_union(polygon)

    for i in np.array(poly.exterior.coords):
        poly_pts.append(~transform * tuple(i))

    new_poly = Polygon(poly_pts)
    print(new_poly)
    sys.exit("worked?")
    return new_poly


poly_shp = []
im_size = (src.meta['height'], src.meta['width'])
for num, row in train_df.iterrows():
    if row['geometry'].geom_type == 'Polygon':
        poly = poly_from_utm(row['geometry'], src.meta['transform'])
        poly_shp.append(poly)
    else:
        for p in row['geometry']:
            poly = poly_from_utm(p, src.meta['transform'])
            poly_shp.append(poly)

if (len(poly_shp) > 0):
    mask = rasterize(shapes=poly_shp, out_shape=im_size)
else:
    return

# Save
mask = mask.astype("uint16")

bin_mask_meta = src.meta.copy()
bin_mask_meta.update({'count': 1})
os.chdir(output_path)
with rasterio.open(file_name, 'w', **bin_mask_meta) as dst:
    dst.write(mask * 255, 1)
