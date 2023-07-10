import os
import sys
import rasterio
from rasterio.plot import reshape_as_image
import rasterio.mask
from rasterio.features import rasterize

import pandas as pd
import geopandas as gpd
from shapely.geometry import mapping, Point, Polygon
from shapely.ops import cascaded_union

import numpy as np
import cv2
import matplotlib.pyplot as plt

def generate_mask(raster_path, shape_path, output_path, file_name):
    
    """
    Function that generates a binary mask from a vector file (shp or geojson)
    
    raster_path = path to the .tif;

    shape_path = path to the shapefile or GeoJson.

    output_path = Path to save the binary mask.

    file_name = Name of the file.
    
    """
    
    #load raster
    with rasterio.open(raster_path, "r") as src:
        raster_img = src.read()
        raster_meta = src.meta
    
    #load a shapefile or GeoJson
    train_df = gpd.read_file(shape_path)
    
    #Verify crs
    if train_df.crs != src.crs:
        print(" Raster crs : {}, Vector crs : {}.\n Convert vector and raster to the same CRS.".format(src.crs,train_df.crs))
        sys.exit("crs error")
    #print(" Raster crs : {}, Vector crs : {}.\n Convert vector and raster to the same CRS.".format(src.crs,train_df.crs))
        
    #Function that generates the mask
    def poly_from_utm(polygon):
        poly_pts = []

        poly = cascaded_union(polygon)

        for i in np.array(poly.exterior.coords):
            poly_pts.append(tuple(i))
        
        new_poly = Polygon(poly_pts)  
        return new_poly
    
    
    poly_shp = []
    im_size = (src.meta['height'], src.meta['width'])
    for num, row in train_df.iterrows():
        if row['geometry'].geom_type == 'Polygon':
            poly = poly_from_utm(row['geometry'])
            poly_shp.append(poly)
        else:
            for p in row['geometry']:
                poly = poly_from_utm(p)
                poly_shp.append(poly)

    if(len(poly_shp)>0):
        mask = rasterize(shapes=poly_shp,out_shape=im_size,transform = src.meta['transform'])
    else:
        mask = np.zeros(im_size)
    
    #Save
    mask = mask.astype("uint16")
    
    bin_mask_meta = src.meta.copy()
    
    bin_mask_meta.update({'count': 1})
    os.chdir(output_path)
    with rasterio.open(file_name, 'w', **bin_mask_meta) as dst:
        dst.write(np.array([mask * 255]))
        
        
        
if not os.path.exists(sys.argv[1]) or not os.path.exists(sys.argv[2]):
    sys.exit("Bad Params")

geojsonDir = sys.argv[1]
tifDirList = os.listdir(sys.argv[2])

print("\ntifDirList length: " + str(len(tifDirList)))

print("\nCreating output directory if it doesnt exist")
if not os.path.exists(sys.argv[3]):
    os.mkdir(sys.argv[3])
    print("\nCreated: " + sys.argv[3])
else:
    print("\nAlready Exists: " + sys.argv[3])
    
if len(os.listdir(sys.argv[3])) == len(tifDirList):
    sys.exit("Operation Done Already")
    
print("\nCommencing mask gen")

#this might need to be changed based on the format of your dataset names
count = 0
for raster_path in tifDirList:
    imgnum = raster_path[27:-4]
    shape_path = "buildings_AOI_3_Paris_"+imgnum+".geojson"
    #print(raster_path, shape_path)
    if not os.path.exists(sys.argv[3] + "buildings_AOI_3_Paris_"+imgnum+".tif"):
        generate_mask(sys.argv[2]+"/"+raster_path,sys.argv[1]+"/"+shape_path, sys.argv[3],"buildings_AOI_3_Paris_"+imgnum+".tif")
    count+=1
    #if count == 10:
      #break
    

print("\nMask gen complete. " + str(count) + " masks made. Please check your directory")