import glob
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import Affine
from rasterio.warp import reproject
from rasterio.merge import merge
from osgeo import gdal

#### PARAMETERS TO BE PASSED IN ####
outdir = '/projectnb/modislc/users/seamorez/HLS_FCover/S1_input/retiled/'
indir = '/projectnb/modislc/users/seamorez/HLS_FCover/S1_input/DV/'

#%%
################################### STEP 1 ###################################
# Retile
def retile(tile, img, out_path):
    # Open the target raster (the one with the grid you want to match)
    with rasterio.open('/projectnb/modislc/projects/above/tiles/above_regions/Region_ID.B'+tile+'.tif') as tgt:
        target_profile = tgt.profile  # Get the profile of the target raster
    
    target_profile.update(dtype='int16',nodata=32767,count=2)
    #print(target_profile)
                
    # Open the source raster
    with rasterio.open(img) as src:
        source_data = src.read()  # Read the data from the first band (adjust as needed)
    
    #source_data = source_data.astype(float)
    #source_data[source_data==32767] = np.nan
    
    # Create a new raster file for the reprojected data
    with rasterio.open(out_path, 'w', **target_profile) as dst:
        for band_idx in range(src.count):
            reproject(
                source=source_data[band_idx],
                src_nodata=32767,
                destination=rasterio.band(dst, band_idx+1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=target_profile['transform'],
                dst_crs=target_profile['crs'],
                resampling=rasterio.enums.Resampling.med  # Adjust resampling method if needed
            )
        
    return None


def merge_rasters(output_path, img_paths):
    with rasterio.open(img_paths[0]) as src:
        count = src.count
        dtype = src.dtypes[0]
        crs = src.crs

    src_files = [rasterio.open(img) for img in img_paths]
    merged_raster, merged_transform = merge(src_files)

    with rasterio.open(
        output_path, "w", driver="GTiff",
        height=merged_raster.shape[1], width=merged_raster.shape[2],
        count=count, dtype=dtype, nodata=32767, crs=crs, transform=merged_transform
    ) as dst:
        dst.write(merged_raster)

    for src in src_files:
        src.close()


# Retile imagery
# Get tile list
s1_imgs = [os.path.basename(path) for path in glob.glob(indir+'h*.tif')]
s1_imgs = list(set([name[:18] for name in s1_imgs]))

# Check outdir to see if the retiled raster already exists
for file_name in s1_imgs:
    if os.path.exists(outdir+file_name+'.tif'):
        print(file_name+': Retiled raster exists!')
    
    else:
        # For rasters that need retiling, first merge the chunks
        s1_imyrsn_pairs = glob.glob(indir+file_name+'*.tif')
        print(file_name+': '+str(len(s1_imyrsn_pairs))+' chunks')
        # Merging takes a long time, so make sure it hasn't already been merged!
        if os.path.exists(indir+file_name+'.tif'):
            print('Merged raster exists!')
        else:
            merge_rasters(indir+file_name+'.tif', s1_imyrsn_pairs)
        
        # Then, retile to ABoVE grid at 30 m
        retile(file_name[:6], indir+file_name+'.tif', outdir+file_name+'.tif')
