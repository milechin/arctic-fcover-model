import glob
import os
import shutil
import numpy as np
import pandas as pd
import rasterio
from rasterio import Affine
from rasterio.warp import reproject
from osgeo import gdal

#### PARAMETERS TO BE PASSED IN ####
outdir = '/projectnb/modislc/users/seamorez/HLS_FCover/output/'
indir = '/projectnb/modislc/users/seamorez/HLS_FCover/input/HLS30/'
start_yr = 2016
end_yr = 2023
phenometrics = ['50PCGI','Peak','50PCGD']
metrics = ['dem', 'water', 'slope', 'aspect']
# Import ABoVE tiles UID list
# Import reprojection info csv

#### PARAMETERS ####
years = np.arange(start_yr, end_yr+1, 1)

################################### STEP 1 ###################################
# Recover dem and water tiles
tiles_list = glob.glob(outdir+'0*')
tiles_list = [os.path.basename(tile.split("/")[-1]) for tile in tiles_list]

for tile in tiles_list:
    if (os.path.isdir(indir+tile)):
        print(tile,'exists')
        
    else:
        print(tile)
        os.makedirs(indir+tile+'/images')
        shutil.copy('/projectnb/modislc/data/dem/usgs_ned/hls_tiles/dem/dem_'+tile+'.tif',
                        indir+tile+'/images')
        shutil.copy('/projectnb/modislc/data/dem/usgs_ned/hls_tiles/slope/slope_'+tile+'.tif',
                        indir+tile+'/images')
        shutil.copy('/projectnb/modislc/data/dem/usgs_ned/hls_tiles/aspect/aspect_'+tile+'.tif',
                        indir+tile+'/images')
        shutil.copy('/projectnb/modislc/data/water/hls_tiles/water_'+tile+'.tif',
                        indir+tile+'/images')