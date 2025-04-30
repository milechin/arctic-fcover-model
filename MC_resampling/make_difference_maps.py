#%% This script outputs difference maps
import numpy as np
import pandas as pd
import os
import glob
import xarray as xr
import rioxarray
import rasterio as rio
from rasterio.mask import mask
import geopandas as gpd
import matplotlib.pyplot as plt
import colormaps as cmaps
import seaborn as sb
from dask.distributed import Client, LocalCluster
from dask.diagnostics import ProgressBar
import dask
import pymannkendall as mk
from scipy.stats import linregress


#%% Functions
def difference(path_1, path_2, out_path): 
    with rio.open(path_1) as src1:
        meta = src1.meta; ndata = src1.nodata
        map_1 = src1.read()
    with rio.open(path_2) as src2:
        map_2 = src2.read()
    change_map = map_2 - map_1
    change_map[np.isnan(map_1)] = np.nan
    change_map[np.isnan(map_2)] = np.nan
    
    with rio.open(out_path, 'w', **meta) as dst:
        dst.write(change_map)
    
    return None

#%% Output trends for each LC class and the significance levels
# Configure Dask LocalCluster to use NSLOTS threads
with open('/projectnb/modislc/users/seamorez/HLS_FCover/scripts/techmanuscript_tiles.txt', 'r') as f:
    tiles = [line.strip() for line in f if line.strip()]
    
for tile in tiles:
    if os.path.exists('/projectnb/modislc/users/seamorez/HLS_FCover/model_outputs/MC_outputs/final/change/FCover_'+tile+'_2016_2023.tif'):
       continue
   
    print(tile)
    path_1 = '/projectnb/modislc/users/seamorez/HLS_FCover/model_outputs/MC_outputs/final/2016/FCover_'+tile+'_2016_rfr_mean.tif'
    path_2 = '/projectnb/modislc/users/seamorez/HLS_FCover/model_outputs/MC_outputs/final/2023/FCover_'+tile+'_2023_rfr_mean.tif'
    difference(path_1, path_2, '/projectnb/modislc/users/seamorez/HLS_FCover/model_outputs/MC_outputs/final/change/FCover_'+tile+'_2016_2023.tif')
