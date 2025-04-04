import sys
import os
import time
import pickle
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
import rioxarray
import numpy as np
from sklearn.ensemble import RandomForestRegressor

#%%
start_time = time.time()

tile = 'Bh07v01'#sys.argv[1]
start_year = 2016#sys.argv[2]
end_year = 2023#sys.argv[3]

#### PARAMETERS TO BE PASSED IN ####
nmodels=20
chunk_size={'x':1000, 'y':1000}

outdir = '/projectnb/modislc/users/seamorez/HLS_FCover/output/'
model_traindir = '/projectnb/modislc/users/seamorez/HLS_FCover/model_training/MC_resampling/'
model_outdir = '/projectnb/modislc/users/seamorez/HLS_FCover/model_outputs/MC_outputs/'

#%%
for year in range(int(start_year), int(end_year)+1):
    print(year)
    # List to hold all the opened rasters
    data_arrays = []
    
    for model in range(1,nmodels+1):
        print('Stacking model '+str(model))
        path = f"{model_outdir}{year}/FCover_{tile}_{year}_rfr_{model}.tif"
        da = rioxarray.open_rasterio(path, masked=True, chunks=chunk_size).squeeze()  # squeeze to remove band dim if present
        data_arrays.append(da)
    
    # Stack along a new dimension called 'model'
    stacked = xr.concat(data_arrays, dim='model')
    
    # Compute mean and standard error (SE) across models
    mean = stacked.mean(dim='model').compute()
    se = (stacked.std(dim='model') / np.sqrt(nmodels)).compute()
    
    # Output maps
    if not os.path.isdir(f"{model_outdir}final/{year}"):
        os.makedirs(f"{model_outdir}final/{year}")
    if os.path.exists(f"{model_outdir}final/{year}/FCover_{tile}_{year}_rfr_mean.tif"):
        continue
    
    mean.rio.to_raster(f"{model_outdir}final/{year}/FCover_{tile}_{year}_rfr_mean.tif")
    se.rio.to_raster(f"{model_outdir}final/{year}/FCover_{tile}_{year}_rfr_se.tif")


#%%
end_time = time.time()
elapsed_time = end_time - start_time
# Convert to hours, minutes, and seconds
hours = int(elapsed_time // 3600)
minutes = int((elapsed_time % 3600) // 60)
seconds = int(elapsed_time % 60)

# Print to log file
print(f"Job completed in {hours}h {minutes}m {seconds}s.")