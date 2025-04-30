#%%
import sys
import os
import shutil
import time
import pickle
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
import rioxarray
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import dask


#%%
start_time = time.time()

tile = sys.argv[1]
start_year = sys.argv[2]
end_year = sys.argv[3]
#tile = 'Bh07v01'   #sys.argv[1]
#start_year = 2016  #sys.argv[2]
#end_year = 2023    #sys.argv[3]

#### PARAMETERS TO BE PASSED IN ####
nmodels = int(sys.argv[4])

# Have each band be its own chunk
chunk_size={'band': 1, 'x':6000, 'y':6000}

# Get TMPDIR and NSLOTS from environment variables
TMPDIR = os.environ.get("TMPDIR")
NSLOTS = int(os.environ.get("NSLOTS"))

# Configure dask to run as threads and set number of workers based on NSLOT value.
dask.config.set(scheduler='threads') 
dask.config.set(num_workers=NSLOTS)

# Set paths.  Output is set to TMPDIR of the job.
outdir = f"{TMPDIR}/output/"
model_traindir = '/projectnb/modislc/users/seamorez/HLS_FCover/model_training/MC_resampling/'
model_outdir = '/projectnb/modislc/users/seamorez/HLS_FCover/model_outputs/MC_outputs/'


#%%
for year in range(int(start_year), int(end_year)+1):
    print(year)


    # Check if output directory exists, if not create it
    if not os.path.isdir(f"{outdir}final/{year}"):
        os.makedirs(f"{outdir}final/{year}")

    # Check if the output file already exists
    if os.path.exists(f"{model_outdir}final/{year}/FCover_{tile}_{year}_rfr_mean.tif"):
       continue

    # Assemble a list of paths
    file_paths = []
    for model in range(1, nmodels+1):
        path = f"{model_outdir}{year}/FCover_{tile}_{year}_rfr_{model}.tif"
        file_paths.append(path)
    
    # Read in the raster data and stack them
    stacked = xr.open_mfdataset(
                                file_paths,
                                concat_dim=[
                                    pd.Index(
                                        np.arange(len(file_paths)),
                                        name="model"
                                        ),
                                    ],
                                combine="nested",
                                parallel=True,
                                chunks=chunk_size)["band_data"]
    
    # Compute mean and standard error (SE) across models
    # Since this is a dask array, the calculation will not occur now, but will be 
    # triggered automatically when rio.to_raster is called.
    mean = stacked.mean(dim='model')
    se = (stacked.std(dim='model') / np.sqrt(nmodels))
    
    # Save the results. (This will trigger compute.)
    mean.rio.to_raster(f"{outdir}final/{year}/FCover_{tile}_{year}_rfr_mean.tif", num_threads=str(NSLOTS))
    se.rio.to_raster(f"{outdir}final/{year}/FCover_{tile}_{year}_rfr_se.tif", num_threads=str(NSLOTS))


## TO DO
#for year in range(int(start_year), int(end_year)+1):
    # Define paths in TMPDIR
    mean_tmp = f"{outdir}final/{year}/FCover_{tile}_{year}_rfr_mean.tif"
    se_tmp = f"{outdir}final/{year}/FCover_{tile}_{year}_rfr_se.tif"
    
    # Define final output path (outside TMPDIR if outdir is just a placeholder for TMP)
    final_dir = f"{model_outdir}final/{year}/"
    os.makedirs(final_dir, exist_ok=True)
    
    # Copy files to final destination
    shutil.copy(mean_tmp, os.path.join(final_dir, os.path.basename(mean_tmp)))
    shutil.copy(se_tmp, os.path.join(final_dir, os.path.basename(se_tmp)))

#%%
end_time = time.time()
elapsed_time = end_time - start_time
# Convert to hours, minutes, and seconds
hours = int(elapsed_time // 3600)
minutes = int((elapsed_time % 3600) // 60)
seconds = int(elapsed_time % 60)

# Print to log file
print(f"Job completed in {hours}h {minutes}m {seconds}s.")