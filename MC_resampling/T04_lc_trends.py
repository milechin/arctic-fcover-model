#%% This script outputs trend maps
import numpy as np
import os
import glob
import xarray as xr
import rioxarray
import rasterio as rio
from dask.distributed import Client, LocalCluster, progress
import dask
import pymannkendall as mk



#%% Functions
def trend(data):

    # Check if data is empty or containsonly one non-NaN value
    if data.size == 0 or data[~np.isnan(data)].size == 1:
        return -9999, -9999, -9999

    # Check if data only contains NaN values
    if np.all(np.isnan(data)):
        return -8888, -8888, -8888

    try:
        reg = mk.original_test(data)
    except ValueError as e:
        print(f"Error in trend calculation: {e}")
        return -7777, -7777, -7777
    
    return reg.slope, reg.intercept, reg.p


def calc_trend(tile, lc):

    # file_pattern = "/projectnb/modislc/users/seamorez/HLS_Pheno/clim_dat/ERA5_Land_T/2.2.1/AGDD_*.tif"
    file_pattern = "/projectnb/modislc/users/seamorez/HLS_FCover/model_outputs/MC_outputs/final/**/FCover_"+tile+"_*_rfr_mean.tif"
    raster_files = glob.glob(file_pattern)
    raster_files.sort()
    print(raster_files)

    # Open and stack rasters over time
    # Initialize lists for datasets and time coordinates
    datasets = []
    time_coords = []

    # Read each raster file
    for file in raster_files:
        # Extract the year
        year = int(os.path.basename(file)[-17:-13])
        print(year)
        time_coords.append(year)

        # Open the raster file as a dataset
        # ds = xr.open_dataset(file, engine="rasterio")
        ds = rioxarray.open_rasterio(file, chunks={'x': 6000, 'y': 6000})

        ds = ds.isel(band=lc)
        ds = ds.where(ds != -9999)
        # Add a time dimension and coordinate to the dataset
        ds = ds.expand_dims(time=[year])
        # Append the dataset to the list
        datasets.append(ds)

    # Concatenate along the time dimension
    LC_ts = xr.concat(datasets, dim='time')
    LC_ts = LC_ts.chunk({'time': -1})
    print(LC_ts)
    print(type(LC_ts.data))

    # Drop the band coordinate as it is not needed.
    # LC_ts = LC_ts.reset_coords("band", drop=True)

    # Apply the function using apply_ufunc
    # The function should return two arrays: slope and intercept
    slope, intercept, pval = xr.apply_ufunc(
        trend,
        LC_ts,  # The data variable to apply the function to
        input_core_dims=[['time']],  # Specify that 'time' is the core dimension
        output_core_dims=[[], [], []],
        vectorize='True',  # Vectorize the function to apply it over the spatial dimensions
        dask='parallelized',
        output_dtypes=[np.float32, np.float32, np.float32],  # Specify the output data types
    )

    # When using LocalCluster, need to use the following lines to make progress bar to work
    # source: https://docs.dask.org/en/stable/diagnostics-distributed.html#progress-bar
    x = dask.persist(slope, intercept, pval)

    # Get progress bar to appear
    progress(x)

    # Get the results
    result = dask.compute(x)

    # Parse the results
    slope = result[0][0]
    intercept = result[0][1]
    pval = result[0][2]

    rast_0 = raster_files[0]

    # Print the results
    print(slope, intercept, pval)

    return slope, intercept, pval, rast_0

# Local Cluster should be started within the __name__ == "__main__" block
# https://stackoverflow.com/questions/55057957/an-attempt-has-been-made-to-start-a-new-process-before-the-current-process-has-f
if __name__ == "__main__":
    # %% Output trends for each LC class and the significance levels
    # Configure Dask LocalCluster to use NSLOTS threads
    NSLOTS = int(os.environ.get("NSLOTS"))

    cluster = LocalCluster(
        n_workers=1,               # 1 worker with multiple threads
        threads_per_worker=NSLOTS,  # Number of threads = NSLOTS
        processes=False,           # Use threads, not separate processes
        # dashboard_address=None     # Disable dashboard (optional on HPC)
    )

    client = Client(cluster)
    print(client)

    with open('/projectnb/modislc/users/seamorez/HLS_FCover/scripts/techmanuscript_tiles.txt', 'r') as f:
        tiles = [line.strip() for line in f if line.strip()]

    for tile in tiles[1:2]:
        if os.path.exists('/projectnb/modislc/users/seamorez/HLS_FCover/model_outputs/MC_outputs/final/trends/FCover_'+tile+'_shrub.tif'):
            print('Trend already exists for '+tile)
            continue

        print(tile)
        slope, intercept, pval, rast_0 = calc_trend(tile, 4)

        # Export results
        with rio.open(rast_0) as src:
            meta = src.meta
            transform1 = src.transform

        # Scale and Prepare output shape: [band, y, x]
        slope_np = np.around(slope.values*1000)  # shape: [band, y, x]
        intercept_np = np.around(intercept.values)
        pval_np = np.around(pval.values*1000)    # shape: [band, y, x]
        combined = np.concatenate([slope_np, intercept_np, pval_np], axis=0)  # shape: [3*bands, y, x]
        print(combined.shape)

        # Update metadata
        meta.update(count=3, transform=transform1, dtype='int16', nodata=32767)

        with rio.open('/projectnb/modislc/users/seamorez/HLS_FCover/model_outputs/MC_outputs/final/trends/FCover_'+tile+'_shrub.tif', 'w', **meta) as dst:
            dst.write(combined)

