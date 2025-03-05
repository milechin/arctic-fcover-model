import glob
import os
import numpy as np
import pandas as pd
import rasterio
from rasterio import Affine
from rasterio.warp import reproject
from osgeo import gdal


#%%
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


# Read HLS tiles to handle
with open('/usr3/graduate/seamorez/GitHub/MSLSP/tileLists/CAN_Tuk.txt', 'r') as file:
        lines = file.read().splitlines()

#%%
################################### STEP 1 ###################################
# Extract synthetic imagery for all tiles

# For each HLS tile
for subdir in lines:
    print(subdir)
    # For each year
    for year in years:
        # For each phenometric: 50PCGI, Peak, 50PCGD
        for idx, metric in enumerate(phenometrics):
            file_list = ['/phenoMetrics/MSLSP_'+subdir+'_'+str(year)+'.nc:'+metric+'_b2',
                         '/phenoMetrics/MSLSP_'+subdir+'_'+str(year)+'.nc:'+metric+'_b3',
                         '/phenoMetrics/MSLSP_'+subdir+'_'+str(year)+'.nc:'+metric+'_b4',
                         '/phenoMetrics/MSLSP_'+subdir+'_'+str(year)+'.nc:'+metric+'_b5',
                         '/phenoMetrics/MSLSP_'+subdir+'_'+str(year)+'.nc:'+metric+'_b6',
                         '/phenoMetrics/MSLSP_'+subdir+'_'+str(year)+'.nc:'+metric+'_b7']
            # Generate synethetic spectra tifs
            # Read metadata of first file
            with rasterio.open('netcdf:'+outdir+subdir+file_list[0], 'r') as src0:
                meta = src0.meta
            # Update meta to reflect the number of layers
            meta.update(driver = 'GTiff', count = len(file_list))
            
            #if (os.path.isfile(outdir+subdir+'/phenoMetrics/MSLSP_'+subdir+'_'+str(year)+'_'+metric+'.tif')):
            #    continue  # COMMENT OUT IF YOU NEED TO RERUN
            
            # Write out new stacked raster for the phenometric
            with rasterio.open(outdir+subdir+'/phenoMetrics/MSLSP_'+subdir+'_'+str(year)+'_'+metric+'.tif', 'w', **meta) as dst:
                for id, layer in enumerate(file_list, start=1):
                    with rasterio.open('netcdf:'+outdir+subdir+layer) as src1:
                        dst.write_band(id, src1.read(1))


#%%
################################### STEP 2 ###################################
# Retile and delete all input tiles

def add_pixel_fn(filename: str, band: str):
    """inserts pixel-function into vrt file named 'filename'
    Args:
        filename (:obj:`string`): name of file, into which the function will be inserted
        resample_name (:obj:`string`): name of resampling method
    """

    #header = """  <VRTRasterBand dataType="Int16" band="1" subClass="VRTDerivedRasterBand">"""
    header = '  <VRTRasterBand dataType="Int16" band="'+band+'" subClass="VRTDerivedRasterBand">'
    contents = """
    <PixelFunctionType>average</PixelFunctionType>
    <PixelFunctionLanguage>Python</PixelFunctionLanguage>
    <PixelFunctionCode><![CDATA[
import numpy as np
def average(in_ar, out_ar, xoff, yoff, xsize, ysize, raster_xsize,raster_ysize, buf_radius, gt, **kwargs):
    x = np.ma.masked_greater(in_ar, 30000)
    np.mean(x, axis = 0,out = out_ar, dtype = 'int16')
    mask = np.all(x.mask,axis = 0)
    out_ar[mask]=32767
]]>
    </PixelFunctionCode>"""

    lines = open(filename, 'r').readlines()
    if band==str(1):
        index = [idx for idx, s in enumerate(lines) if 'VRTRasterBand dataType="Int16" band="1"' in s][0]
    else:
        phrase = 'VRTRasterBand dataType="Int16" band="'+band+'"'
        index = [idx for idx, s in enumerate(lines) if phrase in s][0]
    lines[index] = header
    lines.insert(index+1, contents)
    open(filename, 'w').write("".join(lines))
    
    return None


def retile(yr, convert_list, uids):
    yrfold = str(yr)+'/'
    if not os.path.exists(outdir+yrfold):
        os.mkdir(outdir+yrfold)
        
    # Open the target raster (the one with the grid you want to match)
    with rasterio.open('/projectnb/modislc/projects/above/tiles/above_regions/Region_ID.Bh05v00.tif') as tgt:
        target_profile = tgt.profile  # Get the profile of the target raster
    
    target_profile.update(dtype='int16',nodata=32767,count=6)
    #print(target_profile)
    
    # For each UID, iterate through each HLS tile
    for tile in uids: # CHANGE ONCE READY TO RUN ON ALL!
        hls_tiles = convert_list.loc[convert_list['UID']==tile]
        
        # Create dir if doesn't exist
        tilefold = 'Bh'+str(hls_tiles['Ahh'].unique()[0])+str(hls_tiles['Bh'].unique()[0])+'v'+str(hls_tiles['Avv'].unique()[0])+str(hls_tiles['Bv'].unique()[0])
        print(tilefold)
        if not os.path.exists(outdir+yrfold+tilefold):
            os.mkdir(outdir+yrfold+tilefold)
        
        for idx, hls_tile in hls_tiles.iterrows():
            target_profile.update(transform=Affine(30,0,hls_tile['x'],0,-30,hls_tile['y']))
            #print(target_profile)
            print(hls_tile['Name'])
            img_list = glob.glob(outdir+hls_tile['Name']+'/phenoMetrics/MSLSP_'+hls_tile['Name']+'_'+str(yr)+'**.tif')
            #print(img_list)
            for img in img_list:
                phenometric = os.path.basename(img.split("_")[4])
                print(phenometric)
                
                # Open the source raster
                with rasterio.open(img) as src:
                    source_data = src.read()  # Read the data from the first band (adjust as needed)
                
                #source_data = source_data.astype(float)
                #source_data[source_data==32767] = np.nan
                
                # Create a new raster file for the reprojected data
                with rasterio.open(outdir+yrfold+tilefold+'/'+tilefold+'_'+str(yr)+'_'+hls_tile['Name']+'_'+phenometric, 'w', **target_profile) as dst:
                    for band_idx in range(src.count):
                        reproject(
                            source=source_data[band_idx],
                            src_nodata=32767,
                            destination=rasterio.band(dst, band_idx+1),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=target_profile['transform'],
                            dst_crs=target_profile['crs'],
                            resampling=rasterio.enums.Resampling.nearest  # Adjust resampling method if needed
                        )
        
        for pmetric in phenometrics:
            # Get list of tifs and set out name
            out_file = outdir+yrfold+tilefold+'/'+tilefold+'_'+str(yr)+'_'+pmetric
            tifs = glob.glob(outdir+yrfold+tilefold+'/*'+pmetric+'.tif')
            # Create final tifs for this phenometric
            vrt_file = gdal.BuildVRT(f'{out_file}.vrt', tifs, options=gdal.BuildVRTOptions(VRTNodata=32767))
            vrt_file.FlushCache()
            # For each band, add overlap-averaging pixel function 
            for n in [1,2,3,4,5,6]:
                add_pixel_fn(f'{out_file}.vrt',str(n))
            # Create averaged tif
            gdal.SetConfigOption('GDAL_VRT_ENABLE_PYTHON', 'YES')
            translate_file = gdal.Translate(f'{out_file}_final.tif',f'{out_file}.vrt')
            print('  Completed merging to ABoVE tile.')
            
            # Delete input tifs for this phenometric
            for f in glob.glob(outdir+yrfold+tilefold+'/*'+pmetric+'.tif'):
                os.remove(f)
            print('  Removed input tifs.')
                
    return None


def retile_topo(convert_list, uids):       
    # Open the target raster (the one with the grid you want to match)
    with rasterio.open('/projectnb/modislc/projects/above/tiles/above_regions/Region_ID.Bh05v00.tif') as tgt:
        target_profile = tgt.profile  # Get the profile of the target raster
    
    # For each UID, iterate through each HLS tile
    for tile in uids: 
        hls_tiles = convert_list.loc[convert_list['UID']==tile]
        
        # Create dir if doesn't exist
        tilefold = 'Bh'+str(hls_tiles['Ahh'].unique()[0])+str(hls_tiles['Bh'].unique()[0])+'v'+str(hls_tiles['Avv'].unique()[0])+str(hls_tiles['Bv'].unique()[0])
        print(tilefold)
        if not os.path.exists(outdir+'2016/'+tilefold):
            os.mkdir(outdir+'2016/'+tilefold)
        
        for idx, hls_tile in hls_tiles.iterrows():
            target_profile.update(transform=Affine(30,0,hls_tile['x'],0,-30,hls_tile['y']))
            #print(target_profile)
            print(hls_tile['Name'])
            img_list = glob.glob(indir+hls_tile['Name']+'/images/'+'**.tif')
            #print(img_list)
            for img in img_list:
                metric = os.path.basename(img.split("_")[1])
                print(metric)
                
                if ((metric=='dem') | (metric=='water')):
                    ndat = -32768
                    target_profile.update(dtype='int16',nodata=ndat,count=1)
                if ((metric=='slope') | (metric=='aspect')):
                    ndat = 65535
                    target_profile.update(dtype='uint16',nodata=ndat,count=1)
                
                # Open the source raster
                with rasterio.open(img) as src:
                    source_data = src.read()  # Read the data from the first band (adjust as needed)
                
                # Create a new raster file for the reprojected data
                with rasterio.open(outdir+'2016/'+tilefold+'/'+tilefold+'_'+hls_tile['Name']+'_'+metric+'.tif', 'w', **target_profile) as dst:
                    for band_idx in range(src.count):
                        reproject(
                            source=source_data[band_idx],
                            src_nodata=ndat,
                            destination=rasterio.band(dst, band_idx+1),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=target_profile['transform'],
                            dst_crs=target_profile['crs'],
                            resampling=rasterio.enums.Resampling.nearest  # Adjust resampling method if needed
                        )
        
        for met in metrics:
            if ((met=='dem') | (met=='water')):
                ndat = -32768
            if ((met=='slope') | (met=='aspect')):
                ndat = 65535
                
            # Get list of tifs and set out name
            out_file = outdir+'2016/'+tilefold+'/'+tilefold+'_'+met
            tifs = glob.glob(outdir+'2016/'+tilefold+'/*'+met+'.tif')
            # Create final tifs for this phenometric
            vrt_file = gdal.BuildVRT(f'{out_file}.vrt', tifs, options=gdal.BuildVRTOptions(VRTNodata=ndat))
            vrt_file.FlushCache()
            # Create merged tif (not averaging overlaps)
            translate_file = gdal.Translate(f'{out_file}_final.tif',f'{out_file}.vrt')
            print('  Completed merging to ABoVE tile.')
            
            # Delete input tifs for this phenometric
            for f in glob.glob(outdir+'2016/'+tilefold+'/*'+met+'.tif'):
                os.remove(f)
            print('  Removed input tifs.')

    return None


# Retile imagery
# Get UID list
convert_list = pd.read_csv('/projectnb/modislc/users/seamorez/HLS_FCover/HLS_tiles/AKTundra/Tuk_convert.csv')
uids = convert_list['UID'].unique()

# Retile spectral features to ABoVE grid for all years
for year in years:
    print(year)
    retile(year, convert_list, uids)

# Retile topographic features to ABoVE grid and save in 2016 directory
retile_topo(convert_list, uids)


#%%
################################### STEP 3 ###################################
# Remove HLS tile input tifs (ONLY DO THIS STEP AFTER CHECKING ALL RETILED RESULTS)

# For each HLS tile
for subdir in lines:
    print(subdir)
    # For each year
    for year in years:
        file = outdir+subdir+'/phenoMetrics/MSLSP_'+subdir+'_'+str(year)+'_'+metric+'.tif'
        if (os.path.isfile(file)):
            os.remove(file)
    print('  Removed input tifs.')

