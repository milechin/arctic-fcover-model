import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skimage import data
from skimage.registration import phase_cross_correlation
from skimage.registration._phase_cross_correlation import _upsampled_dft
from scipy.ndimage import fourier_shift

import glob
import os

import rasterio
from rasterio import Affine
from rasterio.warp import reproject
from osgeo import gdal, osr


#%%
################################## FUNCTIONS ##################################
def mask_ndata(map_path, img_path):
    with rasterio.open(map_path, "r+") as dataset:
        band1 = dataset.read(1)  # Read only Band 1
        
        with rasterio.open(img_path) as img:
            img_b = img.read(1)
            
        band1[img_b==img.nodata] = dataset.nodata  # Modify pixel values
                
        dataset.write(band1, 1)  # Write back to Band 1

def reproj_Maxar_ABoVE(maxar_path, tmp_path, out_path, res=3):
    '''
    Reproject Maxar imagery from WGS 84 to the ABoVE grid
    
    Parameters
    ----------
    maxar_path : for the original Maxar image
    temp_path : for the 3 m Maxar image before correction
    out_path : for the output 3 m Maxar image in the ABoVE grid

    Returns
    -------
    None.
    '''
    
    with rasterio.open(maxar_path) as src:
        orig_img = src.read()
        if res==2:
            print('2 m')
            # Reproject and resample to 2 m
            # First iteration: This one is slightly incorrect, hence the following readjustment
            # Set the GDAL configuration options
            options = gdal.WarpOptions(format='GTiff',xRes=2,yRes=2,targetAlignedPixels=True,resampleAlg='near',dstSRS='ESRI:102001')
            # Perform the warp operation
            gdal.Warp(out_path, maxar_path, options=options)
        
        elif res==3:
            print('3 m')
            # Reproject and resample to 3 m
            # First iteration: This one is slightly incorrect, hence the following readjustment
            # Set the GDAL configuration options
            options = gdal.WarpOptions(format='GTiff',xRes=3,yRes=3,targetAlignedPixels=True,resampleAlg='near',dstSRS='ESRI:102001')
            # Perform the warp operation
            gdal.Warp(tmp_path, maxar_path, options=options)
        
            # Readjust to correct for the projection error for the ABoVE grid
            with rasterio.open(tmp_path) as proj_error:
                fix_profile = proj_error.profile
                fix_profile.update(height=fix_profile['height']+1, transform=Affine(3,0,fix_profile['transform'].c,0,-3,fix_profile['transform'].f+2))
                print(fix_profile)
                # Write out the reprojected 3 m Maxar image
                with rasterio.open(out_path, 'w', **fix_profile) as dst_3:
                    for i in range(src.count):
                        reproject(
                            source=orig_img[i],
                            src_nodata=0,
                            destination=rasterio.band(dst_3, i+1),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_nodata=0,
                            dst_transform=fix_profile['transform'],
                            dst_crs=fix_profile['crs'],
                            resampling=rasterio.enums.Resampling.nearest)  # Can try bilinear here as well!
        
            os.remove(tmp_path)
    
    return None

def separate_classes(maxar_path, out_path, bands=4):
    '''
    Separate each lc class into a binary band to prepare for fractional mapping
    
    Parameters
    ----------
    maxar_path : 3 m map path
    out_path : 3 m binary map path

    Returns
    -------
    None.
    '''
    
    with rasterio.open(maxar_path) as src:
        lc_map = src.read()
        meta = src.meta; meta.update(dtype='int16', count=5, nodata=32767)  # 5 classes
        nsamples, nx, ny = lc_map.shape
        bin_map = np.full((5,nx,ny), 32767)                             # binary bands setup
        
        # For each class, separate into binary bands
        for i in range(1,bands+1):
            lc_idx = np.argwhere(lc_map[0,:,:]==i)
            nlc_idx = np.argwhere((lc_map[0,:,:]!=i) & (lc_map[0,:,:]!=0))
            bin_map[i-1,lc_idx[:,0],lc_idx[:,1]] = 1
            bin_map[i-1,nlc_idx[:,0],nlc_idx[:,1]] = 0
        # If no trees, make sure trees band is all 0% cover
        if bands==4:
            lc_idx = np.argwhere(lc_map[0,:,:]!=0)
            bin_map[4,lc_idx[:,0],lc_idx[:,1]] = 0
        
    # Output the binary map
    with rasterio.open(out_path, 'w', **meta) as dst:
        dst.write(bin_map)
    
    return None
    
def coregister_and_scale(maxar_path, hls_path, out_path, shft):
    with rasterio.open(hls_path) as trgt:
        profile = trgt.profile
    
    with rasterio.open(maxar_path) as src:
        lc_map = src.read()
        lc_map = lc_map.astype(float)
        
    # Shift the image
    shift = (np.round(shft[0]).astype(int), np.round(shft[1]).astype(int))
    img_shifted = np.roll(lc_map, shift, axis=(1, 2))
    img_shifted[lc_map==32767] = 32767
    img_shifted[img_shifted==32767] = -32767  # Making a large negative to identify incorrectly-averaged edge pixels
    print(img_shifted[img_shifted==0].shape)
    
    profile.update(count=img_shifted.shape[0], dtype='float32')
    
    # Reproject and resample to 30 m
    #target_profile.update(width=hls_clipped.shape[2], height=hls_clipped.shape[1], transform=Affine(30,0,np.round(bound_src.transform.c),0,-30,np.round(bound_src.transform.f)))
    with rasterio.open(out_path, 'w', **profile) as dst:
        for i in range(src.count):
            reproject(
                source=img_shifted[i],
                src_nodata=32767,
                destination=rasterio.band(dst, i+1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_nodata=32767,
                dst_transform=profile['transform'],
                dst_crs=profile['crs'],
                resampling=rasterio.enums.Resampling.average)  # This averaging has been demonstrated to be accurate in this case!

    with rasterio.open(out_path) as src2:
        shifted_map = src2.read()
        meta = src2.meta
    
    shifted_map[shifted_map<0] = 32767
    
    with rasterio.open(out_path, 'w', **meta) as dst2:
        dst2.write(shifted_map)

    return None


#%%
################################## EXECUTION ##################################
# Inputs to be passed in #
base_dir = '/projectnb/modislc/users/seamorez/HLS_FCover/Maxar_NAtundra/training/'

# Get paths from coreg_info.csv
coreg_list = pd.read_csv('/projectnb/modislc/users/seamorez/HLS_FCover/Maxar_NAtundra/coreg_info/coreg_info.csv')
#coreg_list = coreg_list.iloc[[0,1,2,3,4,5,6,7,11,12,13,17,18]] # REMOVE THIS ONCE READY TO RUN ON ALL!
coreg_list = coreg_list.iloc[[17,18]]

# Coregister by ecoregion
for ecoregion in coreg_list['L3_Code'].unique():
    # Setup
    coreg_l = coreg_list.loc[coreg_list['L3_Code']==ecoregion]
    working_dir = base_dir+ecoregion
    # Coregister one site at a time
    for i, row in coreg_l.iterrows():
        orig_maxar_path = row['Maxar_map']
        hls_path = row['HLS_path']
        
        ### STEP 0 ###
        # Make sure the no data areas of the map are masked out!
        mask_ndata(working_dir+orig_maxar_path, working_dir+'/'+row['Maxar_MS'])

        #### STEP 1 ####
        # Reproject the Maxar image to fit the ABoVE grid 
        tmp_path_3m = working_dir+orig_maxar_path[:-4]+'_3mtmp.tif'
        out_path_3m = working_dir+orig_maxar_path[:-4]+'_3m.tif'
        reproj_Maxar_ABoVE(working_dir+orig_maxar_path, tmp_path_3m, out_path_3m, 2)
        
        #### STEP 2 ####
        # Separate classes into individual bands
        if row['Trees']==1: bands = 5
        else: bands = 4
        out_path_bin = out_path_3m[:-4]+'_bin.tif'
        separate_classes(out_path_3m, out_path_bin, bands)
        
        #### STEP 3 ####
        # Coregister and scale imagery to 30 m
        out_path = working_dir+orig_maxar_path[:-4]+'_30m.tif'
        
        # Read in the text file with the shift information
        with open(working_dir+orig_maxar_path[:-4]+'_shift.txt', 'r') as txt_file:
            lines = txt_file.readlines()
            cleaned_lines = [line.strip() for line in lines]
            print(cleaned_lines)
            cleaned_lines = [float(i) for i in cleaned_lines]
                
        maxar_map = coregister_and_scale(out_path_bin, hls_path[:-4]+'_clipped.tif', 
                                         out_path, [cleaned_lines[0],cleaned_lines[1]])
        
        