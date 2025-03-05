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
def create_mask(qa_rast, band1, band2, threshold, tree=False, shadow=False):
    # 1) Get differences
    difference = np.abs(qa_rast[band2,:] - qa_rast[band1,:])
    copy = np.copy(difference)
    # 2) Mask by QA rules
    if (tree==False) & (shadow==False):
        difference[copy<=threshold] = 1
    else:
        difference[copy<=threshold] = -32767  # if masking for water/shadow/tree confusion
    difference[copy>threshold] = 0
    # 3) Remove mask from pixels where selected classes not 1st and 2nd likeliest classes
    max1_class = np.argmax(qa_rast, axis=0)
    max2_class = np.argsort(qa_rast, axis=0)[-2]
    common_values1 = np.intersect1d(np.argwhere(max1_class!=band1), np.argwhere(max1_class!=band2))
    common_values2 = np.intersect1d(np.argwhere(max2_class!=band2), np.argwhere(max2_class!=band1))
    print(common_values1.shape, common_values2.shape)
    difference[common_values1] = 0
    difference[common_values2] = 0
        
    return difference

def apply_qa(map_path, qa_path, tree):
    with rasterio.open(map_path) as src:
        qa_bands = src.read(); qa_bands = qa_bands[1:,:,:]  # Get QA bands
        meta = src.meta
        meta.update(count=1, nodata=32767, dtype='int16')
        
    nclasses, nx, ny = qa_bands.shape
    qa_flat = qa_bands.reshape(nclasses, nx*ny)
    print(qa_flat.shape)
    
    if tree==False:
        if (nclasses==5) | (nclasses==6):
            print('No trees, yes clouds/cloud shadows')
            nonwoody_shrub = create_mask(qa_flat, 3, 4, 0.1)
            barren_nonwoody = create_mask(qa_flat, 2, 3, 0.1)
            water_shrub = create_mask(qa_flat, 1, 4, 0.3, shadow=True)
            shadows_water = create_mask(qa_flat, 0, 1, 0.3, shadow=True)
            # Combine arrays
            qa_mask = np.logical_or(nonwoody_shrub, barren_nonwoody)
            qa_mask = qa_mask.astype(int)
            indices0 = np.where(water_shrub==-32767)
            qa_mask[indices0] = -32767
            indices1 = np.where(shadows_water==-32767)
            qa_mask[indices1] = -32767
            print('Negative values:', qa_mask[qa_mask==-32767].shape)
            
        else:
            nonwoody_shrub = create_mask(qa_flat, 2, 3, 0.1)
            barren_nonwoody = create_mask(qa_flat, 1, 2, 0.1)
            water_shrub = create_mask(qa_flat, 0, 3, 0.3, shadow=True)
            # Combine arrays
            qa_mask = np.logical_or(nonwoody_shrub, barren_nonwoody)
            qa_mask = qa_mask.astype(int)
            indices0 = np.where(water_shrub==-32767)
            qa_mask[indices0] = -32767
            
        plt.imshow(nonwoody_shrub.reshape(nx,ny))
        plt.show()
    
    # Combine arrays
    #qa_mask = np.logical_or(nonwoody_shrub, barren_nonwoody)
        
    elif tree==True:
        if nclasses==6:
            nonwoody_shrub = create_mask(qa_flat, 3, 4, 0.1)
            barren_nonwoody = create_mask(qa_flat, 2, 3, 0.1)
            water_shrub = create_mask(qa_flat, 1, 4, 0.3, shadow=True)
            water_trees = create_mask(qa_flat, 1, 5, 0.3, tree=True)
            shadows_water = create_mask(qa_flat, 0, 1, 0.3, shadow=True)
            # Combine arrays
            qa_mask = np.logical_or(nonwoody_shrub, barren_nonwoody)
            qa_mask = qa_mask.astype(int)
            indices0 = np.where(water_shrub==-32767)
            qa_mask[indices0] = -32767
            indices1 = np.where(water_trees==-32767)
            qa_mask[indices1] = -32767
            indices2 = np.where(shadows_water==-32767)
            qa_mask[indices2] = -32767
            
        else:
            nonwoody_shrub = create_mask(qa_flat, 2, 3, 0.1)
            barren_nonwoody = create_mask(qa_flat, 1, 2, 0.1)
            water_shrub = create_mask(qa_flat, 0, 3, 0.3, shadow=True)
            water_trees = create_mask(qa_flat, 0, 4, 0.3, tree=True)
            # Combine arrays
            qa_mask = np.logical_or(nonwoody_shrub, barren_nonwoody)
            qa_mask = qa_mask.astype(int)
            indices0 = np.where(water_shrub==-32767)
            qa_mask[indices0] = -32767
            indices = np.where(water_trees==-32767)
            qa_mask[indices] = -32767
    
    with rasterio.open(qa_path, 'w', **meta) as dst:
        dst.write(qa_mask.reshape(1,nx,ny))
    
    return qa_mask

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
        meta.update(dtype='int16', nodata=0)
    
    shifted_map[shifted_map<0] = 1
    # If more than 30% of pixels in each 30 m pixel is masked, mask the 30 m pixel
    shifted_map[shifted_map>=0.3] = 1
    shifted_map[(shifted_map<0.3) & (shifted_map>=0)] = 0
    print(shifted_map[shifted_map==1])
    
    with rasterio.open(out_path, 'w', **meta) as dst2:
        dst2.write(shifted_map)
        print('overwrote?')

    return None

def mask_30m_map(map_path, qa_path, out_path):
    with rasterio.open(map_path) as src:
        map = src.read()
        meta = src.meta
    with rasterio.open(qa_path) as msrc:
        mask = msrc.read()

    indices = np.where(mask==1)
    map[:, indices[1], indices[2]] = 32767

    with rasterio.open(out_path, 'w', **meta) as dst:
        dst.write(map)
        
    return None


#%%
################################## EXECUTION ##################################
# Inputs to be passed in #
base_dir = '/projectnb/modislc/users/seamorez/HLS_FCover/Maxar_NAtundra/training/'

# Apply QA rules and resample to ABoVE grid
# Get paths from coreg_info.csv
coreg_list = pd.read_csv('/projectnb/modislc/users/seamorez/HLS_FCover/Maxar_NAtundra/coreg_info/coreg_info.csv')
coreg_list = coreg_list.iloc[[0,11,12,13,17,18]] # REMOVE THIS ONCE READY TO RUN ON ALL!

# Apply QA rules by ecoregion
for ecoregion in coreg_list['L3_Code'].unique():
    # Setup
    coreg_l = coreg_list.loc[coreg_list['L3_Code']==ecoregion]
    working_dir = base_dir+ecoregion
    #### STEP 1 ####
    # Apply QA rules one site at a time
    for i, row in coreg_l.iterrows():
        orig_maxar_path = row['Maxar_map']
        print(orig_maxar_path)
        hls_path = row['HLS_path']
        map_path = working_dir+orig_maxar_path[:-4]+'_3m.tif'
        qa_3m_path = working_dir+orig_maxar_path[:-4]+'_3m_QA.tif'
        
        # If trees true, we need to apply a QA rule for trees
        trees = (row['Trees']==1)
        
        qa_mask_out = apply_qa(map_path, qa_3m_path, trees)

        #### STEP 2 ####
        # Coregister and scale imagery to 30 m
        out_path = working_dir+orig_maxar_path[:-4]+'_30m_QA.tif'
        
        # Read in the text file with the shift information
        with open(working_dir+orig_maxar_path[:-4]+'_shift.txt', 'r') as txt_file:
            lines = txt_file.readlines()
            cleaned_lines = [line.strip() for line in lines]
            print(cleaned_lines)
            cleaned_lines = [float(i) for i in cleaned_lines]
                
        maxar_map = coregister_and_scale(qa_3m_path, hls_path[:-4]+'_clipped.tif', 
                                         out_path, [cleaned_lines[0],cleaned_lines[1]])

# Mask 30 m maps and output final maps
for ecoregion in coreg_list['L3_Code'].unique():
    # Setup
    coreg_l = coreg_list.loc[coreg_list['L3_Code']==ecoregion]
    working_dir = base_dir+ecoregion
    #### STEP 1 ####
    # Apply QA rules one site at a time
    for i, row in coreg_l.iterrows():
        orig_maxar_path = row['Maxar_map']
        
        map_path_30m = working_dir+orig_maxar_path[:-4]+'_30m.tif'
        qa_path = working_dir+orig_maxar_path[:-4]+'_30m_QA.tif'
        final_map_path = working_dir+orig_maxar_path[:-4]+'_30m_train.tif'
        
        mask_30m_map(map_path_30m, qa_path, final_map_path)