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

def fix_edges(in_path, out_path, hls_path, hls_out_path):
    '''
    This function corrects an error with GDAL where no data values are included
    in a reprojection + upsampling task. It also clips the HLS image to match 
    the Maxar extent.
    
    Parameters
    ----------
    in_path : for the image with negative edge and no data values
    out_path : for the output image that is corrected
    hls_path : for the HLS image to be clipped
    hls_out_path : for the output HLS image clipped to the Maxar image extent

    Returns
    -------
    img, hls_img
    '''
    # Readjust the negative values to be no-data
    with rasterio.open(in_path) as dst_30:
        img_1 = dst_30.read(1)
        img_1[img_1<0] = 32767
    
        nodata = dst_30.nodata
    
        # Identify the valid data extent
        valid_y, valid_x = np.where(img_1 != nodata)
        min_x, max_x = valid_x.min(), valid_x.max()
        min_y, max_y = valid_y.min(), valid_y.max()
    
        # Create a window to crop to the valid extent
        window = rasterio.windows.Window.from_slices((min_y, max_y+1), (min_x, max_x+1))
        img = dst_30.read(window=window, boundless=True)
        img[img<0] = 32767
    
        # Update the profile for the output raster
        profile = dst_30.profile
        profile.update({
            'height': max_y - min_y + 1,
            'width': max_x - min_x + 1,
            'transform': rasterio.windows.transform(window, dst_30.transform),
            'nodata': nodata})
    
        # Create the output raster
        with rasterio.open(out_path, 'w', **profile) as dest:
            # Copy valid data to the output raster
            dest.write(img)
            
        # Clip the HLS image to the 30 m Maxar image's bounds and save out
        with rasterio.open(hls_path) as hls:
            hls_img = hls.read(window=window, boundless=True)
            print(np.where(img[0,:]==32767))
            hls_img[:, np.where(img[0,:]==32767)[0], np.where(img[0,:]==32767)[1]] = 32767
    
            profile2 = profile
            profile2.update(count=hls_img.shape[0])
            with rasterio.open(hls_out_path, 'w', **profile2) as hls_dst:
                hls_dst.write(hls_img)
    
    return img, hls_img, profile

def upscale_Maxar(hls_path, maxar_path, tmp_path, hls_out_path, maxar_out_path):
    '''
    This takes the 3 m Maxar image and reprojects it to 30 m and clips the HLS 
    image to match the extent of the Maxar image.
    
    Parameters
    ----------
    hls_path : for the HLS image gridded to the ABoVE grid that covers the Maxar
    image extent
    maxar_path : for the 3 m Maxar image
    tmp_path : for the first iteration of the 30 m Maxar image
    hls_out_path : for the output HLS image clipped to the Maxar image extent
    maxar_out_path : for the upscaled 30 m Maxar image

    Returns
    -------
    maxar_img, hls_img
    '''
    # Read in HLS image to get transform info
    with rasterio.open(hls_path) as sample:
        target_profile = sample.profile  # Get the profile of the target raster
    
        # Read in the reprojected 3 m Maxar image
        with rasterio.open(maxar_path) as bound_src:
            bound_img = bound_src.read()
            bound_img = bound_img.astype('int16')
            bound_img[bound_img==0] = -32767  # Making a large negative to identify incorrectly-averaged edge pixels
            target_profile.update(count=bound_img.shape[0])
    
            # Reproject and resample to 30 m
            with rasterio.open(tmp_path, 'w', **target_profile) as dst:
                for i in range(bound_src.count):
                    reproject(
                        source=bound_img[i],
                        src_nodata=0,
                        destination=rasterio.band(dst, i+1),
                        src_transform=bound_src.transform,
                        src_crs=bound_src.crs,
                        dst_nodata=32767,
                        dst_transform=target_profile['transform'],
                        dst_crs=target_profile['crs'],
                        resampling=rasterio.enums.Resampling.average)  # This averaging has been demonstrated to be accurate in this case!
    
    # Adjust the negative values to no data values and clip the HLS image
    maxar_img, hls_img, profile = fix_edges(tmp_path, maxar_out_path, hls_path, hls_out_path)
    
    os.remove(tmp_path)
    
    return maxar_img, hls_img, profile

def find_shift(hls_img, maxar_img):
    '''
    Use phase cross-correlation to identify the sub-pixel shift between the 
    images; adapted from https://scikit-image.org/docs/stable/auto_examples/registration/plot_register_translation.html 
    (see DOI:10.1364/OL.33.000156)

    Parameters
    ----------
    hls_img : the clipped HLS image
    maxar_img : the 30 m Maxar image

    Returns
    -------
    shift : the y and x shifts detected between the input images
    '''
    mask1 = np.zeros_like(hls_img, dtype=bool)
    mask2 = np.zeros_like(maxar_img, dtype=bool)
    mask1[hls_img!=32767] = True
    mask2[maxar_img!=32767] = True
    hls_img *= mask1
    maxar_img *= mask2
    
    shift, error, diffphase = phase_cross_correlation(hls_img, maxar_img)

    fig = plt.figure(figsize=(8, 3))
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
    ax3 = plt.subplot(1, 3, 3)
    
    ax1.imshow(hls_img, cmap='gray')
    ax1.set_axis_off()
    ax1.set_title('Reference image')
    
    ax2.imshow(maxar_img.real, cmap='gray')
    ax2.set_axis_off()
    ax2.set_title('Offset image')
    
    # Show the output of a cross-correlation to show what the algorithm is
    # doing behind the scenes
    image_product = np.fft.fft2(hls_img) * np.fft.fft2(maxar_img).conj()
    cc_image = np.fft.fftshift(np.fft.ifft2(image_product))
    ax3.imshow(cc_image.real)
    ax3.set_axis_off()
    ax3.set_title("Cross-correlation")
    
    plt.show()
    
    print(f'Detected pixel offset (y, x): {shift}')
    
    # subpixel precision
    shift, error, diffphase = phase_cross_correlation(hls_img, maxar_img, upsample_factor=100)
    
    fig = plt.figure(figsize=(8, 3))
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
    ax3 = plt.subplot(1, 3, 3)
    
    ax1.imshow(hls_img, cmap='gray')
    ax1.set_axis_off()
    ax1.set_title('Reference image')
    
    ax2.imshow(maxar_img.real, cmap='gray')
    ax2.set_axis_off()
    ax2.set_title('Offset image')
    
    # Calculate the upsampled DFT, again to show what the algorithm is doing
    # behind the scenes.  Constants correspond to calculated values in routine.
    # See source code for details.
    cc_image = _upsampled_dft(image_product, 150, 100, (shift*100)+75).conj()
    ax3.imshow(cc_image.real)
    ax3.set_axis_off()
    ax3.set_title("Supersampled XC sub-area")
    
    
    plt.show()
    
    print(f'Detected subpixel offset (y, x): {shift}')

    return shift

def coreg_image(shft, profile, maxar_path, out_path, up_path):
    '''
    Coregister the Maxar image to the HLS image by the detected shift

    Parameters
    ----------
    shft : the detected shift
    maxar_path : for the 3 m Maxar image
    out_path : for the coregistered 3 m Maxar image
    up_path : for the coregistered 30 m Maxar image (useful for visual 
    assessment of performance)

    Returns
    -------
    None.
    '''
    # Read in the reprojected 3 m Maxar image
    with rasterio.open(maxar_path) as bound_src:
        meta = bound_src.meta
        bound_img = bound_src.read()
        bound_img = bound_img.astype('int16')
    
        # Shift the image
        shift = (np.round(shft[0]).astype(int), np.round(shft[1]).astype(int))
        img_shifted = np.roll(bound_img, shift, axis=(1, 2))
        img_shifted[bound_img==0] = 0
        
        with rasterio.open(out_path, 'w', **meta) as dst:
            dst.write(img_shifted)
            
    img_shifted[img_shifted==0] = -32767  # Making a large negative to identify incorrectly-averaged edge pixels
    print(img_shifted[img_shifted==0].shape)
    
    profile.update(count=img_shifted.shape[0])
    
    # Reproject and resample to 30 m
    #target_profile.update(width=hls_clipped.shape[2], height=hls_clipped.shape[1], transform=Affine(30,0,np.round(bound_src.transform.c),0,-30,np.round(bound_src.transform.f)))
    with rasterio.open(up_path, 'w', **profile) as dst:
        for i in range(bound_src.count):
            reproject(
                source=img_shifted[i],
                src_nodata=0,
                destination=rasterio.band(dst, i+1),
                src_transform=bound_src.transform,
                src_crs=bound_src.crs,
                dst_nodata=32767,
                dst_transform=profile['transform'],
                dst_crs=profile['crs'],
                resampling=rasterio.enums.Resampling.average)  # This averaging has been demonstrated to be accurate in this case!
    
    return None


#%%
################################## EXECUTION ##################################
# Inputs to be passed in #
base_dir = '/projectnb/modislc/users/seamorez/HLS_FCover/Maxar_NAtundra/training/'
#base_dir = '/projectnb/modislc/users/seamorez/HLS_FCover/Maxar_NAtundra/training/'

# Get paths from coreg_info.csv
coreg_list = pd.read_csv('/projectnb/modislc/users/seamorez/HLS_FCover/Maxar_NAtundra/coreg_info/coreg_info.csv')
coreg_list = coreg_list.iloc[[0,1,2,3,4,5,6,7,11,12,13,17,18,19,20,21,22]] # REMOVE THIS ONCE READY TO RUN ON ALL!

# Coregister by ecoregion
for ecoregion in coreg_list['L3_Code'].unique():
    # Setup
    coreg_l = coreg_list.loc[coreg_list['L3_Code']==ecoregion]
    working_dir = base_dir+ecoregion+'/'
    # Coregister one site at a time
    for i, row in coreg_l.iterrows():
        orig_maxar_path = row['Maxar_MS']
        hls_path = row['HLS_path']

        #### STEP 1 ####
        # Reproject the Maxar image to fit the ABoVE grid 
        tmp_path_3m = working_dir+orig_maxar_path[:-4]+'_3mtmp.tif'
        out_path_3m = working_dir+orig_maxar_path[:-4]+'_3m.tif'
        reproj_Maxar_ABoVE(working_dir+orig_maxar_path, tmp_path_3m, out_path_3m, 2)
        
        #### STEP 2 ####
        # Resample the Maxar image to 30 m and clip the HLS image
        tmp_path_30m = working_dir+orig_maxar_path[:-4]+'_30mtmp.tif'
        hls_out_path = hls_path[:-4]+'_clipped.tif'
        out_path_30m = working_dir+orig_maxar_path[:-4]+'_30m.tif'
        maxar_img, hls_img, profile = upscale_Maxar(hls_path, out_path_3m, tmp_path_30m, hls_out_path, out_path_30m)
        
        if orig_maxar_path[0:2]=='WV':
            shift = find_shift(hls_img[13], maxar_img[6])
        else:
            shift = find_shift(hls_img[13], maxar_img[3])
        
        #### STEP 3 ####
        # Coregister the Maxar image
        out_path = working_dir+orig_maxar_path[:-4]+'_3m_shift.tif'
        up_path = working_dir+orig_maxar_path[:-4]+'_30m_shift.tif'
        coreg_image(shift*15, profile, out_path_3m, out_path, up_path)
        
        #### STEP 4 ####
        # Save out the shift to use on the fractional map!
        print(working_dir+row['Maxar_map'][:-4]+'_shift.txt')
        np.savetxt(working_dir+row['Maxar_map'][:-4]+'_shift.txt', np.round(shift*15).astype(int), delimiter=',')
        