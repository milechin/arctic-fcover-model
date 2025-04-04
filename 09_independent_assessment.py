import os
import glob
import json
import numpy as np
import pandas as pd
import rasterio as rio
from rasterio import Affine
from rasterio.warp import reproject
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from scipy.stats import gaussian_kde
import pickle
import matplotlib.pyplot as plt
import seaborn as sb


#%%
#### PARAMETERS TO BE PASSED IN ####
outdir = '/projectnb/modislc/users/seamorez/HLS_FCover/output/'
indir = '/projectnb/modislc/users/seamorez/HLS_FCover/input/HLS30/'
traindir = '/projectnb/modislc/users/seamorez/HLS_FCover/Maxar_NAtundra/training/'
model_traindir = '/projectnb/modislc/users/seamorez/HLS_FCover/model_training/'
model_outdir = '/projectnb/modislc/users/seamorez/HLS_FCover/model_outputs/'
model_assessdir = '/projectnb/modislc/users/seamorez/HLS_FCover/model_indep_assess/'
start_yr = 2016
end_yr = 2023
phenometrics = ['50PCGI','Peak','50PCGD']
metrics = ['dem', 'water', 'slope', 'aspect']
do_s1 = True  # Bool for testing S1

#### PARAMETERS ####
years = np.arange(start_yr, end_yr+1, 1)
# assess_key = pd.read_csv(traindir+'assessment_key.csv')


#%%
def make_binary(map_info):
    map_path = map_info['path']
    print(map_path)
    with rio.open(map_path) as src:
        # Set-up
        out_path = map_info['path'][:-4]+'_bin.tif'
        meta = src.meta
        meta.update(dtype='int16', count=5, nodata=32767)
        
        # Read and check shape
        hr_map = src.read(); print(hr_map.shape)
        nsamples, nx, ny = hr_map.shape
        # Binary bands setup
        bin_map = np.full((5,nx,ny), 32767)
        # Set all original no data pixels to -32767
        mask = hr_map[0] == src.nodata
        bin_map[:,mask] = -32767
        
        # For each class, separate into binary bands
        # water
        water_ids = json.loads(map_info['water'])
        if map_info['unique_water']==0:
            for id in water_ids:
                lc_idx = np.argwhere(hr_map[0,:,:]==id)
                nlc_idx = np.argwhere((hr_map[0,:,:]!=id) & (hr_map[0,:,:]!=src.nodata))
                bin_map[0,lc_idx[:,0],lc_idx[:,1]] = 0
                bin_map[0,nlc_idx[:,0],nlc_idx[:,1]] = 0
        else:
            for id in water_ids:
                lc_idx = np.argwhere(hr_map[0,:,:]==id)
                nlc_idx = np.argwhere((hr_map[0,:,:]!=id) & (hr_map[0,:,:]!=src.nodata) & (bin_map[0,:,:]!=1))
                bin_map[0,lc_idx[:,0],lc_idx[:,1]] = 1
                bin_map[0,nlc_idx[:,0],nlc_idx[:,1]] = 0
        # barren
        barren_ids = json.loads(map_info['barren'])
        for id in barren_ids:
            lc_idx = np.argwhere(hr_map[0,:,:]==id)
            nlc_idx = np.argwhere((hr_map[0,:,:]!=id) & (hr_map[0,:,:]!=src.nodata) & (bin_map[1,:,:]!=1))
            bin_map[1,lc_idx[:,0],lc_idx[:,1]] = 1
            bin_map[1,nlc_idx[:,0],nlc_idx[:,1]] = 0
        # nonwoody
        nonwoody_ids = json.loads(map_info['nonwoody'])
        for id in nonwoody_ids:
            lc_idx = np.argwhere(hr_map[0,:,:]==id)
            nlc_idx = np.argwhere((hr_map[0,:,:]!=id) & (hr_map[0,:,:]!=src.nodata) & (bin_map[2,:,:]!=1))
            bin_map[2,lc_idx[:,0],lc_idx[:,1]] = 1
            bin_map[2,nlc_idx[:,0],nlc_idx[:,1]] = 0
        # shrub
        shrub_ids = json.loads(map_info['shrub'])
        for id in shrub_ids:
            lc_idx = np.argwhere(hr_map[0,:,:]==id)
            nlc_idx = np.argwhere((hr_map[0,:,:]!=id) & (hr_map[0,:,:]!=src.nodata) & (bin_map[3,:,:]!=1))
            bin_map[3,lc_idx[:,0],lc_idx[:,1]] = 1
            bin_map[3,nlc_idx[:,0],nlc_idx[:,1]] = 0
        # trees
        trees_ids = json.loads(map_info['trees'])
        if not trees_ids:
            nlc_idx = np.argwhere((hr_map[0,:,:]!=src.nodata))
            bin_map[4,nlc_idx[:,0],nlc_idx[:,1]] = 0
        else:
            for id in trees_ids:
                lc_idx = np.argwhere(hr_map[0,:,:]==id)
                nlc_idx = np.argwhere((hr_map[0,:,:]!=id) & (hr_map[0,:,:]!=src.nodata) & (bin_map[4,:,:]!=1))
                bin_map[4,lc_idx[:,0],lc_idx[:,1]] = 1
                bin_map[4,nlc_idx[:,0],nlc_idx[:,1]] = 0
        # shadows
        nd_ids = json.loads(map_info['nd'])
        if nd_ids:
            for id in nd_ids:
                lc_idx = np.argwhere(hr_map[0,:,:]==id)
                bin_map[:,lc_idx[:,0],lc_idx[:,1]] = 32767
        
    # Output the binary map
    with rio.open(out_path, 'w', **meta) as dst:
        dst.write(bin_map)
    
    return out_path

def make_fractions(bin_path, hls_path, out_path):
    with rio.open(hls_path) as trgt:
        profile = trgt.profile
    
    with rio.open(bin_path) as src:
        lc_map = src.read()
        lc_map = lc_map.astype(float)
        
    profile.update(count=lc_map.shape[0], dtype='float32')
    
     # Reproject and resample to 30 m
    with rio.open(out_path, 'w', **profile) as dst:
        for i in range(src.count):
            # Create an empty array for resampled data
            resampled_data = np.empty((profile['height'], profile['width']), dtype='float32')
            
            reproject(
                source=lc_map[i],
                src_nodata=32767,
                destination=resampled_data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_nodata=32767,
                dst_transform=profile['transform'],
                dst_crs=profile['crs'],
                resampling=rio.enums.Resampling.average)  # This averaging has been demonstrated to be accurate in this case!
            
            # Replace negative values with 32767
            resampled_data[resampled_data < 0] = 32767
    
            # Write the processed band to the output file
            dst.write(resampled_data, i+1)

    return None

def make_plots(final_assess_pts, final_assess_pred):
    # Assessment Plots
    fig, ((ax1, ax2, ax5), (ax3, ax4, ax6)) = plt.subplots(2, 3)
    #fig.set_size_inches(34, 18)
    fig.set_size_inches(30, 18)
    #axes.set_facecolor('black')

    #compare_array = np.vstack([final_assess_pts[0], final_assess_pred[0]])
    #z = gaussian_kde(compare_array)(compare_array)
    #plt1=ax1.scatter(final_assess_pts[0], final_assess_pred[0], color='#0071dc', alpha=.7,s=20)
    plt1=ax1.hexbin(final_assess_pts[0], final_assess_pred[0], bins='log', gridsize=25, cmap='Blues')
    ax1.plot([0,100],[0,100], color='black', linewidth=.5)
    sb.regplot(y=final_assess_pred[0],x=final_assess_pts[0], scatter=False, color='dodgerblue', scatter_kws={'alpha':0.2}, ax=ax1)
    ax1.text(5, 95, '$R^2$ = '+str(np.round(np.corrcoef(final_assess_pts[0], final_assess_pred[0])[0,1]**2,2)), fontsize=20,color='black')
    ax1.text(5, 90, 'MAE = '+str(np.round(mean_absolute_error(final_assess_pts[0], final_assess_pred[0]),2))+'%', fontsize=20,color='black')
    ax1.text(5, 85, 'RMSE = '+str(np.round(np.sqrt(mean_squared_error(final_assess_pts[0], final_assess_pred[0])),2))+'%', fontsize=20,color='black')
    ax1.text(5, 80, 'Bias = '+str(np.round(-np.mean(final_assess_pts[0]-final_assess_pred[0]),2))+'%', fontsize=20,color='black')
    ax1.set_xlim(0,100)
    ax1.set_ylim(0,100)
    ax1.set_title('Water fractional cover', fontsize=32)
    ax1.set_xlabel('Observed', fontsize=24); ax1.set_ylabel('Predicted', fontsize=24)
    ax1.set_facecolor('white')
    plt.colorbar(plt1, ax=ax1)

    compare_array = np.vstack([final_assess_pts[1], final_assess_pred[1]])
    #z = gaussian_kde(compare_array)(compare_array)
    #plt2=ax2.scatter(final_assess_pts[1], final_assess_pred[1], color='#616161', alpha=.7,s=20)
    plt2=ax2.hexbin(final_assess_pts[1], final_assess_pred[1], bins='log', gridsize=25, cmap='Greys')
    ax2.plot([0,100],[0,100], color='black', linewidth=.5)
    sb.regplot(y=final_assess_pred[1],x=final_assess_pts[1], scatter=False, color='grey', scatter_kws={'alpha':0.2}, ax=ax2)
    ax2.text(5, 95, '$R^2$ = '+str(np.round(np.corrcoef(final_assess_pts[1],final_assess_pred[1])[0,1]**2,2)), fontsize=20,color='black')
    ax2.text(5, 90, 'MAE = '+str(np.round(mean_absolute_error(final_assess_pts[1],final_assess_pred[1]),2))+'%', fontsize=20,color='black')
    ax2.text(5, 85, 'RMSE = '+str(np.round(np.sqrt(mean_squared_error(final_assess_pts[1],final_assess_pred[1])),2))+'%', fontsize=20,color='black')
    ax2.text(5, 80, 'Bias = '+str(np.round(-np.mean(final_assess_pts[1]-final_assess_pred[1]), 2))+'%', fontsize=20,color='black')
    ax2.set_xlim(0,100)
    ax2.set_ylim(0,100)
    ax2.set_title('Barren fractional cover', fontsize=32)
    ax2.set_xlabel('Observed', fontsize=24); ax2.set_ylabel('Predicted', fontsize=24)
    ax2.set_facecolor('white')
    plt.colorbar(plt2, ax=ax2)

    compare_array = np.vstack([final_assess_pts[2], final_assess_pred[2]])
    #z = gaussian_kde(compare_array)(compare_array)
    #plt3=ax3.scatter(final_assess_pts[2], final_assess_pred[2], color='orange', alpha=.7,s=20)
    plt3=ax3.hexbin(final_assess_pts[2], final_assess_pred[2], bins='log', gridsize=25, cmap='YlOrBr')
    ax3.plot([0,100],[0,100], color='black', linewidth=.5)
    sb.regplot(y=final_assess_pred[2],x=final_assess_pts[2], scatter=False, color='orange', scatter_kws={'alpha':0.2}, ax=ax3)
    ax3.text(5, 95, '$R^2$ = '+str(np.round(np.corrcoef(final_assess_pts[2],final_assess_pred[2])[0,1]**2,2)), fontsize=20,color='black')
    ax3.text(5, 90, 'MAE = '+str(np.round(mean_absolute_error(final_assess_pts[2],final_assess_pred[2]),2))+'%', fontsize=20,color='black')
    ax3.text(5, 85, 'RMSE = '+str(np.round(np.sqrt(mean_squared_error(final_assess_pts[2],final_assess_pred[2])),2))+'%', fontsize=20,color='black')
    ax3.text(5, 80, 'Bias = '+str(np.round(-np.mean(final_assess_pts[2]-final_assess_pred[2]), 2))+'%', fontsize=20,color='black')
    ax3.set_xlim(0,100)
    ax3.set_ylim(0,100)
    ax3.set_title('Non-woody fractional cover', fontsize=32)
    ax3.set_xlabel('Observed', fontsize=24); ax3.set_ylabel('Predicted', fontsize=24)
    ax3.set_facecolor('white')
    plt.colorbar(plt3, ax=ax3)

    compare_array = np.vstack([final_assess_pts[3], final_assess_pred[3]])
    #z = gaussian_kde(compare_array)(compare_array)
    #plt4=ax4.scatter(final_assess_pts[3], final_assess_pred[3], color='#4d8f00', alpha=.7,s=20)
    plt4=ax4.hexbin(final_assess_pts[3], final_assess_pred[3], bins='log', gridsize=25, cmap='YlGn')
    ax4.plot([0,100],[0,100], color='black', linewidth=.5)
    sb.regplot(y=final_assess_pred[3],x=final_assess_pts[3], scatter=False, color='#4d8f00', scatter_kws={'alpha':0.2}, ax=ax4)
    ax4.text(5, 95, '$R^2$ = '+str(np.round(np.corrcoef(final_assess_pts[3],final_assess_pred[3])[0,1]**2,2)), fontsize=20,color='black')
    ax4.text(5, 90, 'MAE = '+str(np.round(mean_absolute_error(final_assess_pts[3],final_assess_pred[3]),2))+'%', fontsize=20,color='black')
    ax4.text(5, 85, 'RMSE = '+str(np.round(np.sqrt(mean_squared_error(final_assess_pts[3],final_assess_pred[3])),2))+'%', fontsize=20,color='black')
    ax4.text(5, 80, 'Bias = '+str(np.round(-np.mean(final_assess_pts[3]-final_assess_pred[3]), 2))+'%', fontsize=20,color='black')
    ax4.set_xlim(0,100)
    ax4.set_ylim(0,100)
    ax4.set_title('Shrub fractional cover', fontsize=32)
    ax4.set_xlabel('Observed', fontsize=24); ax4.set_ylabel('Predicted', fontsize=24)
    ax4.set_facecolor('white')
    plt.colorbar(plt4, ax=ax4)

    compare_array = np.vstack([final_assess_pts[4], final_assess_pred[4]])
    #z = gaussian_kde(compare_array)(compare_array)
    #plt5=ax5.scatter(final_assess_pts[4], final_assess_pred[4], color='#173114', alpha=.7,s=20)
    plt5=ax5.hexbin(final_assess_pts[4], final_assess_pred[4], bins='log', gridsize=25, cmap='PuBuGn')
    ax5.plot([0,100],[0,100], color='black', linewidth=.5)
    sb.regplot(y=final_assess_pred[4],x=final_assess_pts[4], scatter=False, color='darkgreen', scatter_kws={'alpha':0.2}, ax=ax5)
    ax5.text(5, 95, '$R^2$ = '+str(np.round(np.corrcoef(final_assess_pts[4],final_assess_pred[4])[0,1]**2,2)), fontsize=20,color='black')
    ax5.text(5, 90, 'MAE = '+str(np.round(mean_absolute_error(final_assess_pts[4],final_assess_pred[4]),2))+'%', fontsize=20,color='black')
    ax5.text(5, 85, 'RMSE = '+str(np.round(np.sqrt(mean_squared_error(final_assess_pts[4],final_assess_pred[4])),2))+'%', fontsize=20,color='black')
    ax5.text(5, 80, 'Bias = '+str(np.round(-np.mean(final_assess_pts[4]-final_assess_pred[4]), 2))+'%', fontsize=20,color='black')
    ax5.set_xlim(0,100)
    ax5.set_ylim(0,100)
    ax5.set_title('Trees fractional cover', fontsize=32)
    ax5.set_xlabel('Observed', fontsize=24); ax5.set_ylabel('Predicted', fontsize=24)
    ax5.set_facecolor('white')
    plt.colorbar(plt5, ax=ax5)
    
    plt.show()
    
def make_hists(final_assess_pts, final_assess_pred):
    # Histograms
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(8, 6)
    try:
        sb.histplot(final_assess_pts[0], kde=True, binwidth=10, binrange=(0,100), color='#0071dc', label='Observed')
    except:
        print('No water')
    try:
        sb.histplot(final_assess_pred[0], kde=True, binwidth=10, binrange=(0,100), color='#982649', label='Predicted')
    except:
        print('No water')
    plt.legend(frameon=False, fontsize=12)
    plt.title('Water', fontsize=20)
    plt.xlim(0,100)
    plt.xlabel('Cover fraction, %', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.show()

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(8, 6)
    sb.histplot(final_assess_pts[1], kde=True, binwidth=10, binrange=(0,100), color='#616161', label='Observed')
    sb.histplot(final_assess_pred[1], kde=True, binwidth=10, binrange=(0,100), color='#982649', label='Predicted')
    plt.legend(frameon=False, fontsize=12)
    plt.title('Barren', fontsize=20)
    plt.xlim(0,100)
    plt.xlabel('Cover fraction, %', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.show()

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(8, 6)
    try:
        sb.histplot(final_assess_pts[4], kde=True, binwidth=10, binrange=(0,100), color='darkgreen', label='Observed')
    except:
        print('No trees')
    try:
        sb.histplot(final_assess_pred[4], kde=True, binwidth=10, binrange=(0,100), color='#982649', label='Predicted')
    except:
        print('No trees')
    plt.legend(frameon=False, fontsize=12)
    plt.title('Trees', fontsize=20)
    plt.xlim(0,100)
    plt.xlabel('Cover fraction, %', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.show()

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(8, 6)
    sb.histplot(final_assess_pts[2], kde=True, binwidth=10, binrange=(0,100), color='orange', label='Observed')
    sb.histplot(final_assess_pred[2], kde=True, binwidth=10, binrange=(0,100), color='#982649', label='Predicted')
    plt.legend(frameon=False, fontsize=12)
    plt.title('Low vegetation', fontsize=20)
    plt.xlim(0,100)
    plt.xlabel('Cover fraction, %', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.show()

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(8, 6)
    sb.histplot(final_assess_pts[3], kde=True, binwidth=10, binrange=(0,100), color='#4d8f00', label='Observed')
    sb.histplot(final_assess_pred[3], kde=True, binwidth=10, binrange=(0,100), color='#982649', label='Predicted')
    plt.legend(frameon=False, fontsize=12)
    plt.title('Low & tall shrubs', fontsize=20)
    plt.xlim(0,100)
    plt.xlabel('Cover fraction, %', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.show()


#%%
# WARNING: DO NOT RE-RUN IF ALREADY MADE. THIS TAKES A WHILE. Read in rasters one by one
map_list = pd.read_csv(model_assessdir+'map_paths.csv')
for i, row in map_list[26:].iterrows():
    print(row['map'])
    
    # Convert to binary maps
    bin_path = make_binary(row)
    print('Done with binary map. Onto fractional map!')
        
    # Convert to fractional maps in ESRI 102001 projection
    hls_ref_path = outdir+str(row['year'])+'/'+row['hls_tile']+'/'+row['hls_tile']+'_'+str(row['year'])+'_Peak_final.tif'
    if not os.path.exists(hls_ref_path):
        print('File '+hls_ref_path+' has not been created yet!')
        continue
    make_fractions(bin_path, hls_ref_path, model_assessdir+row['map']+'_30m_assess.tif')


#%%
# Collect assessment points
map_list = pd.read_csv(model_assessdir+'map_paths.csv')

# If comparing S1, select only assessment maps where we have S1 data & results
if do_s1==True:
    map_list = map_list.loc[map_list['S1']==1]
    map_list.reset_index(inplace=True)
    
    # For results with S1
    for i, row in map_list.iterrows():
        print(row['map'])
        
        # Map with models over the reference map
        # Read the 30 m map
        if not os.path.exists(model_assessdir+row['map']+'_30m_assess.tif'):
            print('File '+model_assessdir+row['map']+'_30m_assess.tif'+' has not been created yet!')
            continue
        
        with rio.open(model_assessdir+row['map']+'_30m_assess.tif') as src:  
            ref_map = src.read()
            
            # Read the modeled fcover map and clip to map
            with rio.open(model_outdir+str(row['year'])+'/FCover_'+row['hls_tile']+'_'+str(row['year'])+'_rfr_s1.tif') as pred_src:   
                pred_map = pred_src.read()
        
        # Filter to where both maps have data
        pred_map[ref_map==32767] = 32767; ref_map[pred_map==32767] = 32767
            
        nclasses, nx, ny = ref_map.shape
        ref_map = ref_map.reshape(5, nx*ny); pred_map = pred_map.reshape(5, nx*ny)
        ref_map = ref_map[:,ref_map[0]!=32767]; pred_map = pred_map[:,pred_map[0]!=32767]
        print(ref_map.shape)
        ref_map = np.round(ref_map*100)
        
        # Combine water and barren classes if not separated in reference map
        if row['unique_water']==0:
            pred_map[1] = pred_map[0] + pred_map[1]
            pred_map[0] = 0
        
        if ref_map.size==0: 
            print('Empty!')
            continue
        if i==0: # CHANGE TO 0 WHEN ALL MAPS ARE READY
            s1_assess_pts = [ref_map]
            s1_assess_pred = [pred_map]
            #all_used_pixels = [all_pixels]
        else:
            s1_assess_pts.append(ref_map)
            s1_assess_pred.append(pred_map)
    
# For results without S1    
for i, row in map_list.iterrows():
    print(row['map'])
    
    # Map with models over the reference map
    # Read the 30 m map
    if not os.path.exists(model_assessdir+row['map']+'_30m_assess.tif'):
        print('File '+model_assessdir+row['map']+'_30m_assess.tif'+' has not been created yet!')
        continue
    
    with rio.open(model_assessdir+row['map']+'_30m_assess.tif') as src:  
        ref_map = src.read()
        
        # Read the modeled fcover map and clip to map
        with rio.open(model_outdir+str(row['year'])+'/FCover_'+row['hls_tile']+'_'+str(row['year'])+'_rfr.tif') as pred_src:   
            pred_map = pred_src.read()
    
    # Filter to where both maps have data
    pred_map[ref_map==32767] = 32767; ref_map[pred_map==32767] = 32767
        
    nclasses, nx, ny = ref_map.shape
    ref_map = ref_map.reshape(5, nx*ny); pred_map = pred_map.reshape(5, nx*ny)
    ref_map = ref_map[:,ref_map[0]!=32767]; pred_map = pred_map[:,pred_map[0]!=32767]
    print(ref_map.shape)
    ref_map = np.round(ref_map*100)
    
    # Combine water and barren classes if not separated in reference map
    if row['unique_water']==0:
        pred_map[1] = pred_map[0] + pred_map[1]
        pred_map[0] = 0
    
    if ref_map.size==0: 
        print('Empty!')
        continue
    if i==0: # CHANGE TO 0 WHEN ALL MAPS ARE READY
        all_assess_pts = [ref_map]
        all_assess_pred = [pred_map]
        #all_used_pixels = [all_pixels]
    else:
        all_assess_pts.append(ref_map)
        all_assess_pred.append(pred_map)
        

#%%
# Assess for S1

if do_s1==True:
    # Orndahl maps
    orndahl_obs = np.hstack((s1_assess_pts[0], s1_assess_pts[1], s1_assess_pts[2], s1_assess_pts[3], 
                             s1_assess_pts[4], s1_assess_pts[5], s1_assess_pts[6], s1_assess_pts[7],
                             s1_assess_pts[8], s1_assess_pts[9], s1_assess_pts[10]))
    orndahl_pred = np.hstack((s1_assess_pred[0], s1_assess_pred[1], s1_assess_pred[2], s1_assess_pred[3], 
                             s1_assess_pred[4], s1_assess_pred[5], s1_assess_pred[6], s1_assess_pred[7],
                             s1_assess_pred[8], s1_assess_pred[9], s1_assess_pred[10]))
    make_plots(orndahl_obs, orndahl_pred)
    #make_hists(orndahl_obs, orndahl_pred)
    
    # Greaves maps
    greaves_obs = np.hstack((s1_assess_pts[11], s1_assess_pts[12], s1_assess_pts[13]))
    greaves_pred = np.hstack((s1_assess_pred[11], s1_assess_pred[12], s1_assess_pred[13]))
    col_indices = np.random.choice(greaves_obs.shape[1], 1500, replace=False)
    make_plots(greaves_obs[:, col_indices], greaves_pred[:, col_indices])
    # make_plots(greaves_obs, greaves_pred)
    #make_hists(greaves_obs, greaves_pred)
    
    # Yang maps
    yang_obs = np.hstack((s1_assess_pts[14], s1_assess_pts[15], s1_assess_pts[16], s1_assess_pts[17], 
                             s1_assess_pts[18], s1_assess_pts[19], s1_assess_pts[20], s1_assess_pts[21],
                             s1_assess_pts[22], s1_assess_pts[23], s1_assess_pts[24], s1_assess_pts[25],
                             s1_assess_pts[26], s1_assess_pts[27]))
    yang_pred = np.hstack((s1_assess_pred[14], s1_assess_pred[15], s1_assess_pred[16], s1_assess_pred[17], 
                             s1_assess_pred[18], s1_assess_pred[19], s1_assess_pred[20], s1_assess_pred[21],
                             s1_assess_pred[22], s1_assess_pred[23], s1_assess_pred[24], s1_assess_pred[25],
                             s1_assess_pred[26], s1_assess_pred[27]))
    col_indices = np.random.choice(yang_obs.shape[1], 1500, replace=False)
    make_plots(yang_obs[:, col_indices], yang_pred[:, col_indices])
    #make_hists(yang_obs, yang_pred)
    
    # Now, without S1
    # Orndahl maps
    orndahl_obs = np.hstack((all_assess_pts[0], all_assess_pts[1], all_assess_pts[2], all_assess_pts[3], 
                             all_assess_pts[4], all_assess_pts[5], all_assess_pts[6], all_assess_pts[7],
                             all_assess_pts[8], all_assess_pts[9], all_assess_pts[10]))
    orndahl_pred = np.hstack((all_assess_pred[0], all_assess_pred[1], all_assess_pred[2], all_assess_pred[3], 
                             all_assess_pred[4], all_assess_pred[5], all_assess_pred[6], all_assess_pred[7],
                             all_assess_pred[8], all_assess_pred[9], all_assess_pred[10]))
    make_plots(orndahl_obs, orndahl_pred)
    #make_hists(orndahl_obs, orndahl_pred)
    
    # Greaves maps
    greaves_obs = np.hstack((all_assess_pts[11], all_assess_pts[12], all_assess_pts[13]))
    greaves_pred = np.hstack((all_assess_pred[11], all_assess_pred[12], all_assess_pred[13]))
    col_indices = np.random.choice(greaves_obs.shape[1], 1500, replace=False)
    make_plots(greaves_obs[:, col_indices], greaves_pred[:, col_indices])
    # make_plots(greaves_obs, greaves_pred)
    #make_hists(greaves_obs, greaves_pred)
    
    # Yang maps
    yang_obs = np.hstack((all_assess_pts[14], all_assess_pts[15], all_assess_pts[16], all_assess_pts[17], 
                             all_assess_pts[18], all_assess_pts[19], all_assess_pts[20], all_assess_pts[21],
                             all_assess_pts[22], all_assess_pts[23], all_assess_pts[24], all_assess_pts[25],
                             all_assess_pts[26], all_assess_pts[27]))
    yang_pred = np.hstack((all_assess_pred[14], all_assess_pred[15], all_assess_pred[16], all_assess_pred[17], 
                             all_assess_pred[18], all_assess_pred[19], all_assess_pred[20], all_assess_pred[21],
                             all_assess_pred[22], all_assess_pred[23], all_assess_pred[24], all_assess_pred[25],
                             all_assess_pred[26], all_assess_pred[27]))
    col_indices = np.random.choice(yang_obs.shape[1], 1500, replace=False)
    make_plots(yang_obs[:, col_indices], yang_pred[:, col_indices])
    #make_hists(yang_obs, yang_pred)
    

#%%
# Assess
# Fraser map
fraser_obs = all_assess_pts[0]
fraser_pred = all_assess_pred[0]
make_plots(fraser_obs, fraser_pred)
make_hists(fraser_obs, fraser_pred)

# Orndahl maps
orndahl_obs = np.hstack((all_assess_pts[1], all_assess_pts[2], all_assess_pts[3], all_assess_pts[4], 
                         all_assess_pts[5], all_assess_pts[6], all_assess_pts[7], all_assess_pts[8],
                         all_assess_pts[9], all_assess_pts[10], all_assess_pts[11]))
orndahl_pred = np.hstack((all_assess_pred[1], all_assess_pred[2], all_assess_pred[3], all_assess_pred[4], 
                          all_assess_pred[5], all_assess_pred[6], all_assess_pred[7], all_assess_pred[8],
                          all_assess_pred[9], all_assess_pred[10], all_assess_pred[11]))
make_plots(orndahl_obs, orndahl_pred)
make_hists(orndahl_obs, orndahl_pred)

# Greaves maps
greaves_obs = np.hstack((all_assess_pts[12], all_assess_pts[13], all_assess_pts[14]))
greaves_pred = np.hstack((all_assess_pred[12], all_assess_pred[13], all_assess_pred[14]))
# col_indices = np.random.choice(greaves_obs.shape[1], 1500, replace=False)
# make_plots(greaves_obs[:, col_indices], greaves_pred[:, col_indices])
# make_plots(greaves_obs, greaves_pred)
make_hists(greaves_obs, greaves_pred)

# Grunberg maps

# Yang maps
