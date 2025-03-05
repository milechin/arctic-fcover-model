import glob
import numpy as np
import pandas as pd
import rasterio as rio
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
start_yr = 2016
end_yr = 2023
phenometrics = ['50PCGI','Peak','50PCGD']
metrics = ['dem', 'water', 'slope', 'aspect']

#### PARAMETERS ####
years = np.arange(start_yr, end_yr+1, 1)
assess_key = pd.read_csv(traindir+'assessment_key.csv')


#%%
def sample_bias(map_path, premask_path, classified_map, nclasses, n):
    # Read the unmasked map
    with rio.open(premask_path) as rawsrc:
        premask_map = rawsrc.read()
        
    # Read the 30 m map
    with rio.open(map_path) as src:
        train_map = src.read()
        
    # Stratify sample by shrub/tree fractional distr
    if nclasses==5:
        strat_band = premask_map[4,:,:]
    elif nclasses==4:
        strat_band = premask_map[3,:,:]
    total_nobs = strat_band[strat_band!=32767].size  # number of pixels total
    plt.hist(strat_band[strat_band!=32767]); plt.show()
    
    # Find all indices of bins with 20% cover range
    for lower_bound in np.linspace(0,0.8,5):
        upper_bound = lower_bound+0.2
        
        if lower_bound==0: 
            indices = np.where((strat_band>=lower_bound) & (strat_band<=upper_bound))
        else:
            indices = np.where((strat_band>lower_bound) & (strat_band<=upper_bound))
        
        range_nobs = indices[0].size  # number of pixels in this range/bin
        print(range_nobs)
        # Sample size > 10 or all pixels in range if < 10 pixels
        sample_prop = range_nobs/total_nobs
        #sample_prop = np.mean([sample_prop,0.2])
        sample_size = np.round(sample_prop*n).astype(int)
        if sample_size<10: sample_size = min(range_nobs, 10)
        print(sample_size)
        
        # Randomly sample indices with sample size given by the stratification
        # # Stratify sample by shrub/tree fractional distr
        if nclasses==5:
            strat_b = train_map[4,:,:]
        elif nclasses==4:
            strat_b = train_map[3,:,:]
        
        # Find all indices of bins with 20% cover range        
        if lower_bound==0: 
            indices2 = np.where((strat_b>=lower_bound) & (strat_b<=upper_bound))
        else:
            indices2 = np.where((strat_b>lower_bound) & (strat_b<=upper_bound))
        
        range_nobs2 = indices2[0].size  # number of pixels in this range/bin
        
        sample_size = min(sample_size, range_nobs2)  # in case we have fewer usuable obs than the sample size after qa
        if sample_size==0: continue
        selected = np.random.choice(range(0, range_nobs2), size=sample_size, replace=False)
        
        # Collect training points and features from train map using indices
        x_inds = indices2[0]; y_inds = indices2[1]
        bias_sample = train_map[:, x_inds[selected], y_inds[selected]]
        bias_pred_sample = classified_map[:, x_inds[selected], y_inds[selected]]
        if lower_bound==0: 
            bias_pts = bias_sample
            bias_predicted = bias_pred_sample
        else:
            bias_pts = np.hstack((bias_pts, bias_sample))
            bias_predicted = np.hstack((bias_predicted, bias_pred_sample))
        print(bias_pts.shape, bias_predicted.shape)
        
    # if nclasses==4: 
    #     new_row = np.zeros((1, bias_pts.shape[1]))
    #     bias_pts = np.vstack([bias_pts, new_row])
    #     print('Classes:',bias_pts.shape[0])
        
    return bias_pts, bias_predicted


#%%
# 3) Run models, bias correction, and post-processing on training images and assess 
#    against an independent sample of pixels

# Read in models
trees_regressor = pickle.load(open(model_traindir+"trees_rfr.pickle", "rb"))
shrub_regressor = pickle.load(open(model_traindir+"shrub_rfr.pickle", "rb"))
herb_regressor = pickle.load(open(model_traindir+"herb_rfr.pickle", "rb"))
barren_regressor = pickle.load(open(model_traindir+"barren_rfr.pickle", "rb"))
water_regressor = pickle.load(open(model_traindir+"water_rfr.pickle", "rb"))
# Read in bias models 
water_bias = pickle.load(open(model_traindir+"water_bias.pickle", "rb"))
barren_bias = pickle.load(open(model_traindir+"barren_bias.pickle", "rb"))
herb_bias = pickle.load(open(model_traindir+"herb_bias.pickle", "rb"))
shrub_bias = pickle.load(open(model_traindir+"shrub_bias.pickle", "rb"))
trees_bias = pickle.load(open(model_traindir+"trees_bias.pickle", "rb"))

# Predict FCover on training images
for i, row in assess_key.iterrows():
    # Read the 30 m map
    with rio.open(traindir+row['ecoreg_code']+'/'+row['assess_img_name']) as src:        
        # Read the feats and clip to map
        with rio.open(outdir+str(row['year'])+'/'+row['above_tile']+'/feats_full_'+str(row['year'])+'.tif') as featsrc:   
            window = rio.windows.from_bounds(*src.bounds, transform=featsrc.transform)
            full_stack_img = featsrc.read(window=window, boundless=True)
            full_stack_img = full_stack_img.astype('float32'); full_stack_img[full_stack_img==32767] = np.nan

    #meta = full_stack.meta
    nsamples, nx, ny = full_stack_img.shape 
    full_stack_img_2d = full_stack_img.reshape((nsamples,nx*ny))
    full_stack_img_2d = full_stack_img_2d.T
    
    # Water fractional classification and display
    water_predict = water_regressor.predict(full_stack_img_2d)
    # Barren fractional classification and display
    barren_predict = barren_regressor.predict(full_stack_img_2d)
    # Herb fractional classification and display
    herb_predict = herb_regressor.predict(full_stack_img_2d)
    # Shrub fractional classification and display
    shrub_predict = shrub_regressor.predict(full_stack_img_2d)
    # Trees fractional classification and display
    trees_predict = trees_regressor.predict(full_stack_img_2d)
    
    # Do bias correction
    water_corrected = water_predict - water_bias.predict(water_predict.reshape(-1,1))
    barren_corrected = barren_predict - barren_bias.predict(barren_predict.reshape(-1,1))
    herb_corrected = herb_predict - herb_bias.predict(herb_predict.reshape(-1,1))
    shrub_corrected = shrub_predict - shrub_bias.predict(shrub_predict.reshape(-1,1))
    trees_corrected = trees_predict - trees_bias.predict(trees_predict.reshape(-1,1))
    
    # Correct negative values to 0% cover
    water_corrected[water_corrected<0]=0
    barren_corrected[barren_corrected<0]=0
    herb_corrected[herb_corrected<0]=0
    shrub_corrected[shrub_corrected<0]=0
    trees_corrected[trees_corrected<0]=0
    
    # Remove water fraction if fractions from other modeled cover is already >95%
    nonwater = np.vstack((barren_corrected,herb_corrected,shrub_corrected,trees_corrected))
    totals_nonwater = nonwater.sum(axis=0)
    water_corrected[totals_nonwater>=0.95] = 0
    # Remove barren fraction if fractions from other modeled cover is already >95%
    nonbarren = np.vstack((water_corrected,herb_corrected,shrub_corrected,trees_corrected))
    totals_nonbarren = nonbarren.sum(axis=0)
    barren_corrected[(totals_nonbarren>=0.97) & (barren_corrected>=.04)] = 0
    # Remove tree fraction if fractions from other modeled cover is already >95%
    nontree = np.vstack((water_corrected,barren_corrected,herb_corrected,shrub_corrected))
    totals_nontree = nontree.sum(axis=0)
    trees_corrected[totals_nontree>=0.95] = 0
    
    # Phenology masking
    try:
        with rio.open(outdir+str(row['year'])+'/'+row['above_tile']+'/'+row['above_tile']+'_'+str(row['year'])+'_pheno_final.tif','r') as src0:
            #pheno = src0.read(1).reshape(nx*ny)
            pheno = src0.read(1, window=window, boundless=True).reshape(nx*ny)
    except:
        print('No pheno file yet for this tile!')
        pheno = np.zeros_like(water_corrected)
    with rio.open(outdir+'2016/'+row['above_tile']+'/'+row['above_tile']+'_water_final.tif','r') as src1:
        water_presence = src1.read(window=window, boundless=True).reshape(nx*ny)
    if np.any(pheno):
        # 100% water if no phenology and water mask is True
        water_corrected[(pheno==32767) & ((water_presence==0) | (water_presence==2))] = 1
        barren_corrected[(pheno==32767) & ((water_presence==0) | (water_presence==2))] = 0
        herb_corrected[(pheno==32767) & ((water_presence==0) | (water_presence==2))] = 0
        shrub_corrected[(pheno==32767) & ((water_presence==0) | (water_presence==2))] = 0
        trees_corrected[(pheno==32767) & ((water_presence==0) | (water_presence==2))] = 0
        # 100% barren if no phenology and water mask is False
        # water_corrected[(pheno==32767) & (water_presence==1)] = 0
        # barren_corrected[(pheno==32767) & (water_presence==1)] = 1
        # herb_corrected[(pheno==32767) & (water_presence==1)] = 0
        # shrub_corrected[(pheno==32767) & (water_presence==1)] = 0
        # trees_corrected[(pheno==32767) & (water_presence==1)] = 0
    else:
        print('Still masking water despite no pheno!')
        # 100% water if water mask is True
        water_corrected[(water_presence==0) | (water_presence==2)] = 1
        barren_corrected[(water_presence==0) | (water_presence==2)] = 0
        herb_corrected[(water_presence==0) | (water_presence==2)] = 0
        shrub_corrected[(water_presence==0) | (water_presence==2)] = 0
        trees_corrected[(water_presence==0) | (water_presence==2)] = 0
    
    # Water and Barren post-processing step: >= 90% cover post-correction becomes 100% and 0% for all other cover types
    water_corrected[water_corrected>=0.95] = 1
    barren_corrected[water_corrected>=0.95] = 0
    herb_corrected[water_corrected>=0.95] = 0
    shrub_corrected[water_corrected>=0.95] = 0
    trees_corrected[water_corrected>=0.95] = 0
    barren_corrected[barren_corrected>=0.95] = 1
    water_corrected[barren_corrected>=0.95] = 0
    herb_corrected[barren_corrected>=0.95] = 0
    shrub_corrected[barren_corrected>=0.95] = 0
    trees_corrected[barren_corrected>=0.95] = 0
    
    # Trees post-processing step: remove fractional trees <= 2% cover (low values lead to inaccuracies since so sparse)
    trees_corrected[trees_corrected<=0.02] = 0

    # Normalize and fix errors
    # First step to normalize results: joining the fractional cover
    all_corrected = np.vstack((water_corrected,barren_corrected,herb_corrected,shrub_corrected,trees_corrected))
    # Sums, which need to be normalized to 1
    totals_corrected = all_corrected.sum(axis=0)
    # Normalized fractional cover for each class
    all_norm = all_corrected/totals_corrected

    # Separating back out to classes after normalization
    water_norm_f = all_norm[0].reshape(1, nx*ny)
    barren_norm_f = all_norm[1].reshape(1, nx*ny)
    herb_norm_f = all_norm[2].reshape(1, nx*ny)
    shrub_norm_f = all_norm[3].reshape(1, nx*ny)
    trees_norm_f = all_norm[4].reshape(1, nx*ny)

    # Removing no-data values and scaling (SOME MAY SUM TO >100 THIS WAY)
    water_normf = water_norm_f*100
    barren_normf = barren_norm_f*100
    herb_normf = herb_norm_f*100
    shrub_normf = shrub_norm_f*100
    trees_normf = trees_norm_f*100
    water_norm = np.around(water_normf)
    barren_norm = np.around(barren_normf)
    herb_norm = np.around(herb_normf)
    shrub_norm = np.around(shrub_normf)
    trees_norm = np.around(trees_normf)
    
    # Fix normalization errors
    all_normf = np.concatenate([water_normf,barren_normf,herb_normf,shrub_normf,trees_normf], axis=0)
    all_norm = np.concatenate([water_norm,barren_norm,herb_norm,shrub_norm,trees_norm], axis=0)
    totals_norm = all_norm.sum(axis=0)
    totals_norm = totals_norm.reshape(1, nx*ny)
    incorrect_sums = totals_norm[totals_norm!=100]
    
    # 99% case
    adj_i = np.argwhere(totals_norm<100)[:,1]
    adj_mask = np.zeros((5,nx*ny), dtype=int)
    adj_mask[:, adj_i] = 1
    difs = all_norm - all_normf
    mins_i = np.argwhere(difs == np.min(difs, axis=0))
    mins_mask = np.zeros((5,nx*ny), dtype=int)
    mins_mask[mins_i[:,0], mins_i[:,1]] = 1
    all_norm[(adj_mask==1) & (mins_mask==1)] += 1
    totals_norm = all_norm.sum(axis=0).reshape(1, nx*ny)
    print(np.unique(totals_norm[totals_norm!=100]))
    # 98% case
    adj_i = np.argwhere(totals_norm<100)[:,1]
    adj_mask = np.zeros((5,nx*ny), dtype=int)
    adj_mask[:, adj_i] = 1
    difs[mins_i[:,0],mins_i[:,1]] = 200
    mins_i = np.argwhere(difs == np.min(difs, axis=0))
    mins_mask = np.zeros((5,nx*ny), dtype=int)
    mins_mask[mins_i[:,0], mins_i[:,1]] = 1
    all_norm[(adj_mask==1) & (mins_mask==1)] += 1
    totals_norm = all_norm.sum(axis=0).reshape(1, nx*ny)
    print(np.unique(totals_norm[totals_norm!=100]))
    # 101% case
    adj_i = np.argwhere(totals_norm>100)[:,1]
    adj_mask = np.zeros((5,nx*ny), dtype=int)
    adj_mask[:, adj_i] = 1
    difs = all_norm - all_normf
    maxs_i = np.argwhere(difs == np.max(difs, axis=0))
    maxs_mask = np.zeros((5,nx*ny), dtype=int)
    maxs_mask[maxs_i[:,0], maxs_i[:,1]] = 1
    all_norm[(adj_mask==1) & (maxs_mask==1)] -= 1
    totals_norm = all_norm.sum(axis=0).reshape(1, nx*ny)
    print(np.unique(totals_norm[totals_norm!=100]))
    # 102% case
    adj_i = np.argwhere(totals_norm>100)[:,1]
    adj_mask = np.zeros((5,nx*ny), dtype=int)
    adj_mask[:, adj_i] = 1
    difs[maxs_i[:,0],maxs_i[:,1]] = -200
    maxs_i = np.argwhere(difs == np.max(difs, axis=0))
    maxs_mask = np.zeros((5,nx*ny), dtype=int)
    maxs_mask[maxs_i[:,0], maxs_i[:,1]] = 1
    all_norm[(adj_mask==1) & (maxs_mask==1)] -= 1
    totals_norm = all_norm.sum(axis=0).reshape(1, nx*ny)
    print(np.unique(totals_norm[totals_norm!=100]))
    
    print('Fixed normalizations')
    
    # Get assessment sample
    print(row['assess_img_name'])
    assess_pts, assess_pred = sample_bias(traindir+row['ecoreg_code']+'/'+row['assess_img_name'],
                                                    traindir+row['ecoreg_code']+'/'+row['assess_img_name'],#[:-10]+'.tif',
                                                    all_norm.reshape(5,nx,ny),
                                                    row['nclasses'],
                                                    row['nsamples'])
    
    if i==0:
        all_assess_pts = assess_pts
        all_assess_pred = assess_pred
        #all_used_pixels = [all_pixels]
    else:
        all_assess_pts = np.hstack((all_assess_pts, assess_pts))
        all_assess_pred = np.hstack((all_assess_pred, assess_pred))
        #all_used_pixels.append(all_pixels)
    
    temp_assess_pts = np.delete(all_assess_pts, np.where((all_assess_pts==32767).any(axis=0))[0], axis=1)
    temp_assess_pred = np.delete(all_assess_pred, np.where((all_assess_pts==32767).any(axis=0))[0], axis=1)
    final_assess_pts = np.delete(temp_assess_pts, np.where((temp_assess_pred==32767).any(axis=0))[0], axis=1)
    final_assess_pred = np.delete(temp_assess_pred, np.where((temp_assess_pred==32767).any(axis=0))[0], axis=1)

# Scale
final_assess_pts = final_assess_pts*100


#%%
# Assessment Plots
fig, ((ax1, ax2, ax5), (ax3, ax4, ax6)) = plt.subplots(2, 3)
fig.set_size_inches(34, 18)
#axes.set_facecolor('black')

compare_array = np.vstack([final_assess_pts[0], final_assess_pred[0]])
z = gaussian_kde(compare_array)(compare_array)
#plt1=ax1.scatter(final_assess_pts[0], final_assess_pred[0], c=z, cmap='plasma', s=100)
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
z = gaussian_kde(compare_array)(compare_array)
#plt2=ax2.scatter(final_assess_pts[1], final_assess_pred[1], c=z, cmap='plasma', s=100)
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
z = gaussian_kde(compare_array)(compare_array)
#plt3=ax3.scatter(final_assess_pts[2], final_assess_pred[2], c=z, cmap='plasma', s=100)
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
z = gaussian_kde(compare_array)(compare_array)
#plt4=ax4.scatter(final_assess_pts[3], final_assess_pred[3], c=z, cmap='plasma', s=100)
plt4=ax4.hexbin(final_assess_pts[3], final_assess_pred[3], bins='log', gridsize=25, cmap='YlGn')
ax4.plot([0,100],[0,100], color='black', linewidth=.5)
sb.regplot(y=final_assess_pred[3],x=final_assess_pts[3], scatter=False, color='green', scatter_kws={'alpha':0.2}, ax=ax4)
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
z = gaussian_kde(compare_array)(compare_array)
#plt5=ax5.scatter(final_assess_pts[4], final_assess_pred[4], c=z, cmap='plasma', s=50)
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

# Histograms
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(8, 6)
sb.histplot(final_assess_pts[0], kde=True, bins=10, color='#0071dc', label='Observed')
sb.histplot(final_assess_pred[0], kde=True, bins=10, color='#982649', label='Predicted')
plt.legend(frameon=False, fontsize=12)
plt.title('Water', fontsize=20)
plt.xlim(0,100)
plt.xlabel('Cover fraction, %', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.show()

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(8, 6)
sb.histplot(final_assess_pts[1], kde=True, bins=10, color='#616161', label='Observed')
sb.histplot(final_assess_pred[1], kde=True, bins=10, color='#982649', label='Predicted')
plt.legend(frameon=False, fontsize=12)
plt.title('Barren', fontsize=20)
plt.xlim(0,100)
plt.xlabel('Cover fraction, %', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.show()

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(8, 6)
sb.histplot(final_assess_pts[4], kde=True, bins=10, color='darkgreen', label='Observed')
sb.histplot(final_assess_pred[4], kde=True, bins=10, color='#982649', label='Predicted')
plt.legend(frameon=False, fontsize=12)
plt.title('Trees', fontsize=20)
plt.xlim(0,100)
plt.xlabel('Cover fraction, %', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.show()

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(8, 6)
sb.histplot(final_assess_pts[2], kde=True, bins=10, color='orange', label='Observed')
sb.histplot(final_assess_pred[2], kde=True, bins=10, color='#982649', label='Predicted')
plt.legend(frameon=False, fontsize=12)
plt.title('Low vegetation', fontsize=20)
plt.xlim(0,100)
plt.xlabel('Cover fraction, %', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.show()

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(8, 6)
sb.histplot(final_assess_pts[3], kde=True, bins=10, color='#4d8f00', label='Observed')
sb.histplot(final_assess_pred[3], kde=True, bins=10, color='#982649', label='Predicted')
plt.legend(frameon=False, fontsize=12)
plt.title('Low & tall shrubs', fontsize=20)
plt.xlim(0,100)
plt.xlabel('Cover fraction, %', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.show()


#%%
#---------------------------------------------------------------------------------------------#
# Individual map assessments (filter by indices)
fig, ((ax1, ax2, ax5), (ax3, ax4, ax6)) = plt.subplots(2, 3)
fig.set_size_inches(34, 18)

compare_array = np.vstack([final_assess_pts[0,2004:], final_assess_pred[0,2004:]])
#z = gaussian_kde(compare_array)(compare_array)
#plt1=ax1.scatter(final_assess_pts[0], final_assess_pred[0], c=z, cmap='plasma', s=100)
plt1=ax1.hexbin(final_assess_pts[0,2004:], final_assess_pred[0,2004:], bins='log', gridsize=25, cmap='Blues')
ax1.plot([0,100],[0,100], color='black', linewidth=.5)
sb.regplot(y=final_assess_pred[0,2004:],x=final_assess_pts[0,2004:], scatter=False, color='dodgerblue', scatter_kws={'alpha':0.2}, ax=ax1)
ax1.text(5, 95, '$R^2$ = '+str(np.round(np.corrcoef(final_assess_pts[0,2004:], final_assess_pred[0,2004:])[0,1]**2,2)), fontsize=20,color='black')
ax1.text(5, 90, 'MAE = '+str(np.round(mean_absolute_error(final_assess_pts[0,2004:], final_assess_pred[0,2004:]),2))+'%', fontsize=20,color='black')
ax1.text(5, 85, 'RMSE = '+str(np.round(np.sqrt(mean_squared_error(final_assess_pts[0,2004:], final_assess_pred[0,2004:])),2))+'%', fontsize=20,color='black')
ax1.text(5, 80, 'Bias = '+str(np.round(-np.mean(final_assess_pts[0,1001:2004]-final_assess_pred[0,1001:2004]),2))+'%', fontsize=20,color='black')
ax1.set_xlim(0,100)
ax1.set_ylim(0,100)
ax1.set_title('Water fractional cover', fontsize=32)
ax1.set_xlabel('Observed', fontsize=24); ax1.set_ylabel('Predicted', fontsize=24)
ax1.set_facecolor('white')
plt.colorbar(plt1, ax=ax1)

compare_array = np.vstack([final_assess_pts[1,2004:], final_assess_pred[1,2004:]])
#z = gaussian_kde(compare_array)(compare_array)
#plt2=ax2.scatter(final_assess_pts[1], final_assess_pred[1], c=z, cmap='plasma', s=100)
plt2=ax2.hexbin(final_assess_pts[1,2004:], final_assess_pred[1,2004:], bins='log', gridsize=25, cmap='Greys')
ax2.plot([0,100],[0,100], color='black', linewidth=.5)
sb.regplot(y=final_assess_pred[1,2004:],x=final_assess_pts[1,2004:], scatter=False, color='grey', scatter_kws={'alpha':0.2}, ax=ax2)
ax2.text(5, 95, '$R^2$ = '+str(np.round(np.corrcoef(final_assess_pts[1,2004:],final_assess_pred[1,2004:])[0,1]**2,2)), fontsize=20,color='black')
ax2.text(5, 90, 'MAE = '+str(np.round(mean_absolute_error(final_assess_pts[1,2004:],final_assess_pred[1,2004:]),2))+'%', fontsize=20,color='black')
ax2.text(5, 85, 'RMSE = '+str(np.round(np.sqrt(mean_squared_error(final_assess_pts[1,2004:],final_assess_pred[1,2004:])),2))+'%', fontsize=20,color='black')
ax2.text(5, 80, 'Bias = '+str(np.round(-np.mean(final_assess_pts[1,2004:]-final_assess_pred[1,2004:]), 2))+'%', fontsize=20,color='black')
ax2.set_xlim(0,100)
ax2.set_ylim(0,100)
ax2.set_title('Barren fractional cover', fontsize=32)
ax2.set_xlabel('Observed', fontsize=24); ax2.set_ylabel('Predicted', fontsize=24)
ax2.set_facecolor('white')
plt.colorbar(plt2, ax=ax2)

compare_array = np.vstack([final_assess_pts[2,2004:], final_assess_pred[2,2004:]])
#z = gaussian_kde(compare_array)(compare_array)
#plt3=ax3.scatter(final_assess_pts[2], final_assess_pred[2], c=z, cmap='plasma', s=100)
plt3=ax3.hexbin(final_assess_pts[2,2004:], final_assess_pred[2,2004:], bins='log', gridsize=25, cmap='YlOrBr')
ax3.plot([0,100],[0,100], color='black', linewidth=.5)
sb.regplot(y=final_assess_pred[2,2004:],x=final_assess_pts[2,2004:], scatter=False, color='yellowgreen', scatter_kws={'alpha':0.2}, ax=ax3)
ax3.text(5, 95, '$R^2$ = '+str(np.round(np.corrcoef(final_assess_pts[2,2004:],final_assess_pred[2,2004:])[0,1]**2,2)), fontsize=20,color='black')
ax3.text(5, 90, 'MAE = '+str(np.round(mean_absolute_error(final_assess_pts[2,2004:],final_assess_pred[2,2004:]),2))+'%', fontsize=20,color='black')
ax3.text(5, 85, 'RMSE = '+str(np.round(np.sqrt(mean_squared_error(final_assess_pts[2,2004:],final_assess_pred[2,2004:])),2))+'%', fontsize=20,color='black')
ax3.text(5, 80, 'Bias = '+str(np.round(-np.mean(final_assess_pts[2,2004:]-final_assess_pred[2,2004:]), 2))+'%', fontsize=20,color='black')
ax3.set_xlim(0,100)
ax3.set_ylim(0,100)
ax3.set_title('Non-woody fractional cover', fontsize=32)
ax3.set_xlabel('Observed', fontsize=24); ax3.set_ylabel('Predicted', fontsize=24)
ax3.set_facecolor('white')
plt.colorbar(plt3, ax=ax3)

compare_array = np.vstack([final_assess_pts[3,2004:], final_assess_pred[3,2004:]])
#z = gaussian_kde(compare_array)(compare_array)
#plt4=ax4.scatter(final_assess_pts[3], final_assess_pred[3], c=z, cmap='plasma', s=100)
plt4=ax4.hexbin(final_assess_pts[3,2004:], final_assess_pred[3,2004:], bins='log', gridsize=25, cmap='YlGn')
ax4.plot([0,100],[0,100], color='black', linewidth=.5)
sb.regplot(y=final_assess_pred[3,2004:],x=final_assess_pts[3,2004:], scatter=False, color='green', scatter_kws={'alpha':0.2}, ax=ax4)
ax4.text(5, 95, '$R^2$ = '+str(np.round(np.corrcoef(final_assess_pts[3,2004:],final_assess_pred[3,2004:])[0,1]**2,2)), fontsize=20,color='black')
ax4.text(5, 90, 'MAE = '+str(np.round(mean_absolute_error(final_assess_pts[3,2004:],final_assess_pred[3,2004:]),2))+'%', fontsize=20,color='black')
ax4.text(5, 85, 'RMSE = '+str(np.round(np.sqrt(mean_squared_error(final_assess_pts[3,2004:],final_assess_pred[3,2004:])),2))+'%', fontsize=20,color='black')
ax4.text(5, 80, 'Bias = '+str(np.round(-np.mean(final_assess_pts[3,2004:]-final_assess_pred[3,2004:]), 2))+'%', fontsize=20,color='black')
ax4.set_xlim(0,100)
ax4.set_ylim(0,100)
ax4.set_title('Shrub fractional cover', fontsize=32)
ax4.set_xlabel('Observed', fontsize=24); ax4.set_ylabel('Predicted', fontsize=24)
ax4.set_facecolor('white')
plt.colorbar(plt4, ax=ax4)

compare_array = np.vstack([final_assess_pts[4,2004:], final_assess_pred[4,2004:]])
#z = gaussian_kde(compare_array)(compare_array)
#plt5=ax5.scatter(final_assess_pts[4], final_assess_pred[4], c=z, cmap='plasma', s=50)
plt5=ax5.hexbin(final_assess_pts[4,2004:], final_assess_pred[4,2004:], bins='log', gridsize=25, cmap='PuBuGn')
ax5.plot([0,100],[0,100], color='black', linewidth=.5)
sb.regplot(y=final_assess_pred[4,2004:],x=final_assess_pts[4,2004:], scatter=False, color='darkgreen', scatter_kws={'alpha':0.2}, ax=ax5)
ax5.text(5, 95, '$R^2$ = '+str(np.round(np.corrcoef(final_assess_pts[4,2004:],final_assess_pred[4,2004:])[0,1]**2,2)), fontsize=20,color='black')
ax5.text(5, 90, 'MAE = '+str(np.round(mean_absolute_error(final_assess_pts[4,2004:],final_assess_pred[4,2004:]),2))+'%', fontsize=20,color='black')
ax5.text(5, 85, 'RMSE = '+str(np.round(np.sqrt(mean_squared_error(final_assess_pts[4,2004:],final_assess_pred[4,2004:])),2))+'%', fontsize=20,color='black')
ax5.text(5, 80, 'Bias = '+str(np.round(-np.mean(final_assess_pts[4,2004:]-final_assess_pred[4,2004:]), 2))+'%', fontsize=20,color='black')
ax5.set_xlim(0,100)
ax5.set_ylim(0,100)
ax5.set_title('Trees fractional cover', fontsize=32)
ax5.set_xlabel('Observed', fontsize=24); ax5.set_ylabel('Predicted', fontsize=24)
ax5.set_facecolor('white')
plt.colorbar(plt5, ax=ax5)