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

## THIS SCRIPT IS FOR RUNNING A MODEL WITH THE SAME REGRESSIAN AND BIAS MODEL TRAINING PTS 
## FROM THE S1 MODELS BUT WITHOUT S1 FEATURES
## RUN THIS AFTER RUNNING 06_train_models.py WITH do_s1 = True

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
train_key = pd.read_csv(traindir+'training_key.csv')


#%%
################################## FUNCTIONS ##################################
def sample_training(map_path, premask_path, nclasses, feats_path, n):
    # Read the unmasked map
    with rio.open(premask_path) as rawsrc:
        premask_map = rawsrc.read()
        
    # Read the 30 m map
    with rio.open(map_path) as src:
        train_map = src.read()
        
        # Read the feats and clip to map
        with rio.open(feats_path) as featsrc:   
            window = rio.windows.from_bounds(*src.bounds, transform=featsrc.transform)
            full_stack_clipped = featsrc.read(window=window, boundless=True)
        
    # Remove pixels from training map where S1 is nan
    if do_s1==True:
        train_map[:,full_stack_clipped[33]==32767] = 32767
        
    # Stratify sample by shrub/tree fractional distr
    if nclasses==5:
        strat_band = premask_map[4,:,:]
    elif nclasses==4:
        strat_band = premask_map[3,:,:]
    total_nobs = strat_band[strat_band!=32767].size  # number of pixels total
    train_pixels = np.zeros_like(strat_band, dtype=int)  # to keep track of training pixels
    plt.hist(strat_band[strat_band!=32767], bins=np.linspace(0,1,21), color='#006100'); plt.xlim(0,1.05)
    plt.show()
    
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
        sample_prop = np.mean([sample_prop,0.2])
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
        selected = np.random.choice(range(0, range_nobs2), size=sample_size, replace=False)
        
        # Collect training points and features from train map using indices
        x_inds = indices2[0]; y_inds = indices2[1]
        train_sample = train_map[:, x_inds[selected], y_inds[selected]]
        feats_sample = full_stack_clipped[:, x_inds[selected], y_inds[selected]]
        if lower_bound==0: 
            train_pts = train_sample
            feats_pts = feats_sample
        else:
            train_pts = np.hstack((train_pts, train_sample))
            feats_pts = np.hstack((feats_pts, feats_sample))
        print(train_pts.shape, feats_pts.shape)
        
        # Update train_pixels
        train_pixels[x_inds[selected], y_inds[selected]] = 1
        
    print('Total n training pixels:',train_pixels[train_pixels==1].size)
    plt.imshow(train_pixels); plt.show()
    # if nclasses==4: 
    #     new_row = np.zeros((1, train_pts.shape[1]))
    #     train_pts = np.vstack([train_pts, new_row])
    #     print('Classes:',train_pts.shape[0])
        
    return train_pts, feats_pts, train_pixels

def sample_bias(map_path, premask_path, classified_map, train_pixels_path, nclasses, n):
    train_pixels = np.loadtxt(train_pixels_path, dtype='int16', delimiter=',')
    
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
    
    assess_pixels = np.zeros_like(train_pixels, dtype='int16')
    
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
        strat_b[train_pixels==1] = 32767  # excluding pixels used for training
        
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
        
        # Update train_pixels
        assess_pixels[x_inds[selected], y_inds[selected]] = 1
        
    print('Total n assessment pixels:',assess_pixels[assess_pixels==1].size)
    # if nclasses==4: 
    #     new_row = np.zeros((1, bias_pts.shape[1]))
    #     bias_pts = np.vstack([bias_pts, new_row])
    #     print('Classes:',bias_pts.shape[0])
        
    return bias_pts, bias_predicted, assess_pixels

def get_training(train_ids_path, map_path, feats_path):
    # Read the training sample mask
    train_px = np.loadtxt(train_ids_path, dtype='int16', delimiter=',')
    print(train_px.shape)
    
    # Read the 30 m map
    with rio.open(map_path) as src:
        train_map = src.read()
        
        # Read the feats and clip to map
        with rio.open(feats_path) as featsrc:   
            window = rio.windows.from_bounds(*src.bounds, transform=featsrc.transform)
            full_stack_clipped = featsrc.read(window=window, boundless=True)
    
    training_pts = train_map[:, train_px==1]
    features = full_stack_clipped[:, train_px==1]
    
    return training_pts, features, train_px

def get_bias(all_train_ids_path, training_pixels, map_path, classified_map):
    # Read the bias sample mask
    bias_px = np.loadtxt(all_train_ids_path, dtype='int16', delimiter=',')
    bias_px[training_pixels==1] = 0
    
    # Read the 30 m map
    with rio.open(map_path) as src:
        train_map = src.read()
        
    bias_pts = train_map[:, bias_px==1]
    bias_predicted = classified_map[:, bias_px==1]
    
    return bias_pts, bias_predicted

def get_assess(map_path, classified_map, assess_pixels):
    # Read the 30 m map
    with rio.open(map_path) as src:
        train_map = src.read()
        
    obs_map = train_map[:, assess_pixels==1]
    pred_map = classified_map[:, assess_pixels==1]
    
    return obs_map, pred_map


#%%
################################## EXECUTION ##################################
# 1) Train models
# Get training sample
for i, row in train_key.iterrows():
    print(row['train_img_name'])
    train_pts, feat_pts, train_pixels = get_training(model_traindir+row['train_img_name'][:-4]+'_ids_prebias.txt',
                                                     traindir+row['ecoreg_code']+'/training_imgs/'+row['train_img_name'],
                                                     outdir+str(row['year'])+'/'+row['above_tile']+'/feats_full_'+str(row['year'])+'.tif')
    
    if i==0:
        all_train_pts = train_pts
        all_feat_pts = feat_pts
        all_train_pixels = [train_pixels]
    else:
        all_train_pts = np.hstack((all_train_pts, train_pts))
        all_feat_pts = np.hstack((all_feat_pts, feat_pts))
        all_train_pixels.append(train_pixels)
    
    temp_train_pts = np.delete(all_train_pts, np.where((all_train_pts==32767).any(axis=0))[0], axis=1)
    temp_feat_pts = np.delete(all_feat_pts, np.where((all_train_pts==32767).any(axis=0))[0], axis=1)
    final_train_pts = np.delete(temp_train_pts, np.where((temp_feat_pts==32767).any(axis=0))[0], axis=1)
    final_feat_pts = np.delete(temp_feat_pts, np.where((temp_feat_pts==32767).any(axis=0))[0], axis=1)

#%%    
# Train models
water_regressor = RandomForestRegressor(n_estimators=500, min_samples_leaf=5, random_state=7).fit(final_feat_pts.T, final_train_pts[0])
barren_regressor = RandomForestRegressor(n_estimators=500, min_samples_leaf=5, random_state=7).fit(final_feat_pts.T, final_train_pts[1])
herb_regressor = RandomForestRegressor(n_estimators=500, min_samples_leaf=5, random_state=7).fit(final_feat_pts.T, final_train_pts[2])
shrub_regressor = RandomForestRegressor(n_estimators=500, min_samples_leaf=5, random_state=7).fit(final_feat_pts.T, final_train_pts[3])
trees_regressor = RandomForestRegressor(n_estimators=500, min_samples_leaf=5, random_state=7).fit(final_feat_pts.T, final_train_pts[4])

# Variable importances
band_names = np.array(['50PCGI B','50PCGI G','50PCGI R','50PCGI NIR','50PCGI SWIR1','50PCGI SWIR2','50PCGI EVI2','50PCGI TCG','50PCGI TCW','50PCGI TCB',
                       'Peak B','Peak G','Peak R','Peak NIR','Peak SWIR1','Peak SWIR2','Peak EVI2','Peak TCG','Peak TCW','Peak TCB',
                       '50PCGD B','50PCGD G','50PCGD R','50PCGD NIR','50PCGD SWIR1','50PCGD SWIR2','50PCGD EVI2','50PCGD TCG','50PCGD TCW','50PCGD TCB',
                       'Elevation','Slope','Aspect'])

# w_band_names = np.array(['50PCGI NIR','50PCGI SWIR1','50PCGI SWIR2','50PCGI TCG','50PCGI TCW',
#                        'Peak_NIR','Peak SWIR1','Peak SWIR2','Peak TCG','Peak TCW',
#                        '50PCGD NIR','50PCGD SWIR1','50PCGD SWIR2','50PCGD TCG','50PCGD TCW',
#                        'Elevation','Slope','Aspect'])

plt.figure(figsize=(24,16))
herb_variable_importance = pd.DataFrame({'Band':band_names,'Importance':herb_regressor.feature_importances_})
herb_variable_importance.sort_values(by=['Importance'], ascending=True, inplace=True)
plt.barh(herb_variable_importance['Band'][-10:],herb_variable_importance['Importance'][-10:], color='#7bab00')
plt.show()

water_path = model_traindir+"water_rfr.pickle"
barren_path = model_traindir+"barren_rfr.pickle"
herb_path = model_traindir+"herb_rfr.pickle"
shrub_path = model_traindir+"shrub_rfr.pickle"
trees_path = model_traindir+"trees_rfr.pickle"
    
pickle.dump(water_regressor, open(water_path, "wb"))
pickle.dump(barren_regressor, open(barren_path, "wb"))
pickle.dump(herb_regressor, open(herb_path, "wb"))
pickle.dump(shrub_regressor, open(shrub_path, "wb"))
pickle.dump(trees_regressor, open(trees_path, "wb"))

# Save sampled training pixels
# for i, row in train_key.iterrows():
#     np.savetxt(model_traindir+row['train_img_name'][:-4]+'_ids_prebias.txt', all_train_pixels[i], delimiter=',')


#%%
# 2) Run models on training images and create bias correction models
# Predict FCover on training images
for i, row in train_key.iterrows():
    # Read the 30 m map
    with rio.open(traindir+row['ecoreg_code']+'/training_imgs/'+row['train_img_name']) as src:        
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
    
    # Removing no-data values and scaling
    # water_predict = np.around(water_predict*10000)
    # barren_predict = np.around(barren_predict*10000)
    # herb_predict = np.around(herb_predict*10000)
    # shrub_predict = np.around(shrub_predict*10000)
    # trees_predict = np.around(trees_predict*10000)
    
    # Reshape
    # water_predict = water_predict.reshape(1, nx*ny)
    # barren_predict = barren_predict.reshape(1, nx*ny)
    # herb_predict = herb_predict.reshape(1, nx*ny)
    # shrub_predict = shrub_predict.reshape(1, nx*ny)
    # trees_predict = trees_predict.reshape(1, nx*ny)

    # full_stack_band = full_stack_img[10].reshape(1, nx*ny)
    # water_predict[full_stack_band==32767] = 32767
    # barren_predict[full_stack_band==32767] = 32767
    # herb_predict[full_stack_band==32767] = 32767
    # shrub_predict[full_stack_band==32767] = 32767
    # trees_predict[full_stack_band==32767] = 32767

    # Reshape
    water_predict = water_predict.reshape(1, nx, ny)
    barren_predict = barren_predict.reshape(1, nx, ny)
    herb_predict = herb_predict.reshape(1, nx, ny)
    shrub_predict = shrub_predict.reshape(1, nx, ny)
    trees_predict = trees_predict.reshape(1, nx, ny)
    
    plt.imshow(shrub_predict[0])
    plt.show()
    
    # Stack bands
    all_predict = np.concatenate((water_predict,barren_predict,herb_predict,shrub_predict,trees_predict),
                                 axis=0)

    # Get bias correction sample
    print(row['train_img_name'])
    bias_pts, bias_pred = get_bias(model_traindir+row['train_img_name'][:-4]+'_ids.txt',
                                   all_train_pixels[i],
                                   traindir+row['ecoreg_code']+'/training_imgs/'+row['train_img_name'],
                                   all_predict)
    
    if i==0:
        all_bias_pts = bias_pts
        all_bias_pred = bias_pred
        # all_train_bias_pixels = [bias_pixels]
    else:
        all_bias_pts = np.hstack((all_bias_pts, bias_pts))
        all_bias_pred = np.hstack((all_bias_pred, bias_pred))
        # all_train_bias_pixels.append(bias_pixels)
    
    temp_bias_pts = np.delete(all_bias_pts, np.where((all_bias_pts==32767).any(axis=0))[0], axis=1)
    temp_bias_pred = np.delete(all_bias_pred, np.where((all_bias_pts==32767).any(axis=0))[0], axis=1)
    final_bias_pts = np.delete(temp_bias_pts, np.where((temp_bias_pred==32767).any(axis=0))[0], axis=1)
    final_bias_pred = np.delete(temp_bias_pred, np.where((temp_bias_pred==32767).any(axis=0))[0], axis=1)

# Correct the bias
# Bias correction
_, nrows = final_bias_pts.shape
reg0 = LinearRegression().fit(final_bias_pts[0].reshape(nrows,1), final_bias_pred[0]-final_bias_pts[0])
corrected0 = final_bias_pred[0] - reg0.predict(final_bias_pred[0].reshape(nrows,1))
#corrected0[corrected0<0]=0
reg1 = LinearRegression().fit(final_bias_pts[1].reshape(nrows,1), final_bias_pred[1]-final_bias_pts[1])
corrected1 = final_bias_pred[1] - reg1.predict(final_bias_pred[1].reshape(nrows,1))
#corrected1[corrected1<0]=0
reg2 = LinearRegression().fit(final_bias_pts[2].reshape(nrows,1), final_bias_pred[2]-final_bias_pts[2])
corrected2 = final_bias_pred[2] - reg2.predict(final_bias_pred[2].reshape(nrows,1))
#corrected2[corrected2<0]=0
reg3 = LinearRegression().fit(final_bias_pts[3].reshape(nrows,1), final_bias_pred[3]-final_bias_pts[3])
corrected3 = final_bias_pred[3] - reg3.predict(final_bias_pred[3].reshape(nrows,1))
#corrected3[corrected3<0]=0
reg4 = LinearRegression().fit(final_bias_pts[4].reshape(nrows,1), final_bias_pred[4]-final_bias_pts[4])
corrected4 = final_bias_pred[4] - reg4.predict(final_bias_pred[4].reshape(nrows,1))
#corrected4[corrected4<0]=0

fig, ((ax1, ax2, ax5), (ax3, ax4, ax6)) = plt.subplots(2, 3)
fig.set_size_inches(35, 18)
#axes.set_facecolor('black')

compare_array = np.vstack([final_bias_pts[0], corrected0])
z = gaussian_kde(compare_array)(compare_array)
sb.regplot(x=final_bias_pts[0],y=final_bias_pred[0], scatter=False, color='#0071DC', scatter_kws={'alpha':0.2}, ax=ax1)
plt1=ax1.scatter(final_bias_pts[0], corrected0, c=z, cmap='plasma', s=100)
ax1.plot([0,1],[0,1], color='Black', linewidth=.5)
sb.regplot(y=corrected0,x=final_bias_pts[0], scatter=False, color='mediumvioletred', scatter_kws={'alpha':0.2}, ax=ax1)
ax1.text(.05, .95, '$R^2$ = '+str(np.corrcoef(final_bias_pts[0],corrected0)[0,1]**2)[:6], fontsize=14,color='white')
ax1.text(.05, .9, 'MAE = '+str(mean_absolute_error(final_bias_pts[0],corrected0))[:6], fontsize=14,color='white')
ax1.text(.05, .85, 'RMSE = '+str(np.sqrt(mean_squared_error(final_bias_pts[0],corrected0)))[:6], fontsize=14,color='white')
ax1.set_xlim(0,1)
ax1.set_ylim(0,1)
plt.colorbar(plt1, ax=ax1)

compare_array = np.vstack([final_bias_pts[1], corrected1])
z = gaussian_kde(compare_array)(compare_array)
sb.regplot(x=final_bias_pts[1],y=final_bias_pred[1], scatter=False, color='Grey', scatter_kws={'alpha':0.2}, ax=ax2)
plt2=ax2.scatter(final_bias_pts[1], corrected1, c=z, cmap='plasma', s=100)
ax2.plot([0,1],[0,1], color='Black', linewidth=.5)
sb.regplot(y=corrected1,x=final_bias_pts[1], scatter=False, color='mediumvioletred', scatter_kws={'alpha':0.2}, ax=ax2)
ax2.text(.05, .95, '$R^2$ = '+str(np.corrcoef(final_bias_pts[1],corrected1)[0,1]**2)[:6], fontsize=14,color='white')
ax2.text(.05, .9, 'MAE = '+str(mean_absolute_error(final_bias_pts[1],corrected1))[:6], fontsize=14,color='white')
ax2.text(.05, .85, 'RMSE = '+str(np.sqrt(mean_squared_error(final_bias_pts[1],corrected1)))[:6], fontsize=14,color='white')
ax2.set_xlim(0,1)
ax2.set_ylim(0,1)
plt.colorbar(plt2, ax=ax2)

compare_array = np.vstack([final_bias_pts[2], corrected2])
z = gaussian_kde(compare_array)(compare_array)
sb.regplot(x=final_bias_pts[2],y=final_bias_pred[2], scatter=False, color='YellowGreen', scatter_kws={'alpha':0.2}, ax=ax3)
plt3=ax3.scatter(final_bias_pts[2], corrected2, c=z, cmap='plasma', s=100)
ax3.plot([0,1],[0,1], color='Black', linewidth=.5)
sb.regplot(y=corrected2,x=final_bias_pts[2], scatter=False, color='mediumvioletred', scatter_kws={'alpha':0.2}, ax=ax3)
ax3.text(.05, .95, '$R^2$ = '+str(np.corrcoef(final_bias_pts[2],corrected2)[0,1]**2)[:6], fontsize=14,color='white')
ax3.text(.05, .9, 'MAE = '+str(mean_absolute_error(final_bias_pts[2],corrected2))[:6], fontsize=14,color='white')
ax3.text(.05, .85, 'RMSE = '+str(np.sqrt(mean_squared_error(final_bias_pts[2],corrected2)))[:6], fontsize=14,color='white')
ax3.set_xlim(0,1)
ax3.set_ylim(0,1)
plt.colorbar(plt3, ax=ax3)

compare_array = np.vstack([final_bias_pts[3], corrected3])
z = gaussian_kde(compare_array)(compare_array)
sb.regplot(x=final_bias_pts[3],y=final_bias_pred[3], scatter=False, color='Green', scatter_kws={'alpha':0.2}, ax=ax4)
plt4=ax4.scatter(final_bias_pts[3], corrected3, c=z, cmap='plasma', s=100)
ax4.plot([0,1],[0,1], color='Black', linewidth=.5)
sb.regplot(y=corrected3,x=final_bias_pts[3], scatter=False, color='mediumvioletred', scatter_kws={'alpha':0.2}, ax=ax4)
ax4.text(.05, .95, '$R^2$ = '+str(np.corrcoef(final_bias_pts[3],corrected3)[0,1]**2)[:6], fontsize=14,color='white')
ax4.text(.05, .9, 'MAE = '+str(mean_absolute_error(final_bias_pts[3],corrected3))[:6], fontsize=14,color='white')
ax4.text(.05, .85, 'RMSE = '+str(np.sqrt(mean_squared_error(final_bias_pts[3],corrected3)))[:6], fontsize=14,color='white')
ax4.set_xlim(0,1)
ax4.set_ylim(0,1)
plt.colorbar(plt4, ax=ax4)

compare_array = np.vstack([final_bias_pts[4], corrected4])
z = gaussian_kde(compare_array)(compare_array)
sb.regplot(x=final_bias_pts[4],y=final_bias_pred[4], scatter=False, color='darkgreen', scatter_kws={'alpha':0.2}, ax=ax5)
plt5=ax5.scatter(final_bias_pts[4], corrected4, c=z, cmap='plasma', s=100)
ax5.plot([0,1],[0,1], color='Black', linewidth=.5)
sb.regplot(y=corrected4,x=final_bias_pts[4], scatter=False, color='mediumvioletred', scatter_kws={'alpha':0.2}, ax=ax5)
ax5.text(.05, .95, '$R^2$ = '+str(np.corrcoef(final_bias_pts[4],corrected4)[0,1]**2)[:6], fontsize=14,color='white')
ax5.text(.05, .9, 'MAE = '+str(mean_absolute_error(final_bias_pts[4],corrected4))[:6], fontsize=14,color='white')
ax5.text(.05, .85, 'RMSE = '+str(np.sqrt(mean_squared_error(final_bias_pts[4],corrected4)))[:6], fontsize=14,color='white')
ax5.set_xlim(0,1)
ax5.set_ylim(0,1)
plt.colorbar(plt5, ax=ax5)

# Save bias correction models
pickle.dump(reg0, open(model_traindir+'water_bias.pickle', "wb"))
pickle.dump(reg1, open(model_traindir+'barren_bias.pickle', "wb"))
pickle.dump(reg2, open(model_traindir+'herb_bias.pickle', "wb"))
pickle.dump(reg3, open(model_traindir+'shrub_bias.pickle', "wb"))
pickle.dump(reg4, open(model_traindir+'trees_bias.pickle', "wb"))

# Save sampled training pixels
# for i, row in train_key.iterrows():
#     np.savetxt(model_traindir+row['train_img_name'][:-4]+'_ids.txt', all_train_bias_pixels[i], delimiter=',')

#%%
# 3) Run models, bias correction, and post-processing on training images and assess 
#    against an independent sample of pixels
# Predict FCover on training images
for i, row in train_key.iterrows():
    # Read the 30 m map
    with rio.open(traindir+row['ecoreg_code']+'/training_imgs/'+row['train_img_name']) as src:        
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
    water_corrected = water_predict - reg0.predict(water_predict.reshape(-1,1))
    barren_corrected = barren_predict - reg1.predict(barren_predict.reshape(-1,1))
    herb_corrected = herb_predict - reg2.predict(herb_predict.reshape(-1,1))
    shrub_corrected = shrub_predict - reg3.predict(shrub_predict.reshape(-1,1))
    trees_corrected = trees_predict - reg4.predict(trees_predict.reshape(-1,1))
    
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
    # Remove tree fraction if fractions from other modeled cover is already >95%
    nontree = np.vstack((water_corrected,barren_corrected,herb_corrected,shrub_corrected))
    totals_nontree = nontree.sum(axis=0)
    trees_corrected[totals_nontree>=0.95] = 0
    
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
    print(row['train_img_name'])
    assess_pts, assess_pred, assess_pixels = sample_bias(traindir+row['ecoreg_code']+'/training_imgs/'+row['train_img_name'],
                                                    traindir+row['ecoreg_code']+'/training_imgs/'+row['train_img_name'][:-10]+'.tif',
                                                    all_norm.reshape(5,nx,ny),
                                                    model_traindir+row['train_img_name'][:-4]+'_ids.txt',
                                                    row['nclasses'],
                                                    row['nsamples'])
    
    if i==0:
        all_assess_pts = assess_pts
        all_assess_pred = assess_pred
        all_used_pixels = [assess_pixels]
    else:
        all_assess_pts = np.hstack((all_assess_pts, assess_pts))
        all_assess_pred = np.hstack((all_assess_pred, assess_pred))
        all_used_pixels.append(assess_pixels)
    
    temp_assess_pts = np.delete(all_assess_pts, np.where((all_assess_pts==32767).any(axis=0))[0], axis=1)
    temp_assess_pred = np.delete(all_assess_pred, np.where((all_assess_pts==32767).any(axis=0))[0], axis=1)
    final_assess_pts = np.delete(temp_assess_pts, np.where((temp_assess_pred==32767).any(axis=0))[0], axis=1)
    final_assess_pred = np.delete(temp_assess_pred, np.where((temp_assess_pred==32767).any(axis=0))[0], axis=1)

# Scale
final_assess_pts = final_assess_pts*100

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
ax3.set_title('Low vegetation fractional cover', fontsize=32)
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


#%%
# 3) Run S1 models, bias correction, and post-processing on training images and assess 
#    against an the sample sample of pixels

# Read in the S1 models
water_regressor = pickle.load('/projectnb/modislc/users/seamorez/HLS_FCover/model_training/water_rfr_s1.pickle')
barren_regressor = pickle.load('/projectnb/modislc/users/seamorez/HLS_FCover/model_training/barren_rfr_s1.pickle')
herb_regressor = pickle.load('/projectnb/modislc/users/seamorez/HLS_FCover/model_training/herb_rfr_s1.pickle')
shrub_regressor = pickle.load('/projectnb/modislc/users/seamorez/HLS_FCover/model_training/shrub_rfr_s1.pickle')
trees_regressor = pickle.load('/projectnb/modislc/users/seamorez/HLS_FCover/model_training/trees_rfr_s1.pickle')
reg0 = pickle.load('/projectnb/modislc/users/seamorez/HLS_FCover/model_training/water_bias_s1.pickle')
reg1 = pickle.load('/projectnb/modislc/users/seamorez/HLS_FCover/model_training/barren_bias_s1.pickle')
reg2 = pickle.load('/projectnb/modislc/users/seamorez/HLS_FCover/model_training/herb_bias_s1.pickle')
reg3 = pickle.load('/projectnb/modislc/users/seamorez/HLS_FCover/model_training/shrub_bias_s1.pickle')
reg4 = pickle.load('/projectnb/modislc/users/seamorez/HLS_FCover/model_training/trees_bias_s1.pickle')

# Predict FCover on training images
for i, row in train_key.iterrows():
    # Read the 30 m map
    with rio.open(traindir+row['ecoreg_code']+'/training_imgs/'+row['train_img_name']) as src:        
        # Read the feats and clip to map
        with rio.open(outdir+str(row['year'])+'/'+row['above_tile']+'/feats_full_s1_'+str(row['year'])+'.tif') as featsrc:   
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
    water_corrected = water_predict - reg0.predict(water_predict.reshape(-1,1))
    barren_corrected = barren_predict - reg1.predict(barren_predict.reshape(-1,1))
    herb_corrected = herb_predict - reg2.predict(herb_predict.reshape(-1,1))
    shrub_corrected = shrub_predict - reg3.predict(shrub_predict.reshape(-1,1))
    trees_corrected = trees_predict - reg4.predict(trees_predict.reshape(-1,1))
    
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
    # Remove tree fraction if fractions from other modeled cover is already >95%
    nontree = np.vstack((water_corrected,barren_corrected,herb_corrected,shrub_corrected))
    totals_nontree = nontree.sum(axis=0)
    trees_corrected[totals_nontree>=0.95] = 0
    
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
    print(row['train_img_name'])
    assess_pts, assess_pred, assess_pixels = get_assess(traindir+row['ecoreg_code']+'/training_imgs/'+row['train_img_name'],
                                                        all_norm.reshape(5,nx,ny),
                                                        all_used_pixels[i])
    
    if i==0:
        all_assess_pts = assess_pts
        all_assess_pred = assess_pred
    else:
        all_assess_pts = np.hstack((all_assess_pts, assess_pts))
        all_assess_pred = np.hstack((all_assess_pred, assess_pred))
    
    temp_assess_pts = np.delete(all_assess_pts, np.where((all_assess_pts==32767).any(axis=0))[0], axis=1)
    temp_assess_pred = np.delete(all_assess_pred, np.where((all_assess_pts==32767).any(axis=0))[0], axis=1)
    final_assess_pts = np.delete(temp_assess_pts, np.where((temp_assess_pred==32767).any(axis=0))[0], axis=1)
    final_assess_pred = np.delete(temp_assess_pred, np.where((temp_assess_pred==32767).any(axis=0))[0], axis=1)

# Scale
final_assess_pts = final_assess_pts*100

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
ax3.set_title('Low vegetation fractional cover', fontsize=32)
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