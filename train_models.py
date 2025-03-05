import glob
import numpy as np
import pandas as pd
import rasterio
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import pickle
import matplotlib.pyplot as plt
import seaborn as sb

#### PARAMETERS TO BE PASSED IN ####
outdir = '/projectnb/modislc/users/seamorez/HLS_FCover/output/'
indir = '/projectnb/modislc/users/seamorez/HLS_FCover/input/HLS30/'
traindir = '/projectnb/modislc/users/seamorez/HLS_FCover/Maxar_NAtundra/training/'
model_traindir = '/projectnb/modislc/users/seamorez/HLS_FCover/model_training/'
model_outdir = '/projectnb/modislc/users/seamorez/HLS_FCover/model_outputs/'
start_yr = 2016
end_yr = 2022
phenometrics = ['50PCGI','Peak','50PCGD']
metrics = ['dem', 'water', 'slope', 'aspect']

#### PARAMETERS ####
years = np.arange(start_yr, end_yr+1, 1)
train_key = pd.read_csv(traindir+'training_key.csv')

################################### STEP 1 ###################################
# Model training v2
def strat_sample_training(class_path, nclasses, stack_path, n):
    '''
    Generates random sample of pixels, returning an array of the fractional 
    classifications for each class and the full feature stack at each pixel for 
    n pixels (specified by user)

    Parameters
    ----------
    class_path: fractional training image
    nclasses: number of classes in each training image
    stack_path: features image sharing overlap training image
    n: number of training pixels to sample

    Returns
    -------
    training points
    training features
    row_ids: indices for the sampled points 

    '''
    class_30m = rasterio.open(traindir+class_path,'r')
    full_stack = rasterio.open(outdir+stack_path,'r')
    class_30m_img = class_30m.read()
    # Clip feature stack to classified Maxar image
    window = rasterio.windows.from_bounds(*class_30m.bounds, transform=full_stack.transform)
    full_stack_clipped = full_stack.read(window=window, boundless=True)
    # Reshape to 2D array
    nsamples, nx, ny = full_stack_clipped.shape 
    full_stack_clipped_2d = full_stack_clipped.reshape((nsamples,nx*ny))
    full_stack_clipped_2d = full_stack_clipped_2d.T
    # Reshape to 2D array
    nsamples1, nx1, ny1 = class_30m_img.shape 
    class_30m_img_2d = class_30m_img.reshape((nsamples1,nx1*ny1))
    class_30m_img_2d = class_30m_img_2d.T
    # Column join the features to the classifications
    class_feats = np.append(class_30m_img_2d, full_stack_clipped_2d, axis=1)
    # Remove rows with no-data values in the classified image (ndval=32767) and in the feature stack (ndval=0)
    #class_feats = class_feats[(class_feats[:,0]!=32767) & (class_feats[:,0]>0)]
    #class_feats = class_feats[class_feats[:,nclasses:]!=32767]
    class_feats = class_feats[np.where(((class_feats[:,0:nclasses]!=32767).all(1)) & 
                                       ((class_feats[:,0:nclasses]>=0).all(1)) &
                                       ((class_feats[:,nclasses:]!=32767).all(1)))]
    
    # Remove rows where features have nan values
    class_feats = class_feats[~np.any(class_feats==32767, axis=1)]
    
    # Random sample
    row_ids = random.sample(range(0, class_feats.shape[0]-1), n)
    rand_sample = class_feats[row_ids, :]
    
    # Return 2D array of classifications (4 x nsamples) and the corresponding features (nsamples x nfeatures)
    return rand_sample[:,0:nclasses].T, rand_sample[:,nclasses:], row_ids

# Model training
def sample_training(class_path, nclasses, stack_path, n):
    '''
    Generates random sample of pixels, returning an array of the fractional 
    classifications for each class and the full feature stack at each pixel for 
    n pixels (specified by user)

    Parameters
    ----------
    class_path: fractional training image
    nclasses: number of classes in each training image
    stack_path: features image sharing overlap training image
    n: number of training pixels to sample

    Returns
    -------
    training points
    training features
    row_ids: indices for the sampled points 

    '''
    class_30m = rasterio.open(traindir+class_path,'r')
    full_stack = rasterio.open(outdir+stack_path,'r')
    class_30m_img = class_30m.read()
    # Clip feature stack to classified Maxar image
    window = rasterio.windows.from_bounds(*class_30m.bounds, transform=full_stack.transform)
    full_stack_clipped = full_stack.read(window=window, boundless=True)
    # Reshape to 2D array
    nsamples, nx, ny = full_stack_clipped.shape 
    full_stack_clipped_2d = full_stack_clipped.reshape((nsamples,nx*ny))
    full_stack_clipped_2d = full_stack_clipped_2d.T
    # Reshape to 2D array
    nsamples1, nx1, ny1 = class_30m_img.shape 
    class_30m_img_2d = class_30m_img.reshape((nsamples1,nx1*ny1))
    class_30m_img_2d = class_30m_img_2d.T
    # Column join the features to the classifications
    class_feats = np.append(class_30m_img_2d, full_stack_clipped_2d, axis=1)
    # Remove rows with no-data values in the classified image (ndval=32767) and in the feature stack (ndval=0)
    #class_feats = class_feats[(class_feats[:,0]!=32767) & (class_feats[:,0]>0)]
    #class_feats = class_feats[class_feats[:,nclasses:]!=32767]
    class_feats = class_feats[np.where(((class_feats[:,0:nclasses]!=32767).all(1)) & 
                                       ((class_feats[:,0:nclasses]>=0).all(1)) &
                                       ((class_feats[:,nclasses:]!=32767).all(1)))]
    
    # Remove rows where features have nan values
    class_feats = class_feats[~np.any(class_feats==32767, axis=1)]
    
    # Random sample
    row_ids = random.sample(range(0, class_feats.shape[0]-1), n)
    rand_sample = class_feats[row_ids, :]
    
    # Return 2D array of classifications (4 x nsamples) and the corresponding features (nsamples x nfeatures)
    return rand_sample[:,0:nclasses].T, rand_sample[:,nclasses:], row_ids


# Load training images along with overlapping feature stacks
# Map features
for i, row in train_key.iterrows():
    train_pts, train_feats, ids = sample_training(row['ecoreg_code']+'/training_imgs/'+row['train_img_name'], 
                                                  row['nclasses'],
                                                  str(row['year'])+'/'+row['above_tile']+'/feats_full_'+str(row['year'])+'.tif',
                                                  row['nsamples'])
    if row['nclasses']==4:
        trees = np.zeros(train_pts.shape[1])
        print(trees.shape)
        train_pts = np.vstack((train_pts, trees))
        
    print(train_pts.shape)
    
    if i==0: 
        train_pts_full = train_pts; train_feats_full = train_feats; ids_full = ids
    else: 
        train_pts_full = np.append(train_pts_full, train_pts, axis=1)
        train_feats_full = np.append(train_feats_full, train_feats, axis=0)
        ids_full = np.append(ids_full, ids, axis=0)
        print(train_feats_full.shape, ids_full.shape)

plt.hist(train_pts_full[4,:])

# Train models: four regressors, one for each class
water_regressor = RandomForestRegressor(n_estimators=500, min_samples_leaf=5, random_state=7).fit(train_feats_full, train_pts_full[0])
barren_regressor = RandomForestRegressor(n_estimators=500, min_samples_leaf=5, random_state=7).fit(train_feats_full, train_pts_full[1])
herb_regressor = RandomForestRegressor(n_estimators=500, min_samples_leaf=5, random_state=7).fit(train_feats_full, train_pts_full[2])
shrub_regressor = RandomForestRegressor(n_estimators=500, min_samples_leaf=5, random_state=7).fit(train_feats_full, train_pts_full[3])
trees_regressor = RandomForestRegressor(n_estimators=500, min_samples_leaf=5, random_state=7).fit(train_feats_full, train_pts_full[4])

# Save models, training images, training features, and IDs
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
np.savetxt(model_traindir+'train_ids.txt', ids_full, delimiter=',')


################################### STEP 2 ###################################
# Bias correction
def sample_for_bias(class_path, nclasses, modeled_path, stack_path, n, ids):
    class_30m = rasterio.open(traindir+class_path,'r')
    modeled = rasterio.open(model_traindir+modeled_path,'r')
    full_stack = rasterio.open(outdir+stack_path,'r')
    class_30m_img = class_30m.read()
    # Clip feature stack to classified Maxar image
    window = rasterio.windows.from_bounds(*class_30m.bounds, transform=modeled.transform)
    modeled_clipped = modeled.read(window=window, boundless=True)
    full_stack_clipped = full_stack.read(window=window, boundless=True)
    # Reshape to 2D array
    nsamples, nx, ny = modeled_clipped.shape 
    modeled_clipped_2d = modeled_clipped.reshape((nsamples,nx*ny))
    modeled_clipped_2d = modeled_clipped_2d.T
    modeled_clipped_2d = modeled_clipped_2d.astype(float)/10000
    # Reshape to 2D array
    nsamples0, nx0, ny0 = full_stack_clipped.shape
    full_stack_clipped_2d = full_stack_clipped.reshape((nsamples0,nx0*ny0))
    full_stack_clipped_2d = full_stack_clipped_2d.T
    # Reshape to 2D array
    nsamples1, nx1, ny1 = class_30m_img.shape 
    class_30m_img_2d = class_30m_img.reshape((nsamples1,nx1*ny1))
    class_30m_img_2d = class_30m_img_2d.T
    if nclasses<5:
        trees = np.zeros((nx*ny,1))
        class_30m_img_2d = np.concatenate([class_30m_img_2d, trees], axis=1)
    # Column join the modeled to the training
    comparison = np.append(class_30m_img_2d, modeled_clipped_2d, axis=1)
    # Remove rows with no-data values in the classified image (ndval=32767) and in the modeled image
    comparison = comparison[np.where(((comparison[:,0:nclasses]!=32767).all(1)) & 
                                       ((comparison[:,0:nclasses]>=0).all(1)) &
                                       ((full_stack_clipped_2d[:,:]!=32767).all(1)))]
    # Remove rows that were used in training
    train_sample = comparison[ids,:]
    print(train_sample.shape)
    bias_corr_sample = np.delete(comparison, ids, axis=0)
    # Remove rows where prediction is nan
    bias_corr_sample = bias_corr_sample[~np.any(bias_corr_sample==3.2767, axis=1)]
    
    # Random sample for bias correction
    row_ids = random.sample(range(0, bias_corr_sample.shape[0]-1), n)
    rand_sample = bias_corr_sample[row_ids, :]
    
    return train_sample, rand_sample, row_ids


# First, model for the images you need for bias correction
for i, row in train_key.iterrows():
    #if (i>1): continue
    full_stack = rasterio.open(outdir+str(row['year'])+'/'+row['above_tile']+
                               '/feats_full_'+str(row['year'])+'.tif','r')
    meta = full_stack.meta
    full_stack_img = full_stack.read()
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
    water_predict = np.around(water_predict*10000)
    barren_predict = np.around(barren_predict*10000)
    herb_predict = np.around(herb_predict*10000)
    shrub_predict = np.around(shrub_predict*10000)
    trees_predict = np.around(trees_predict*10000)
    
    # Reshape
    water_predict = water_predict.reshape(1, nx*ny)
    barren_predict = barren_predict.reshape(1, nx*ny)
    herb_predict = herb_predict.reshape(1, nx*ny)
    shrub_predict = shrub_predict.reshape(1, nx*ny)
    trees_predict = trees_predict.reshape(1, nx*ny)

    full_stack_band = full_stack_img[10].reshape(1, nx*ny)
    water_predict[full_stack_band==32767] = 32767
    barren_predict[full_stack_band==32767] = 32767
    herb_predict[full_stack_band==32767] = 32767
    shrub_predict[full_stack_band==32767] = 32767
    trees_predict[full_stack_band==32767] = 32767

    # Reshape
    water_predict = water_predict.reshape(nx, ny)
    barren_predict = barren_predict.reshape(nx, ny)
    herb_predict = herb_predict.reshape(nx, ny)
    shrub_predict = shrub_predict.reshape(nx, ny)
    trees_predict = trees_predict.reshape(nx, ny)

    # Saving normalized results as bands in one image
    meta2 = meta
    meta2.update(dtype='int16', nodata=32767, count=5)

    results_list = [water_predict,barren_predict,herb_predict,shrub_predict,trees_predict]

    with rasterio.open(model_traindir+'FCover_'+str(row['year'])+'_'+row['above_tile']+'_rfr.tif', 'w', **meta2) as dst:
        for id,row in enumerate(results_list, start=1):
            print(row[row<0])
            dst.write_band(id, row)    

# Second, do bias correction
for i, row in train_key.iterrows():
    # Get trraining indices
    indices = np.loadtxt(model_traindir+'train_ids.txt')[i*row['nsamples']:i*row['nsamples']+row['nsamples']]
    print(indices.shape)
    train_sample, bias_corr_sample, bias_corr_ids = sample_for_bias(row['ecoreg_code']+'/training_imgs/'+row['train_img_name'], 
                                                      row['nclasses'],
                                                      'FCover_'+str(row['year'])+'_'+row['above_tile']+'_rfr.tif',
                                                      str(row['year'])+'/'+row['above_tile']+'/feats_full_'+str(row['year'])+'.tif',
                                                      row['nsamples'],
                                                      indices.astype(int))
    
    if i==0: 
        train_full = train_sample; bias_full = bias_corr_sample; bias_ids_full = bias_corr_ids
    else: 
        train_full = np.append(train_full, train_sample, axis=0)
        bias_full = np.append(bias_full, bias_corr_sample, axis=0)
        bias_ids_full = np.append(bias_ids_full, bias_corr_ids, axis=0)
        print(train_full.shape, bias_full.shape)
    
np.savetxt(model_traindir+'bias_ids.txt', bias_ids_full, delimiter=',')

# Training scatterplots
fig, ((ax1, ax2, ax5), (ax3, ax4, ax6)) = plt.subplots(2, 3)
fig.set_size_inches(27, 18)

#ax1.scatter(all_class[:,0],all_results[:,0], color='Blue', alpha=0.2)
sb.regplot(x=train_full[:,0],y=train_full[:,5], color='#0071DC', scatter_kws={'alpha':0.2}, ax=ax1)
ax1.text(.05, .95, '$R^2$ = '+str(np.corrcoef(train_full[:,0],train_full[:,5])[0,1]**2)[:6], fontsize=14)
ax1.text(.05, .9, 'MAE = '+str(mean_absolute_error(train_full[:,0],train_full[:,5]))[:6], fontsize=14)
ax1.text(.05, .85, 'RMSE = '+str(np.sqrt(mean_squared_error(train_full[:,0],train_full[:,5])))[:6], fontsize=14)
ax1.plot([0,1],[0,1], color='Black', linewidth=.5)
ax1.set_xlim(0,1)
ax1.set_ylim(0,1)
ax1.set_title('Water fractional cover',fontsize=20)
ax1.set_xlabel('Observed',fontsize=14)
ax1.set_ylabel('Predicted',fontsize=14)

#ax2.scatter(all_class[:,1],all_results[:,1], color='Grey', alpha=0.2)
sb.regplot(x=train_full[:,1],y=train_full[:,6], color='Grey', scatter_kws={'alpha':0.2}, ax=ax2)
ax2.text(.05, .95, '$R^2$ = '+str(np.corrcoef(train_full[:,1],train_full[:,6])[0,1]**2)[:6], fontsize=14)
ax2.text(.05, .9, 'MAE = '+str(mean_absolute_error(train_full[:,1],train_full[:,6]))[:6], fontsize=14)
ax2.text(.05, .85, 'RMSE = '+str(np.sqrt(mean_squared_error(train_full[:,1],train_full[:,6])))[:6], fontsize=14)
ax2.plot([0,1],[0,1], color='Black', linewidth=.5)
ax2.set_xlim(0,1)
ax2.set_ylim(0,1)
ax2.set_title('Barren fractional cover',fontsize=20)
ax2.set_xlabel('Observed',fontsize=14)
ax2.set_ylabel('Predicted',fontsize=14)

#ax3.scatter(all_class[:,2],all_results[:,2], color='YellowGreen', alpha=0.2)
sb.regplot(x=train_full[:,2],y=train_full[:,7], color='YellowGreen', scatter_kws={'alpha':0.2}, ax=ax3)
ax3.text(.05, .95, '$R^2$ = '+str(np.corrcoef(train_full[:,2],train_full[:,7])[0,1]**2)[:6], fontsize=14)
ax3.text(.05, .9, 'MAE = '+str(mean_absolute_error(train_full[:,2],train_full[:,7]))[:6], fontsize=14)
ax3.text(.05, .85, 'RMSE = '+str(np.sqrt(mean_squared_error(train_full[:,2],train_full[:,7])))[:6], fontsize=14)
ax3.plot([0,1],[0,1], color='Black', linewidth=.5)
ax3.set_xlim(0,1)
ax3.set_ylim(0,1)
ax3.set_title('Herbaceous fractional cover',fontsize=20)
ax3.set_xlabel('Observed',fontsize=14)
ax3.set_ylabel('Predicted',fontsize=14)

#ax4.scatter(all_class[:,3],all_results[:,3], color='Green', alpha=0.2)
sb.regplot(x=train_full[:,3],y=train_full[:,8], color='Green', scatter_kws={'alpha':0.2}, ax=ax4)
ax4.text(.05, .95, '$R^2$ = '+str(np.corrcoef(train_full[:,3],train_full[:,8])[0,1]**2)[:6], fontsize=14)
ax4.text(.05, .9, 'MAE = '+str(mean_absolute_error(train_full[:,3],train_full[:,8]))[:6], fontsize=14)
ax4.text(.05, .85, 'RMSE = '+str(np.sqrt(mean_squared_error(train_full[:,3],train_full[:,8])))[:6], fontsize=14)
ax4.plot([0,1],[0,1], color='Black', linewidth=.5)
ax4.set_xlim(0,1)
ax4.set_ylim(0,1)
ax4.set_title('Shrub fractional cover',fontsize=20)
ax4.set_xlabel('Observed',fontsize=14)
ax4.set_ylabel('Predicted',fontsize=14)

#ax5.scatter(all_class[:,3],all_results[:,3], color='Green', alpha=0.2)
sb.regplot(x=train_full[:,4],y=train_full[:,9], color='darkgreen', scatter_kws={'alpha':0.2}, ax=ax5)
ax5.text(.05, .95, '$R^2$ = '+str(np.corrcoef(train_full[:,4],train_full[:,9])[0,1]**2)[:6], fontsize=14)
ax5.text(.05, .9, 'MAE = '+str(mean_absolute_error(train_full[:,4],train_full[:,9]))[:6], fontsize=14)
ax5.text(.05, .85, 'RMSE = '+str(np.sqrt(mean_squared_error(train_full[:,4],train_full[:,9])))[:6], fontsize=14)
ax5.plot([0,1],[0,1], color='Black', linewidth=.5)
ax5.set_xlim(0,1)
ax5.set_ylim(0,1)
ax5.set_title('Trees fractional cover',fontsize=20)
ax5.set_xlabel('Observed',fontsize=14)
ax5.set_ylabel('Predicted',fontsize=14)

# Bias predicted vs observed scatterplots
fig, ((ax1, ax2, ax5), (ax3, ax4, ax6)) = plt.subplots(2, 3)
fig.set_size_inches(27, 18)

#ax1.scatter(all_class[:,0],all_results[:,0], color='Blue', alpha=0.2)
sb.regplot(x=bias_full[:,0],y=bias_full[:,5], color='#0071DC', scatter_kws={'alpha':0.2}, ax=ax1)
ax1.text(.05, .95, '$R^2$ = '+str(np.corrcoef(bias_full[:,0],bias_full[:,5])[0,1]**2)[:6], fontsize=14)
ax1.text(.05, .9, 'MAE = '+str(mean_absolute_error(bias_full[:,0],bias_full[:,5]))[:6], fontsize=14)
ax1.text(.05, .85, 'RMSE = '+str(np.sqrt(mean_squared_error(bias_full[:,0],bias_full[:,5])))[:6], fontsize=14)
ax1.plot([0,1],[0,1], color='Black', linewidth=.5)
ax1.set_xlim(0,1)
ax1.set_ylim(0,1)
ax1.set_title('Water fractional cover',fontsize=20)
ax1.set_xlabel('Observed',fontsize=14)
ax1.set_ylabel('Predicted',fontsize=14)

#ax2.scatter(all_class[:,1],all_results[:,1], color='Grey', alpha=0.2)
sb.regplot(x=bias_full[:,1],y=bias_full[:,6], color='Grey', scatter_kws={'alpha':0.2}, ax=ax2)
ax2.text(.05, .95, '$R^2$ = '+str(np.corrcoef(bias_full[:,1],bias_full[:,6])[0,1]**2)[:6], fontsize=14)
ax2.text(.05, .9, 'MAE = '+str(mean_absolute_error(bias_full[:,1],bias_full[:,6]))[:6], fontsize=14)
ax2.text(.05, .85, 'RMSE = '+str(np.sqrt(mean_squared_error(bias_full[:,1],bias_full[:,6])))[:6], fontsize=14)
ax2.plot([0,1],[0,1], color='Black', linewidth=.5)
ax2.set_xlim(0,1)
ax2.set_ylim(0,1)
ax2.set_title('Barren fractional cover',fontsize=20)
ax2.set_xlabel('Observed',fontsize=14)
ax2.set_ylabel('Predicted',fontsize=14)

#ax3.scatter(all_class[:,2],all_results[:,2], color='YellowGreen', alpha=0.2)
sb.regplot(x=bias_full[:,2],y=bias_full[:,7], color='YellowGreen', scatter_kws={'alpha':0.2}, ax=ax3)
ax3.text(.05, .95, '$R^2$ = '+str(np.corrcoef(bias_full[:,2],bias_full[:,7])[0,1]**2)[:6], fontsize=14)
ax3.text(.05, .9, 'MAE = '+str(mean_absolute_error(bias_full[:,2],bias_full[:,7]))[:6], fontsize=14)
ax3.text(.05, .85, 'RMSE = '+str(np.sqrt(mean_squared_error(bias_full[:,2],bias_full[:,7])))[:6], fontsize=14)
ax3.plot([0,1],[0,1], color='Black', linewidth=.5)
ax3.set_xlim(0,1)
ax3.set_ylim(0,1)
ax3.set_title('Herbaceous fractional cover',fontsize=20)
ax3.set_xlabel('Observed',fontsize=14)
ax3.set_ylabel('Predicted',fontsize=14)

#ax4.scatter(all_class[:,3],all_results[:,3], color='Green', alpha=0.2)
sb.regplot(x=bias_full[:,3],y=bias_full[:,8], color='Green', scatter_kws={'alpha':0.2}, ax=ax4)
ax4.text(.05, .95, '$R^2$ = '+str(np.corrcoef(bias_full[:,3],bias_full[:,8])[0,1]**2)[:6], fontsize=14)
ax4.text(.05, .9, 'MAE = '+str(mean_absolute_error(bias_full[:,3],bias_full[:,8]))[:6], fontsize=14)
ax4.text(.05, .85, 'RMSE = '+str(np.sqrt(mean_squared_error(bias_full[:,3],bias_full[:,8])))[:6], fontsize=14)
ax4.plot([0,1],[0,1], color='Black', linewidth=.5)
ax4.set_xlim(0,1)
ax4.set_ylim(0,1)
ax4.set_title('Shrub fractional cover',fontsize=20)
ax4.set_xlabel('Observed',fontsize=14)
ax4.set_ylabel('Predicted',fontsize=14)

#ax5.scatter(all_class[:,3],all_results[:,3], color='Green', alpha=0.2)
sb.regplot(x=bias_full[:,4],y=bias_full[:,9], color='darkgreen', scatter_kws={'alpha':0.2}, ax=ax5)
ax5.text(.05, .95, '$R^2$ = '+str(np.corrcoef(bias_full[:,4],bias_full[:,9])[0,1]**2)[:6], fontsize=14)
ax5.text(.05, .9, 'MAE = '+str(mean_absolute_error(bias_full[:,4],bias_full[:,9]))[:6], fontsize=14)
ax5.text(.05, .85, 'RMSE = '+str(np.sqrt(mean_squared_error(bias_full[:,4],bias_full[:,9])))[:6], fontsize=14)
ax5.plot([0,1],[0,1], color='Black', linewidth=.5)
ax5.set_xlim(0,1)
ax5.set_ylim(0,1)
ax5.set_title('Trees fractional cover',fontsize=20)
ax5.set_xlabel('Observed',fontsize=14)
ax5.set_ylabel('Predicted',fontsize=14)

# Bias scatterplots
fig, ((ax1, ax2, ax5), (ax3, ax4, ax6)) = plt.subplots(2, 3)
fig.set_size_inches(27, 18)

#ax1.scatter(all_class[:,0],all_results[:,0], color='Blue', alpha=0.2)
sb.regplot(y=bias_full[:,5]-bias_full[:,0],x=bias_full[:,0], color='#0071DC', scatter_kws={'alpha':0.2}, ax=ax1)
ax1.set_xlim(0,1)
ax1.set_ylim(-1,1)
ax1.set_title('Water fractional cover',fontsize=20)
ax1.set_xlabel('Observed',fontsize=14)
ax1.set_ylabel('Predicted-Observed',fontsize=14)

#ax2.scatter(all_class[:,1],all_results[:,1], color='Grey', alpha=0.2)
sb.regplot(y=bias_full[:,6]-bias_full[:,1],x=bias_full[:,1], color='Grey', scatter_kws={'alpha':0.2}, ax=ax2)
ax2.set_xlim(0,1)
ax2.set_ylim(-1,1)
ax2.set_title('Barren fractional cover',fontsize=20)
ax2.set_xlabel('Observed',fontsize=14)
ax2.set_ylabel('Predicted-Observed',fontsize=14)

#ax3.scatter(all_class[:,2],all_results[:,2], color='YellowGreen', alpha=0.2)
sb.regplot(y=bias_full[:,7]-bias_full[:,2],x=bias_full[:,2], color='YellowGreen', scatter_kws={'alpha':0.2}, ax=ax3)
ax3.set_xlim(0,1)
ax3.set_ylim(-1,1)
ax3.set_title('Herbaceous fractional cover',fontsize=20)
ax3.set_xlabel('Observed',fontsize=14)
ax3.set_ylabel('Predicted-Observed',fontsize=14)

#ax4.scatter(all_class[:,3],all_results[:,3], color='Green', alpha=0.2)
sb.regplot(y=bias_full[:,8]-bias_full[:,3],x=bias_full[:,3], color='Green', scatter_kws={'alpha':0.2}, ax=ax4)
ax4.set_xlim(0,1)
ax4.set_ylim(-1,1)
ax4.set_title('Shrub fractional cover',fontsize=20)
ax4.set_xlabel('Observed',fontsize=14)
ax4.set_ylabel('Predicted-Observed',fontsize=14)

#ax5.scatter(all_class[:,3],all_results[:,3], color='Green', alpha=0.2)
sb.regplot(y=bias_full[:,9]-bias_full[:,4],x=bias_full[:,4], color='darkgreen', scatter_kws={'alpha':0.2}, ax=ax5)
ax5.set_xlim(0,1)
ax5.set_ylim(-1,1)
ax5.set_title('Trees fractional cover',fontsize=20)
ax5.set_xlabel('Observed',fontsize=14)
ax5.set_ylabel('Predicted-Observed',fontsize=14)

# Test bias correction with herbaceous
reg = LinearRegression().fit(bias_full[:,7].reshape(4000,1), bias_full[:,2])
corrected = reg.predict(bias_full[:,7].reshape(4000,1))

fig, ((ax1, ax2, ax5), (ax3, ax4, ax6)) = plt.subplots(2, 3)
fig.set_size_inches(27, 18)
sb.regplot(x=bias_full[:,2],y=bias_full[:,7], color='YellowGreen', scatter_kws={'alpha':0.2}, ax=ax3)
ax3.plot([0,1],[0,1], color='Black', linewidth=.5)
sb.regplot(y=corrected,x=bias_full[:,2], color='mediumvioletred', scatter_kws={'alpha':0.2}, ax=ax3)
ax3.text(.05, .95, '$R^2$ = '+str(np.corrcoef(bias_full[:,2],corrected)[0,1]**2)[:6], fontsize=14)
ax3.text(.05, .9, 'MAE = '+str(mean_absolute_error(bias_full[:,2],corrected))[:6], fontsize=14)
ax3.text(.05, .85, 'RMSE = '+str(np.sqrt(mean_squared_error(bias_full[:,2],corrected)))[:6], fontsize=14)
ax3.set_xlim(0,1)
ax3.set_ylim(0,1)

# Test bias correction with herbaceous
reg0 = LinearRegression().fit(bias_full[:,0].reshape(4000,1), bias_full[:,5]-bias_full[:,0])
corrected0 = bias_full[:,5] - reg0.predict(bias_full[:,5].reshape(4000,1))
#corrected0[corrected0>1]=1; corrected0[corrected0<0]=0
reg1 = LinearRegression().fit(bias_full[:,1].reshape(4000,1), bias_full[:,6]-bias_full[:,1])
corrected1 = bias_full[:,6] - reg1.predict(bias_full[:,6].reshape(4000,1))
#corrected1[corrected1>1]=1; corrected1[corrected1<0]=0
reg2 = LinearRegression().fit(bias_full[:,2].reshape(4000,1), bias_full[:,7]-bias_full[:,2])
corrected2 = bias_full[:,7] - reg2.predict(bias_full[:,7].reshape(4000,1))
#corrected[corrected>1]=1; corrected[corrected<0]=0
reg3 = LinearRegression().fit(bias_full[:,3].reshape(4000,1), bias_full[:,8]-bias_full[:,3])
corrected3 = bias_full[:,8] - reg3.predict(bias_full[:,8].reshape(4000,1))
#corrected3[corrected3>1]=1; corrected3[corrected3<0]=0
reg4 = LinearRegression().fit(bias_full[:,4].reshape(4000,1), bias_full[:,9]-bias_full[:,4])
corrected4 = bias_full[:,9] - reg4.predict(bias_full[:,9].reshape(4000,1))
#corrected4[corrected4>1]=1; corrected4[corrected4<0]=0

fig, ((ax1, ax2, ax5), (ax3, ax4, ax6)) = plt.subplots(2, 3)
fig.set_size_inches(27, 18)

sb.regplot(x=bias_full[:,0],y=bias_full[:,5], color='#0071DC', scatter_kws={'alpha':0.2}, ax=ax1)
ax1.plot([0,1],[0,1], color='Black', linewidth=.5)
sb.regplot(y=corrected0,x=bias_full[:,0], color='mediumvioletred', scatter_kws={'alpha':0.2}, ax=ax1)
ax1.text(.05, .95, '$R^2$ = '+str(np.corrcoef(bias_full[:,0],corrected0)[0,1]**2)[:6], fontsize=14)
ax1.text(.05, .9, 'MAE = '+str(mean_absolute_error(bias_full[:,0],corrected0))[:6], fontsize=14)
ax1.text(.05, .85, 'RMSE = '+str(np.sqrt(mean_squared_error(bias_full[:,0],corrected0)))[:6], fontsize=14)
ax1.set_xlim(0,1)
ax1.set_ylim(0,1)
sb.regplot(x=bias_full[:,1],y=bias_full[:,6], color='Grey', scatter_kws={'alpha':0.2}, ax=ax2)
ax2.plot([0,1],[0,1], color='Black', linewidth=.5)
sb.regplot(y=corrected1,x=bias_full[:,1], color='mediumvioletred', scatter_kws={'alpha':0.2}, ax=ax2)
ax2.text(.05, .95, '$R^2$ = '+str(np.corrcoef(bias_full[:,1],corrected1)[0,1]**2)[:6], fontsize=14)
ax2.text(.05, .9, 'MAE = '+str(mean_absolute_error(bias_full[:,1],corrected1))[:6], fontsize=14)
ax2.text(.05, .85, 'RMSE = '+str(np.sqrt(mean_squared_error(bias_full[:,1],corrected1)))[:6], fontsize=14)
ax2.set_xlim(0,1)
ax2.set_ylim(0,1)
sb.regplot(x=bias_full[:,2],y=bias_full[:,7], color='YellowGreen', scatter_kws={'alpha':0.2}, ax=ax3)
ax3.plot([0,1],[0,1], color='Black', linewidth=.5)
sb.regplot(y=corrected2,x=bias_full[:,2], color='mediumvioletred', scatter_kws={'alpha':0.2}, ax=ax3)
ax3.text(.05, .95, '$R^2$ = '+str(np.corrcoef(bias_full[:,2],corrected2)[0,1]**2)[:6], fontsize=14)
ax3.text(.05, .9, 'MAE = '+str(mean_absolute_error(bias_full[:,2],corrected2))[:6], fontsize=14)
ax3.text(.05, .85, 'RMSE = '+str(np.sqrt(mean_squared_error(bias_full[:,2],corrected2)))[:6], fontsize=14)
ax3.set_xlim(0,1)
ax3.set_ylim(0,1)
sb.regplot(x=bias_full[:,3],y=bias_full[:,8], color='Green', scatter_kws={'alpha':0.2}, ax=ax4)
ax4.plot([0,1],[0,1], color='Black', linewidth=.5)
sb.regplot(y=corrected3,x=bias_full[:,3], color='mediumvioletred', scatter_kws={'alpha':0.2}, ax=ax4)
ax4.text(.05, .95, '$R^2$ = '+str(np.corrcoef(bias_full[:,3],corrected3)[0,1]**2)[:6], fontsize=14)
ax4.text(.05, .9, 'MAE = '+str(mean_absolute_error(bias_full[:,3],corrected3))[:6], fontsize=14)
ax4.text(.05, .85, 'RMSE = '+str(np.sqrt(mean_squared_error(bias_full[:,3],corrected3)))[:6], fontsize=14)
ax4.set_xlim(0,1)
ax4.set_ylim(0,1)
sb.regplot(x=bias_full[:,4],y=bias_full[:,9], color='darkgreen', scatter_kws={'alpha':0.2}, ax=ax5)
ax5.plot([0,1],[0,1], color='Black', linewidth=.5)
sb.regplot(y=corrected4,x=bias_full[:,4], color='mediumvioletred', scatter_kws={'alpha':0.2}, ax=ax5)
ax5.text(.05, .95, '$R^2$ = '+str(np.corrcoef(bias_full[:,4],corrected4)[0,1]**2)[:6], fontsize=14)
ax5.text(.05, .9, 'MAE = '+str(mean_absolute_error(bias_full[:,4],corrected4))[:6], fontsize=14)
ax5.text(.05, .85, 'RMSE = '+str(np.sqrt(mean_squared_error(bias_full[:,4],corrected4)))[:6], fontsize=14)
ax5.set_xlim(0,1)
ax5.set_ylim(0,1)

# Save bias models
pickle.dump(reg0, open(model_traindir+'water_bias.pickle', "wb"))
pickle.dump(reg1, open(model_traindir+'barren_bias.pickle', "wb"))
pickle.dump(reg2, open(model_traindir+'herb_bias.pickle', "wb"))
pickle.dump(reg3, open(model_traindir+'shrub_bias.pickle', "wb"))
pickle.dump(reg4, open(model_traindir+'trees_bias.pickle', "wb"))

################################### STEP 3 ###################################
# Independent validation of bias correction + normalization
# Bias correction
def sample_for_validation(class_path, nclasses, modeled_path, stack_path, n, ids, bias_ids):
    class_30m = rasterio.open(traindir+class_path,'r')
    modeled = rasterio.open(model_traindir+modeled_path,'r')
    full_stack = rasterio.open(outdir+stack_path,'r')
    class_30m_img = class_30m.read()
    # Clip feature stack to classified Maxar image
    window = rasterio.windows.from_bounds(*class_30m.bounds, transform=modeled.transform)
    modeled_clipped = modeled.read(window=window, boundless=True)
    full_stack_clipped = full_stack.read(window=window, boundless=True)
    # Reshape to 2D array
    nsamples, nx, ny = modeled_clipped.shape 
    modeled_clipped_2d = modeled_clipped.reshape((nsamples,nx*ny))
    modeled_clipped_2d = modeled_clipped_2d.T
    modeled_clipped_2d = modeled_clipped_2d.astype(float)/10000
    # Reshape to 2D array
    nsamples0, nx0, ny0 = full_stack_clipped.shape
    full_stack_clipped_2d = full_stack_clipped.reshape((nsamples0,nx0*ny0))
    full_stack_clipped_2d = full_stack_clipped_2d.T
    # Reshape to 2D array
    nsamples1, nx1, ny1 = class_30m_img.shape 
    class_30m_img_2d = class_30m_img.reshape((nsamples1,nx1*ny1))
    class_30m_img_2d = class_30m_img_2d.T
    if nclasses<5:
        trees = np.zeros((nx*ny,1))
        class_30m_img_2d = np.concatenate([class_30m_img_2d, trees], axis=1)
    # Column join the modeled to the training
    comparison = np.append(class_30m_img_2d, modeled_clipped_2d, axis=1)
    # Remove rows with no-data values in the classified image (ndval=32767) and in the modeled image
    comparison = comparison[np.where(((comparison[:,0:nclasses]!=32767).all(1)) & 
                                       ((comparison[:,0:nclasses]>=0).all(1)) &
                                       ((full_stack_clipped_2d[:,:]!=32767).all(1)))]
    # Remove rows that were used in training and bias correction
    train_sample = comparison[ids,:]
    bias_pool = np.delete(comparison, ids, axis=0)
    bias_pool = bias_pool[~np.any(bias_pool==3.2767, axis=1)]
    bias_sample = bias_pool[bias_ids,:]
    valid_sample = np.delete(bias_pool, bias_ids, axis=0)
    
    # Random sample for validation 
    row_ids = random.sample(range(0, valid_sample.shape[0]-1), n)
    rand_sample = valid_sample[row_ids, :]
    
    return train_sample, bias_sample, rand_sample, row_ids


images = glob.glob('/projectnb/modislc/users/seamorez/HLS_FCover/model_training/*tif')
for i, row in train_key.iterrows():
    full_stack = rasterio.open(outdir+str(row['year'])+'/'+row['above_tile']+
                               '/feats_full_'+str(row['year'])+'.tif','r')
    meta = full_stack.meta
    full_stack_img = full_stack.read()
    nsamples, nx, ny = full_stack_img.shape 
    full_stack_img_2d = full_stack_img.reshape((nsamples,nx*ny))
    full_stack_img_2d = full_stack_img_2d.T
    
    classified = rasterio.open(model_traindir+'FCover_'+str(row['year'])+'_'+row['above_tile']+'_rfr.tif','r').read()
    nsamples, nx, ny = classified.shape
    # Correct for each class
    water_corrected = classified[0].reshape(nx*ny) - reg0.predict(classified[0].reshape(nx*ny,1))
    barren_corrected = classified[1].reshape(nx*ny) - reg1.predict(classified[1].reshape(nx*ny,1))
    herb_corrected = classified[2].reshape(nx*ny) - reg2.predict(classified[2].reshape(nx*ny,1))
    shrub_corrected = classified[3].reshape(nx*ny) - reg3.predict(classified[3].reshape(nx*ny,1))
    trees_corrected = classified[4].reshape(nx*ny) - reg4.predict(classified[4].reshape(nx*ny,1))

    # Correct negative values to 0% cover
    water_corrected[water_corrected<0]=0
    barren_corrected[barren_corrected<0]=0
    herb_corrected[herb_corrected<0]=0
    shrub_corrected[shrub_corrected<0]=0
    trees_corrected[trees_corrected<0]=0
    
    # Normalize
    # Reshape
    water_corrected = water_corrected.reshape(1, nx*ny)
    barren_corrected = barren_corrected.reshape(1, nx*ny)
    herb_corrected = herb_corrected.reshape(1, nx*ny)
    shrub_corrected = shrub_corrected.reshape(1, nx*ny)
    trees_corrected = trees_corrected.reshape(1, nx*ny)

    # Water and Barren post-processing step: >= 90% cover post-correction becomes 100% and 0% for all other cover types
    water_corrected[water_corrected>=9000] = 10000
    barren_corrected[water_corrected>=9000] = 0
    herb_corrected[water_corrected>=9000] = 0
    shrub_corrected[water_corrected>=9000] = 0
    trees_corrected[water_corrected>=9000] = 0
    barren_corrected[barren_corrected>=9000] = 10000
    water_corrected[barren_corrected>=9000] = 0
    herb_corrected[barren_corrected>=9000] = 0
    shrub_corrected[barren_corrected>=9000] = 0
    trees_corrected[barren_corrected>=9000] = 0

    # Trees post-processing step: remove fractional trees <= 2% cover (low values lead to inaccuracies since so sparse)
    trees_corrected[trees_corrected<=200] = 0

    # First step to normalize results: joining the fractional cover
    all_corrected = np.concatenate([water_corrected,barren_corrected,herb_corrected,shrub_corrected, trees_corrected], axis=0)
    # Sums, which need to be normalized to 1
    totals_corrected = all_corrected.sum(axis=0)
    # Normalized fractional cover for each class
    all_norm = all_corrected/totals_corrected[:, np.newaxis].T

    # Separating back out to classes after normalization
    water_norm = all_norm[0].reshape(1, nx*ny)
    barren_norm = all_norm[1].reshape(1, nx*ny)
    herb_norm = all_norm[2].reshape(1, nx*ny)
    shrub_norm = all_norm[3].reshape(1, nx*ny)
    trees_norm = all_norm[4].reshape(1, nx*ny)

    # Removing no-data values and scaling
    water_norm = np.around(water_norm*10000)
    barren_norm = np.around(barren_norm*10000)
    herb_norm = np.around(herb_norm*10000)
    shrub_norm = np.around(shrub_norm*10000)
    trees_norm = np.around(trees_norm*10000)
    
    full_stack_band = full_stack_img[27].reshape(1, nx*ny)
    water_norm[classified[0].reshape(1,nx*ny)==32767] = 32767
    barren_norm[classified[1].reshape(1,nx*ny)==32767] = 32767
    herb_norm[classified[2].reshape(1,nx*ny)==32767] = 32767
    shrub_norm[classified[3].reshape(1,nx*ny)==32767] = 32767
    trees_norm[classified[4].reshape(1,nx*ny)==32767] = 32767

    # Reshape
    water_norm = water_norm.reshape(nx, ny)
    barren_norm = barren_norm.reshape(nx, ny)
    herb_norm = herb_norm.reshape(nx, ny)
    shrub_norm = shrub_norm.reshape(nx, ny)
    trees_norm = trees_norm.reshape(nx, ny)

    # Saving normalized results as bands in one image
    meta2 = meta
    meta2.update(dtype='int16', nodata=32767, count=5)

    results_list = [water_norm,barren_norm,herb_norm,shrub_norm, trees_norm]

    with rasterio.open(model_traindir+'FCover_'+str(row['year'])+'_'+row['above_tile']+'_rfr_corrected.tif', 'w', **meta2) as dst:
        for id,row in enumerate(results_list, start=1):
            dst.write_band(id, row)

# Second, do validation selection
for i, row in train_key.iterrows():
    # Get trraining indices
    indices = np.loadtxt(model_traindir+'train_ids.txt')[i*row['nsamples']:i*row['nsamples']+row['nsamples']]
    print(indices.shape)
    train_sample, bias_sample, valid_sample, valid_ids = sample_for_validation(row['ecoreg_code']+'/training_imgs/'+row['train_img_name'], 
                                                      row['nclasses'],
                                                      'FCover_'+str(row['year'])+'_'+row['above_tile']+'_rfr_corrected.tif',
                                                      str(row['year'])+'/'+row['above_tile']+'/feats_full_'+str(row['year'])+'.tif',
                                                      row['nsamples'],
                                                      indices.astype(int),
                                                      bias_ids_full[i*row['nsamples']:i*row['nsamples']+row['nsamples']])
    
    if i==0: 
        valid_full = valid_sample; valid_ids_full = valid_ids
    else: 
        valid_full = np.append(valid_full, valid_sample, axis=0)
        valid_ids_full = np.append(valid_ids_full, valid_ids, axis=0)

# Validation scatterplots
fig, ((ax1, ax2, ax5), (ax3, ax4, ax6)) = plt.subplots(2, 3)
fig.set_size_inches(27, 18)

#ax1.scatter(all_class[:,0],all_results[:,0], color='Blue', alpha=0.2)
sb.regplot(x=valid_full[:,0],y=valid_full[:,5], color='#0071DC', scatter_kws={'alpha':0.2}, ax=ax1)
ax1.text(.05, .95, '$R^2$ = '+str(np.corrcoef(valid_full[:,0],valid_full[:,5])[0,1]**2)[:6], fontsize=14)
ax1.text(.05, .9, 'MAE = '+str(mean_absolute_error(valid_full[:,0],valid_full[:,5]))[:6], fontsize=14)
ax1.text(.05, .85, 'RMSE = '+str(np.sqrt(mean_squared_error(valid_full[:,0],valid_full[:,5])))[:6], fontsize=14)
ax1.plot([0,1],[0,1], color='Black', linewidth=.5)
ax1.set_xlim(0,1)
ax1.set_ylim(0,1)
ax1.set_title('Water fractional cover',fontsize=20)
ax1.set_xlabel('Observed',fontsize=14)
ax1.set_ylabel('Predicted',fontsize=14)

#ax2.scatter(all_class[:,1],all_results[:,1], color='Grey', alpha=0.2)
sb.regplot(x=valid_full[:,1],y=valid_full[:,6], color='Grey', scatter_kws={'alpha':0.2}, ax=ax2)
ax2.text(.05, .95, '$R^2$ = '+str(np.corrcoef(valid_full[:,1],valid_full[:,6])[0,1]**2)[:6], fontsize=14)
ax2.text(.05, .9, 'MAE = '+str(mean_absolute_error(valid_full[:,1],valid_full[:,6]))[:6], fontsize=14)
ax2.text(.05, .85, 'RMSE = '+str(np.sqrt(mean_squared_error(valid_full[:,1],valid_full[:,6])))[:6], fontsize=14)
ax2.plot([0,1],[0,1], color='Black', linewidth=.5)
ax2.set_xlim(0,1)
ax2.set_ylim(0,1)
ax2.set_title('Barren fractional cover',fontsize=20)
ax2.set_xlabel('Observed',fontsize=14)
ax2.set_ylabel('Predicted',fontsize=14)

#ax3.scatter(all_class[:,2],all_results[:,2], color='YellowGreen', alpha=0.2)
sb.regplot(x=valid_full[:,2],y=valid_full[:,7], color='YellowGreen', scatter_kws={'alpha':0.2}, ax=ax3)
ax3.text(.05, .95, '$R^2$ = '+str(np.corrcoef(valid_full[:,2],valid_full[:,7])[0,1]**2)[:6], fontsize=14)
ax3.text(.05, .9, 'MAE = '+str(mean_absolute_error(valid_full[:,2],valid_full[:,7]))[:6], fontsize=14)
ax3.text(.05, .85, 'RMSE = '+str(np.sqrt(mean_squared_error(valid_full[:,2],valid_full[:,7])))[:6], fontsize=14)
ax3.plot([0,1],[0,1], color='Black', linewidth=.5)
ax3.set_xlim(0,1)
ax3.set_ylim(0,1)
ax3.set_title('Herbaceous fractional cover',fontsize=20)
ax3.set_xlabel('Observed',fontsize=14)
ax3.set_ylabel('Predicted',fontsize=14)

#ax4.scatter(all_class[:,3],all_results[:,3], color='Green', alpha=0.2)
sb.regplot(x=valid_full[:,3],y=valid_full[:,8], color='Green', scatter_kws={'alpha':0.2}, ax=ax4)
ax4.text(.05, .95, '$R^2$ = '+str(np.corrcoef(valid_full[:,3],valid_full[:,8])[0,1]**2)[:6], fontsize=14)
ax4.text(.05, .9, 'MAE = '+str(mean_absolute_error(valid_full[:,3],valid_full[:,8]))[:6], fontsize=14)
ax4.text(.05, .85, 'RMSE = '+str(np.sqrt(mean_squared_error(valid_full[:,3],valid_full[:,8])))[:6], fontsize=14)
ax4.plot([0,1],[0,1], color='Black', linewidth=.5)
ax4.set_xlim(0,1)
ax4.set_ylim(0,1)
ax4.set_title('Shrub fractional cover',fontsize=20)
ax4.set_xlabel('Observed',fontsize=14)
ax4.set_ylabel('Predicted',fontsize=14)

#ax5.scatter(all_class[:,3],all_results[:,3], color='Green', alpha=0.2)
sb.regplot(x=valid_full[:,4],y=valid_full[:,9], color='darkgreen', scatter_kws={'alpha':0.2}, ax=ax5)
ax5.text(.05, .95, '$R^2$ = '+str(np.corrcoef(valid_full[:,4],valid_full[:,9])[0,1]**2)[:6], fontsize=14)
ax5.text(.05, .9, 'MAE = '+str(mean_absolute_error(valid_full[:,4],valid_full[:,9]))[:6], fontsize=14)
ax5.text(.05, .85, 'RMSE = '+str(np.sqrt(mean_squared_error(valid_full[:,4],valid_full[:,9])))[:6], fontsize=14)
ax5.plot([0,1],[0,1], color='Black', linewidth=.5)
ax5.set_xlim(0,1)
ax5.set_ylim(0,1)
ax5.set_title('Trees fractional cover',fontsize=20)
ax5.set_xlabel('Observed',fontsize=14)
ax5.set_ylabel('Predicted',fontsize=14)




## ***TESTING THE MODEL. FINAL MODELING IN SEPARATE SCRIPT*** ##
# Load in models
trees_regressor = pickle.load(open(model_traindir+"trees_rfr.pickle", "rb"))
shrub_regressor = pickle.load(open(model_traindir+"shrub_rfr.pickle", "rb"))
herb_regressor = pickle.load(open(model_traindir+"herb_rfr.pickle", "rb"))
barren_regressor = pickle.load(open(model_traindir+"barren_rfr.pickle", "rb"))
water_regressor = pickle.load(open(model_traindir+"water_rfr.pickle", "rb"))

# Do for 2016
tiles_2016 = glob.glob(outdir+'2016/*')

for path in tiles_2016:
    full_stack = rasterio.open(path+'/feats_full_2016.tif','r')
    meta = full_stack.meta
    full_stack_img = full_stack.read()
    nsamples, nx, ny = full_stack_img.shape 
    full_stack_img_2d = full_stack_img.reshape((nsamples,nx*ny))
    full_stack_img_2d = full_stack_img_2d.T
    
    # Water fractional classification and display
    water_predict = water_regressor.predict(full_stack_img_2d)
    water_predict = water_predict.reshape(6000,6000)
    #plt.imshow(water_predict, cmap = 'Blues', vmin=0, vmax=1)
    #plt.show()
    # Barren fractional classification and display
    barren_predict = barren_regressor.predict(full_stack_img_2d)
    barren_predict = barren_predict.reshape(6000,6000)
    #plt.imshow(barren_predict, cmap = 'Greys', vmin=0, vmax=1)
    #plt.show()
    # Herb fractional classification and display
    herb_predict = herb_regressor.predict(full_stack_img_2d)
    herb_predict = herb_predict.reshape(6000,6000)
    #plt.imshow(herb_predict, cmap = 'Oranges', vmin=0, vmax=1)
    #plt.show()
    # Shrub fractional classification and display
    shrub_predict = shrub_regressor.predict(full_stack_img_2d)
    shrub_predict = shrub_predict.reshape(6000,6000)
    #plt.imshow(shrub_predict, cmap = 'Greens', vmin=0, vmax=1)
    #plt.show()
    # Trees fractional classification and display
    trees_predict = trees_regressor.predict(full_stack_img_2d)
    trees_predict = trees_predict.reshape(6000,6000)
    #plt.imshow(trees_predict, cmap = 'Greens', vmin=0, vmax=1)
    #plt.show()
    
    # Normalize
    # Reshape
    water_predict = water_predict.reshape(1, nx*ny)
    barren_predict = barren_predict.reshape(1, nx*ny)
    herb_predict = herb_predict.reshape(1, nx*ny)
    shrub_predict = shrub_predict.reshape(1, nx*ny)
    trees_predict = trees_predict.reshape(1, nx*ny)
    
    # Trees post-processing step: remove fractional trees <= 2% cover (low values lead to inaccuracies since so sparse)
    trees_predict[trees_predict<=.02] = 0
    # Water and barren augmenting
    water_predict[water_predict>=.9] = 1
    barren_predict[water_predict>=.9] = 0
    herb_predict[water_predict>=.9] = 0
    shrub_predict[water_predict>=.9] = 0
    trees_predict[water_predict>=.9] = 0
    barren_predict[barren_predict>=.9] = 1
    water_predict[barren_predict>=.9] = 0
    herb_predict[barren_predict>=.9] = 0
    shrub_predict[barren_predict>=.9] = 0
    trees_predict[barren_predict>=.9] = 0
    
    # First step to normalize results: joining the fractional cover
    all_predict = np.concatenate([water_predict,barren_predict,herb_predict,shrub_predict, trees_predict], axis=0)
    # Sums, which need to be normalized to 1
    totals_predict = all_predict.sum(axis=0)
    # Normalized fractional cover for each class
    all_norm = all_predict/totals_predict[:, np.newaxis].T
    
    # Separating back out to classes after normalization
    water_norm = all_norm[0].reshape(1, nx*ny)
    barren_norm = all_norm[1].reshape(1, nx*ny)
    herb_norm = all_norm[2].reshape(1, nx*ny)
    shrub_norm = all_norm[3].reshape(1, nx*ny)
    trees_norm = all_norm[4].reshape(1, nx*ny)
    
    # Removing no-data values and scaling
    water_norm = np.around(water_norm*10000)
    barren_norm = np.around(barren_norm*10000)
    herb_norm = np.around(herb_norm*10000)
    shrub_norm = np.around(shrub_norm*10000)
    trees_norm = np.around(trees_norm*10000)
    
    full_stack_band = full_stack_img[10].reshape(1, nx*ny)
    water_norm[full_stack_band==32767] = 32767
    barren_norm[full_stack_band==32767] = 32767
    herb_norm[full_stack_band==32767] = 32767
    shrub_norm[full_stack_band==32767] = 32767
    trees_norm[full_stack_band==32767] = 32767
    
    # Reshape
    water_norm = water_norm.reshape(nx, ny)
    barren_norm = barren_norm.reshape(nx, ny)
    herb_norm = herb_norm.reshape(nx, ny)
    shrub_norm = shrub_norm.reshape(nx, ny)
    trees_norm = trees_norm.reshape(nx, ny)
    
    # Saving normalized results as bands in one image
    meta2 = meta
    meta2.update(dtype='int16', nodata=32767, count=5)
    
    results_list = [water_norm,barren_norm,herb_norm,shrub_norm, trees_norm]
    
    
    tile = path.split('/')[8]
    with rasterio.open(path+'/FCover_'+tile+'_rfr.tif', 'w', **meta2) as dst:
        for id,row in enumerate(results_list, start=1):
            print(row[row<0])
            dst.write_band(id, row)