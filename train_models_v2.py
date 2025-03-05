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
def sample_training(class_path, nclasses, stack_path, n, i):
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
    class_feats = class_feats[np.where(((class_feats[:,0:nclasses]!=32767).all(1)) & 
                                       ((class_feats[:,0:nclasses]>=0).all(1)) &
                                       ((class_feats[:,nclasses:]!=32767).all(1)))]
    
    # Remove rows where features have nan values
    class_feats = class_feats[~np.any(class_feats==32767, axis=1)]
    class_feats_to_sample = class_feats.copy()
    print('to sample size: ', class_feats_to_sample.shape)
    
    # Random sample
    print('class: ', i)
    if (i==4) & (nclasses==4): 
        row_ids = random.sample(range(0, class_feats.shape[0]-1), n)
        rand_sample = class_feats[row_ids, :]
        row_ids = np.array(row_ids, dtype=int)
    else:
        for j in np.linspace(0,.8,5):
            if j<.8: sample_row = np.argwhere((class_feats[:,i]>=j) & (class_feats[:,i]<(j+.2)) & (class_feats[:,i]!=32767))
            else: sample_row = np.argwhere((class_feats[:,i]>=j) & (class_feats[:,i]<=(j+.2)) & (class_feats[:,i]!=32767))
            sample_row = sample_row.reshape(sample_row.size)
            
            # Number of points to sample based on proportion of total points
            print(sample_row.size, class_feats[:,i][class_feats[:,i]!=32767].size)
            n_samples = round(sample_row.size/class_feats[:,i][class_feats[:,i]!=32767].size*1000)
            if (n_samples < 20) & (i!=4):
                if (i!=0) & (i!=1):
                    n_samples = min(20, sample_row.size)
            if ((i==0)|(i==1)) & (j==.8):
                n_samples = min(sample_row.size, round(n_samples - (n_samples-200)/8))
            print(n_samples)
            
            if j==0: row_ids = np.random.choice(sample_row, n_samples, replace=False)
            else: row_ids = np.append(row_ids, np.random.choice(sample_row, n_samples, replace=False))
                
            # Augment water class with high trees to avoid confusion
            if (j==0) & (i==0) & (nclasses==5):
                extra_sample_row = np.argwhere((class_feats[:,4]>=.4) & (class_feats[:,4]<=1) & (class_feats[:,4]!=32767))
                extra_sample_row = extra_sample_row.reshape(extra_sample_row.size)
                print('trees total: ', extra_sample_row.size)
                extra_ids = np.random.choice(extra_sample_row,min(extra_sample_row.size,200),replace=False)
                row_ids = np.append(row_ids, extra_ids)
                row_ids = np.unique(row_ids)
            
            class_feats[row_ids, :] = 32767
            
        rand_sample = class_feats_to_sample[row_ids, :]
    
    # Return 2D array of classifications (4 x nsamples) and the corresponding features (nsamples x nfeatures)
    return rand_sample[:,0:nclasses].T, rand_sample[:,nclasses:], row_ids


def class_training(cover):
    for i, row in train_key.iterrows():
        # if (cover==4) & (row['nclasses']==4): 
        #     if i==0: samples_image = np.array((0,0))
        #     else: samples_image = np.append(samples_image, samples_image[i]+0)
        #     continue
        train_pts, train_feats, ids = sample_training(row['ecoreg_code']+'/training_imgs/'+row['train_img_name'], 
                                                      row['nclasses'],
                                                      str(row['year'])+'/'+row['above_tile']+'/feats_full_'+str(row['year'])+'.tif',
                                                      row['nsamples'], cover)
        if row['nclasses']==4:
            trees = np.zeros(train_pts.shape[1])
            print(trees.shape)
            train_pts = np.vstack((train_pts, trees))
            
        print(train_pts.shape)
        
        if i==0: 
            train_pts_full = train_pts; train_feats_full = train_feats; ids_full = ids
            samples_image = np.array((0,ids.size))
        else: 
            train_pts_full = np.append(train_pts_full, train_pts, axis=1)
            train_feats_full = np.append(train_feats_full, train_feats, axis=0)
            ids_full = np.append(ids_full, ids, axis=0)
            samples_image = np.append(samples_image, samples_image[i]+ids.size)
            print(train_pts_full.shape, train_feats_full.shape, ids_full.shape)
    
    return train_pts_full, train_feats_full, ids_full, samples_image

water_pts, water_feats, water_ids, water_per_img = class_training(0)
barren_pts, barren_feats, barren_ids, barren_per_img = class_training(1)
herb_pts, herb_feats, herb_ids, herb_per_img = class_training(2)
shrub_pts, shrub_feats, shrub_ids, shrub_per_img = class_training(3)
trees_pts, trees_feats, trees_ids, trees_per_img = class_training(4)

plt.hist(water_pts[0,:], bins=10)

# Train models: four regressors, one for each class
water_feats = water_feats[:,[3,4,5,7,8,13,14,15,17,18,23,24,25,27,28,30,31,32]]  # focus on wetness info
water_regressor = RandomForestRegressor(n_estimators=500, min_samples_leaf=5, random_state=7).fit(water_feats, water_pts[0])
barren_regressor = RandomForestRegressor(n_estimators=500, min_samples_leaf=5, random_state=7).fit(barren_feats, barren_pts[1])
herb_regressor = RandomForestRegressor(n_estimators=500, min_samples_leaf=5, random_state=7).fit(herb_feats, herb_pts[2])
shrub_regressor = RandomForestRegressor(n_estimators=500, min_samples_leaf=5, random_state=7).fit(shrub_feats, shrub_pts[3])
trees_regressor = RandomForestRegressor(n_estimators=500, min_samples_leaf=5, random_state=7).fit(trees_feats, trees_pts[4])

band_names = np.array(['50PCGI B','50PCGI G','50PCGI R','50PCGI NIR','50PCGI SWIR1','50PCGI SWIR2','50PCGI EVI2','50PCGI TCG','50PCGI TCW','50PCGI TCB',
                       'Peak B','Peak G','Peak R','Peak NIR','Peak SWIR1','Peak SWIR2','Peak EVI2','Peak TCG','Peak TCW','Peak TCB',
                       '50PCGD B','50PCGD G','50PCGD R','50PCGD NIR','50PCGD SWIR1','50PCGD SWIR2','50PCGD EVI2','50PCGD TCG','50PCGD TCW','50PCGD TCB',
                       'Elevation','Slope','Aspect'])

w_band_names = np.array(['50PCGI NIR','50PCGI SWIR1','50PCGI SWIR2','50PCGI TCG','50PCGI TCW',
                       'Peak_NIR','Peak SWIR1','Peak SWIR2','Peak TCG','Peak TCW',
                       '50PCGD NIR','50PCGD SWIR1','50PCGD SWIR2','50PCGD TCG','50PCGD TCW',
                       'Elevation','Slope','Aspect'])

herb_variable_importance = pd.DataFrame({'Band':band_names,'Importance':trees_regressor.feature_importances_})
herb_variable_importance.sort_values(by=['Importance'], ascending=True, inplace=True)
plt.figure(figsize=(14,16))
plt.barh(herb_variable_importance['Band'],herb_variable_importance['Importance'], color='#982649')

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

# Save train ids by image
for idx in range(len(train_key)):
    ids_full = np.concatenate((water_ids[water_per_img[idx]:water_per_img[idx+1]], 
                               barren_ids[barren_per_img[idx]:barren_per_img[idx+1]], 
                               herb_ids[herb_per_img[idx]:herb_per_img[idx+1]], 
                               shrub_ids[shrub_per_img[idx]:shrub_per_img[idx+1]], 
                               trees_ids[trees_per_img[idx]:trees_per_img[idx+1]]))
    print(ids_full.shape)
    ids_full_unique = np.unique(ids_full)
    print(ids_full_unique.shape)
    np.savetxt(model_traindir+'train_ids_img'+str(idx)+'.txt', ids_full, delimiter=',')


################################### STEP 2 ###################################
# Bias correction
def sample_for_bias(class_path, nclasses, modeled_path, stack_path, n, i, ids):
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
    print('to sample size: ', comparison.shape)
    # Remove rows that were used in training
    train_sample = comparison[ids,:]
    print(train_sample.shape)
    bias_corr_sample = np.delete(comparison, ids, axis=0)
    #bias_corr_sample = comparison
    # Remove rows where prediction is nan
    bias_corr_sample = bias_corr_sample[~np.any(bias_corr_sample==3.2767, axis=1)]
    bias_corr_to_sample = bias_corr_sample.copy()
    
    # Random sample for bias correction
    #row_ids = random.sample(range(0, bias_corr_sample.shape[0]-1), n)
    #rand_sample = bias_corr_sample[row_ids, :]
    
    # Random sample
    print('class: ', i)
    for j in np.linspace(0,.8,5):
        if j<.8: sample_row = np.argwhere((bias_corr_sample[:,i]>=j) & (bias_corr_sample[:,i]<(j+.2)) & (bias_corr_sample[:,i]!=32767))
        else: sample_row = np.argwhere((bias_corr_sample[:,i]>=j) & (bias_corr_sample[:,i]<=(j+.2)) & (bias_corr_sample[:,i]!=32767))
        sample_row = sample_row.reshape(sample_row.size)
        
        # Number of points to sample based on proportion of total points
        print(sample_row.size, bias_corr_sample[:,i][bias_corr_sample[:,i]!=32767].size)
        n_samples = round(sample_row.size/bias_corr_sample[:,i][bias_corr_sample[:,i]!=32767].size*1000)
        if n_samples < 20:
            n_samples = min(20, sample_row.size)
        #n_samples = min(sample_row.size, 20)#round(n_samples - (n_samples-200)/8))
        #else:
        print(n_samples)
        
        if j==0: row_ids = np.random.choice(sample_row, n_samples, replace=False)
        else: row_ids = np.append(row_ids, np.random.choice(sample_row, n_samples, replace=False))
            
    bias_corr_sample[row_ids, :] = 32767
    rand_sample = bias_corr_to_sample[row_ids, :]
    
    return train_sample, rand_sample, row_ids


def class_bias(cover):
    for i, row in train_key.iterrows():
        print(i)
        indices = np.loadtxt(model_traindir+'train_ids_img'+str(i)+'.txt')
        print(indices.shape)
        if (cover==4) & (row['nclasses']==4): 
            if i==0: samples_image = np.array((0,0))
            else: samples_image = np.append(samples_image, 0)
            continue
        train_pts, bias_pts, bias_ids = sample_for_bias(row['ecoreg_code']+'/training_imgs/'+row['train_img_name'], 
                                                      row['nclasses'],
                                                      'FCover_'+str(row['year'])+'_'+row['above_tile']+'_rfr.tif',
                                                      str(row['year'])+'/'+row['above_tile']+'/feats_full_'+str(row['year'])+'.tif',
                                                      row['nsamples'], cover, indices.astype(int))
        if row['nclasses']==4:
            trees = np.zeros(train_pts.shape[1])
            print(trees.shape)
            train_pts = np.vstack((train_pts, trees))
            
        print(train_pts.shape)
        
        if i==0: 
            train_pts_full = train_pts; bias_pts_full = bias_pts; bias_ids_full = bias_ids
            samples_image = np.array((0,bias_ids.size))
        else: 
            train_pts_full = np.append(train_pts_full, train_pts, axis=0)
            bias_pts_full = np.append(bias_pts_full, bias_pts, axis=0)
            bias_ids_full = np.append(bias_ids_full, bias_ids, axis=0)
            samples_image = np.append(samples_image, samples_image[i]+bias_ids.size)
            print(train_pts_full.shape, bias_pts_full.shape, bias_ids_full.shape)
    
    return train_pts_full, bias_pts_full, bias_ids_full, samples_image


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
    
    water_stack_img_2d = full_stack_img_2d[:,[3,4,5,7,8,13,14,15,17,18,23,24,25,27,28,30,31,32]]

    # Water fractional classification and display
    water_predict = water_regressor.predict(water_stack_img_2d)
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
water_train_sample, water_bias_sample, water_bias_ids, water_bias_per_img = class_bias(0)
barren_train_sample, barren_bias_sample, barren_bias_ids, barren_bias_per_img = class_bias(1)
herb_train_sample, herb_bias_sample, herb_bias_ids, herb_bias_per_img = class_bias(2)
shrub_train_sample, shrub_bias_sample, shrub_bias_ids, shrub_bias_per_img = class_bias(3)
trees_train_sample, trees_bias_sample, trees_bias_ids, trees_bias_per_img = class_bias(4)

# Save bias ids by image
for idx in range(len(train_key)):
    ids_full = np.concatenate((water_bias_ids[water_bias_per_img[idx]:water_bias_per_img[idx+1]], 
                               barren_bias_ids[barren_bias_per_img[idx]:barren_bias_per_img[idx+1]], 
                               herb_bias_ids[herb_bias_per_img[idx]:herb_bias_per_img[idx+1]], 
                               shrub_bias_ids[shrub_bias_per_img[idx]:shrub_bias_per_img[idx+1]], 
                               trees_bias_ids[trees_bias_per_img[idx]:trees_bias_per_img[idx+1]]))
    print(ids_full.shape)
    ids_full_unique = np.unique(ids_full)
    print(ids_full_unique.shape)
    np.savetxt(model_traindir+'bias_ids_img'+str(idx)+'.txt', ids_full, delimiter=',')

# for i, row in train_key.iterrows():
#     # Get trraining indices
#     indices = np.loadtxt(model_traindir+'train_ids_img'+str(i)+'.txt')
#     print(indices.shape)
#     train_sample, bias_corr_sample, bias_corr_ids = sample_for_bias(row['ecoreg_code']+'/training_imgs/'+row['train_img_name'], 
#                                                       row['nclasses'],
#                                                       'FCover_'+str(row['year'])+'_'+row['above_tile']+'_rfr.tif',
#                                                       str(row['year'])+'/'+row['above_tile']+'/feats_full_'+str(row['year'])+'.tif',
#                                                       row['nsamples'],
#                                                       indices.astype(int))
    
#     if i==0: 
#         train_full = train_sample; bias_full = bias_corr_sample; bias_ids_full = bias_corr_ids
#     else: 
#         train_full = np.append(train_full, train_sample, axis=0)
#         bias_full = np.append(bias_full, bias_corr_sample, axis=0)
#         bias_ids_full = np.append(bias_ids_full, bias_corr_ids, axis=0)
#         print(train_full.shape, bias_full.shape)
    
# np.savetxt(model_traindir+'bias_ids.txt', bias_ids_full, delimiter=',')

# Bias correction
nrows0, _ = water_bias_sample.shape
reg0 = LinearRegression().fit(water_bias_sample[:,0].reshape(nrows0,1), water_bias_sample[:,5]-water_bias_sample[:,0])
corrected0 = water_bias_sample[:,5] - reg0.predict(water_bias_sample[:,5].reshape(nrows0,1))
corrected0[corrected0<0]=0
nrows1, _ = barren_bias_sample.shape
reg1 = LinearRegression().fit(barren_bias_sample[:,1].reshape(nrows1,1), barren_bias_sample[:,6]-barren_bias_sample[:,1])
corrected1 = barren_bias_sample[:,6] - reg1.predict(barren_bias_sample[:,6].reshape(nrows1,1))
corrected1[corrected1<0]=0
nrows2, _ = herb_bias_sample.shape
reg2 = LinearRegression().fit(herb_bias_sample[:,2].reshape(nrows2,1), herb_bias_sample[:,7]-herb_bias_sample[:,2])
corrected2 = herb_bias_sample[:,7] - reg2.predict(herb_bias_sample[:,7].reshape(nrows2,1))
corrected2[corrected2<0]=0
nrows3, _ = shrub_bias_sample.shape
reg3 = LinearRegression().fit(shrub_bias_sample[:,3].reshape(nrows3,1), shrub_bias_sample[:,8]-shrub_bias_sample[:,3])
corrected3 = shrub_bias_sample[:,8] - reg3.predict(shrub_bias_sample[:,8].reshape(nrows3,1))
corrected3[corrected3<0]=0
nrows4, _ = trees_bias_sample.shape
reg4 = LinearRegression().fit(trees_bias_sample[:,4].reshape(nrows4,1), trees_bias_sample[:,9]-trees_bias_sample[:,4])
corrected4 = trees_bias_sample[:,9] - reg4.predict(trees_bias_sample[:,9].reshape(nrows4,1))
corrected4[corrected4<0]=0

fig, ((ax1, ax2, ax5), (ax3, ax4, ax6)) = plt.subplots(2, 3)
fig.set_size_inches(27, 18)

sb.regplot(x=water_bias_sample[:,0],y=water_bias_sample[:,5], color='#0071DC', scatter_kws={'alpha':0.2}, ax=ax1)
ax1.plot([0,1],[0,1], color='Black', linewidth=.5)
sb.regplot(y=corrected0,x=water_bias_sample[:,0], color='mediumvioletred', scatter_kws={'alpha':0.2}, ax=ax1)
ax1.text(.05, .95, '$R^2$ = '+str(np.corrcoef(water_bias_sample[:,0],corrected0)[0,1]**2)[:6], fontsize=14)
ax1.text(.05, .9, 'MAE = '+str(mean_absolute_error(water_bias_sample[:,0],corrected0))[:6], fontsize=14)
ax1.text(.05, .85, 'RMSE = '+str(np.sqrt(mean_squared_error(water_bias_sample[:,0],corrected0)))[:6], fontsize=14)
ax1.set_xlim(0,1)
ax1.set_ylim(0,1)
sb.regplot(x=barren_bias_sample[:,1],y=barren_bias_sample[:,6], color='Grey', scatter_kws={'alpha':0.2}, ax=ax2)
ax2.plot([0,1],[0,1], color='Black', linewidth=.5)
sb.regplot(y=corrected1,x=barren_bias_sample[:,1], color='mediumvioletred', scatter_kws={'alpha':0.2}, ax=ax2)
ax2.text(.05, .95, '$R^2$ = '+str(np.corrcoef(barren_bias_sample[:,1],corrected1)[0,1]**2)[:6], fontsize=14)
ax2.text(.05, .9, 'MAE = '+str(mean_absolute_error(barren_bias_sample[:,1],corrected1))[:6], fontsize=14)
ax2.text(.05, .85, 'RMSE = '+str(np.sqrt(mean_squared_error(barren_bias_sample[:,1],corrected1)))[:6], fontsize=14)
ax2.set_xlim(0,1)
ax2.set_ylim(0,1)
sb.regplot(x=herb_bias_sample[:,2],y=herb_bias_sample[:,7], color='YellowGreen', scatter_kws={'alpha':0.2}, ax=ax3)
ax3.plot([0,1],[0,1], color='Black', linewidth=.5)
sb.regplot(y=corrected2,x=herb_bias_sample[:,2], color='mediumvioletred', scatter_kws={'alpha':0.2}, ax=ax3)
ax3.text(.05, .95, '$R^2$ = '+str(np.corrcoef(herb_bias_sample[:,2],corrected2)[0,1]**2)[:6], fontsize=14)
ax3.text(.05, .9, 'MAE = '+str(mean_absolute_error(herb_bias_sample[:,2],corrected2))[:6], fontsize=14)
ax3.text(.05, .85, 'RMSE = '+str(np.sqrt(mean_squared_error(herb_bias_sample[:,2],corrected2)))[:6], fontsize=14)
ax3.set_xlim(0,1)
ax3.set_ylim(0,1)
sb.regplot(x=shrub_bias_sample[:,3],y=shrub_bias_sample[:,8], color='Green', scatter_kws={'alpha':0.2}, ax=ax4)
ax4.plot([0,1],[0,1], color='Black', linewidth=.5)
sb.regplot(y=corrected3,x=shrub_bias_sample[:,3], color='mediumvioletred', scatter_kws={'alpha':0.2}, ax=ax4)
ax4.text(.05, .95, '$R^2$ = '+str(np.corrcoef(shrub_bias_sample[:,3],corrected3)[0,1]**2)[:6], fontsize=14)
ax4.text(.05, .9, 'MAE = '+str(mean_absolute_error(shrub_bias_sample[:,3],corrected3))[:6], fontsize=14)
ax4.text(.05, .85, 'RMSE = '+str(np.sqrt(mean_squared_error(shrub_bias_sample[:,3],corrected3)))[:6], fontsize=14)
ax4.set_xlim(0,1)
ax4.set_ylim(0,1)
sb.regplot(x=trees_bias_sample[:,4],y=trees_bias_sample[:,9], color='darkgreen', scatter_kws={'alpha':0.2}, ax=ax5)
ax5.plot([0,1],[0,1], color='Black', linewidth=.5)
sb.regplot(y=corrected4,x=trees_bias_sample[:,4], color='mediumvioletred', scatter_kws={'alpha':0.2}, ax=ax5)
ax5.text(.05, .95, '$R^2$ = '+str(np.corrcoef(trees_bias_sample[:,4],corrected4)[0,1]**2)[:6], fontsize=14)
ax5.text(.05, .9, 'MAE = '+str(mean_absolute_error(trees_bias_sample[:,4],corrected4))[:6], fontsize=14)
ax5.text(.05, .85, 'RMSE = '+str(np.sqrt(mean_squared_error(trees_bias_sample[:,4],corrected4)))[:6], fontsize=14)
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
def sample_for_validation(class_path, nclasses, modeled_path, stack_path, n, i, ids, bias_ids):
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
    modeled_clipped_2d = modeled_clipped_2d.astype(float)/100
    # Reshape to 2D array
    nsamples0, nx0, ny0 = full_stack_clipped.shape
    full_stack_clipped_2d = full_stack_clipped.reshape((nsamples0,nx0*ny0))
    full_stack_clipped_2d = full_stack_clipped_2d.T
    # Reshape to 2D array
    nsamples1, nx1, ny1 = class_30m_img.shape 
    class_30m_img_2d = class_30m_img.reshape((nsamples1,nx1*ny1))
    class_30m_img_2d = class_30m_img_2d.T
    class_30m_img_2d = np.around(class_30m_img_2d, 2)
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
    # Random sample
    print('class: ', i)
    for j in np.linspace(0,.8,5):
        if j<.8: sample_row = np.argwhere((valid_sample[:,i]>=j) & (valid_sample[:,i]<(j+.2)) & (valid_sample[:,i]!=32767))
        else: sample_row = np.argwhere((valid_sample[:,i]>=j) & (valid_sample[:,i]<=(j+.2)) & (valid_sample[:,i]!=32767))
        sample_row = sample_row.reshape(sample_row.size)
        
        # Number of points to sample based on proportion of total points
        print(sample_row.size, valid_sample[:,i][valid_sample[:,i]!=32767].size)
        n_samples = round(sample_row.size/valid_sample[:,i][valid_sample[:,i]!=32767].size*1000)
        if n_samples < 20:
            n_samples = min(20, sample_row.size)
        #n_samples = min(sample_row.size, 20)#round(n_samples - (n_samples-200)/8))
        #else:
        print(n_samples)
        
        if j==0: row_ids = np.random.choice(sample_row, n_samples, replace=False)
        else: row_ids = np.append(row_ids, np.random.choice(sample_row, n_samples, replace=False))
            
    #valid_sample[row_ids, :] = 32767
    rand_sample = valid_sample[row_ids, :]
    
    return train_sample, bias_sample, rand_sample, row_ids


def class_validation(cover):
    for i, row in train_key.iterrows():
        print(i)
        indices = np.loadtxt(model_traindir+'train_ids_img'+str(i)+'.txt')
        bias_inds = np.loadtxt(model_traindir+'bias_ids_img'+str(i)+'.txt')
        print(indices.shape)
        if (cover==4) & (row['nclasses']==4): 
            if i==0: samples_image = np.array((0,0))
            else: samples_image = np.append(samples_image, 0)
            continue
        train_pts, bias_pts, valid_pts, valid_ids = sample_for_validation(row['ecoreg_code']+'/training_imgs/'+row['train_img_name'], 
                                                                             row['nclasses'],
                                                                             'FCover_'+str(row['year'])+'_'+row['above_tile']+'_rfr_corrected.tif',
                                                                             str(row['year'])+'/'+row['above_tile']+'/feats_full_'+str(row['year'])+'.tif',
                                                                             row['nsamples'], cover, indices.astype(int), bias_inds.astype(int))
        if row['nclasses']==4:
            trees = np.zeros(train_pts.shape[1])
            print(trees.shape)
            train_pts = np.vstack((train_pts, trees))
            
        print(train_pts.shape)
        
        if i==0: 
            valid_pts_full = valid_pts; valid_ids_full = valid_ids
            samples_image = np.array((0,valid_ids.size))
        else: 
            valid_pts_full = np.append(valid_pts_full, valid_pts, axis=0)
            valid_ids_full = np.append(valid_ids_full, valid_ids, axis=0)
            samples_image = np.append(samples_image, samples_image[i]+valid_ids.size)
    
    return valid_pts_full, valid_ids_full, samples_image


# First, post-process and normalize training tile outputs
images = glob.glob('/projectnb/modislc/users/seamorez/HLS_FCover/model_training/*tif')
with open("/projectnb/modislc/users/seamorez/HLS_FCover/model_training/water_bias.pickle", "rb") as f:
    reg0 = pickle.load(f)
with open("/projectnb/modislc/users/seamorez/HLS_FCover/model_training/barren_bias.pickle", "rb") as f:
    reg1 = pickle.load(f)
with open("/projectnb/modislc/users/seamorez/HLS_FCover/model_training/herb_bias.pickle", "rb") as f:
    reg2 = pickle.load(f)
with open("/projectnb/modislc/users/seamorez/HLS_FCover/model_training/shrub_bias.pickle", "rb") as f:
    reg3 = pickle.load(f)
with open("/projectnb/modislc/users/seamorez/HLS_FCover/model_training/trees_bias.pickle", "rb") as f:
    reg4 = pickle.load(f)
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
    
    # Post-process for overprediction of water
    # Reshape
    water_corrected = water_corrected.reshape(1, nx*ny)
    barren_corrected = barren_corrected.reshape(1, nx*ny)
    herb_corrected = herb_corrected.reshape(1, nx*ny)
    shrub_corrected = shrub_corrected.reshape(1, nx*ny)
    trees_corrected = trees_corrected.reshape(1, nx*ny)
    
    # Remove water fraction if fractions from other modeled cover is already >95%
    nonwater = np.concatenate([barren_corrected,herb_corrected,shrub_corrected,trees_corrected], axis=0)
    totals_nonwater = nonwater.sum(axis=0)
    water_corrected[totals_nonwater.reshape(1, nx*ny)>=9500] = 0
    
    # Normalize

    # Water and Barren post-processing step: >= 90% cover post-correction becomes 100% and 0% for all other cover types
    water_corrected[water_corrected>=9500] = 10000
    barren_corrected[water_corrected>=9500] = 0
    herb_corrected[water_corrected>=9500] = 0
    shrub_corrected[water_corrected>=9500] = 0
    trees_corrected[water_corrected>=9500] = 0
    barren_corrected[barren_corrected>=9500] = 10000
    water_corrected[barren_corrected>=9500] = 0
    herb_corrected[barren_corrected>=9500] = 0
    shrub_corrected[barren_corrected>=9500] = 0
    trees_corrected[barren_corrected>=9500] = 0

    # Trees post-processing step: remove fractional trees <= 2% cover (low values lead to inaccuracies since so sparse)
    trees_corrected[trees_corrected<=200] = 0

    # First step to normalize results: joining the fractional cover
    all_corrected = np.concatenate([water_corrected,barren_corrected,herb_corrected,shrub_corrected,trees_corrected], axis=0)
    # Sums, which need to be normalized to 1
    totals_corrected = all_corrected.sum(axis=0)
    # Normalized fractional cover for each class
    all_norm = all_corrected/totals_corrected[:, np.newaxis].T

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
    adj_mask = np.zeros((5,36000000), dtype=int)
    adj_mask[:, adj_i] = 1
    difs = all_norm - all_normf
    mins_i = np.argwhere(difs == np.min(difs, axis=0))
    mins_mask = np.zeros((5,36000000), dtype=int)
    mins_mask[mins_i[:,0], mins_i[:,1]] = 1
    all_norm[(adj_mask==1) & (mins_mask==1)] += 1
    totals_norm = all_norm.sum(axis=0).reshape(1, nx*ny)
    print(np.unique(totals_norm[totals_norm!=100]))
    # 98% case
    adj_i = np.argwhere(totals_norm<100)[:,1]
    adj_mask = np.zeros((5,36000000), dtype=int)
    adj_mask[:, adj_i] = 1
    difs[mins_i[:,0],mins_i[:,1]] = 200
    mins_i = np.argwhere(difs == np.min(difs, axis=0))
    mins_mask = np.zeros((5,36000000), dtype=int)
    mins_mask[mins_i[:,0], mins_i[:,1]] = 1
    all_norm[(adj_mask==1) & (mins_mask==1)] += 1
    totals_norm = all_norm.sum(axis=0).reshape(1, nx*ny)
    print(np.unique(totals_norm[totals_norm!=100]))
    # 101% case
    adj_i = np.argwhere(totals_norm>100)[:,1]
    adj_mask = np.zeros((5,36000000), dtype=int)
    adj_mask[:, adj_i] = 1
    difs = all_norm - all_normf
    maxs_i = np.argwhere(difs == np.max(difs, axis=0))
    maxs_mask = np.zeros((5,36000000), dtype=int)
    maxs_mask[maxs_i[:,0], maxs_i[:,1]] = 1
    all_norm[(adj_mask==1) & (maxs_mask==1)] -= 1
    totals_norm = all_norm.sum(axis=0).reshape(1, nx*ny)
    print(np.unique(totals_norm[totals_norm!=100]))
    # 102% case
    adj_i = np.argwhere(totals_norm>100)[:,1]
    adj_mask = np.zeros((5,36000000), dtype=int)
    adj_mask[:, adj_i] = 1
    difs[maxs_i[:,0],maxs_i[:,1]] = -200
    maxs_i = np.argwhere(difs == np.max(difs, axis=0))
    maxs_mask = np.zeros((5,36000000), dtype=int)
    maxs_mask[maxs_i[:,0], maxs_i[:,1]] = 1
    all_norm[(adj_mask==1) & (maxs_mask==1)] -= 1
    totals_norm = all_norm.sum(axis=0).reshape(1, nx*ny)
    print(np.unique(totals_norm[totals_norm!=100]))
    
    print('Fixed normalizations')
    
    # Reshape and remove values where feat EVI2 is NA for any of phenometrics
    sos_evi2 = full_stack_img[6].reshape(1, nx*ny)
    peak_evi2 = full_stack_img[16].reshape(1, nx*ny)
    eos_evi2 = full_stack_img[26].reshape(1, nx*ny)
    water_norm[(sos_evi2==32767) | (peak_evi2==32767) | (eos_evi2==32767)] = 32767
    barren_norm[(sos_evi2==32767) | (peak_evi2==32767) | (eos_evi2==32767)] = 32767
    herb_norm[(sos_evi2==32767) | (peak_evi2==32767) | (eos_evi2==32767)] = 32767
    shrub_norm[(sos_evi2==32767) | (peak_evi2==32767) | (eos_evi2==32767)] = 32767
    trees_norm[(sos_evi2==32767) | (peak_evi2==32767) | (eos_evi2==32767)] = 32767

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
water_valid, water_valid_ids, water_samples_image = class_validation(0)
barren_valid, barren_valid_ids, barren_samples_image = class_validation(1)
herb_valid, herb_valid_ids, herb_samples_image = class_validation(2)
shrub_valid, shrub_valid_ids, shrub_samples_image = class_validation(3)
trees_valid, trees_valid_ids, trees_samples_image = class_validation(4)

# for i, row in train_key.iterrows():
#     # Get trraining indices
#     indices = np.loadtxt(model_traindir+'train_ids.txt')[i*row['nsamples']:i*row['nsamples']+row['nsamples']]
#     print(indices.shape)
#     train_sample, bias_sample, valid_sample, valid_ids = sample_for_validation(row['ecoreg_code']+'/training_imgs/'+row['train_img_name'], 
#                                                       row['nclasses'],
#                                                       'FCover_'+str(row['year'])+'_'+row['above_tile']+'_rfr_corrected.tif',
#                                                       str(row['year'])+'/'+row['above_tile']+'/feats_full_'+str(row['year'])+'.tif',
#                                                       row['nsamples'],
#                                                       indices.astype(int),
#                                                       bias_ids_full[i*row['nsamples']:i*row['nsamples']+row['nsamples']])
    
#     if i==0: 
#         valid_full = valid_sample; valid_ids_full = valid_ids
#     else: 
#         valid_full = np.append(valid_full, valid_sample, axis=0)
#         valid_ids_full = np.append(valid_ids_full, valid_ids, axis=0)

# Validation scatterplots
fig, ((ax1, ax2, ax5), (ax3, ax4, ax6)) = plt.subplots(2, 3)
ax1.tick_params(axis='both', which='major', labelsize=20)
ax2.tick_params(axis='both', which='major', labelsize=20)
ax3.tick_params(axis='both', which='major', labelsize=20)
ax4.tick_params(axis='both', which='major', labelsize=20)
ax5.tick_params(axis='both', which='major', labelsize=20)
#ax1.set_xticks(fontsize=20)
#ax1.set_yticks(fontsize=18)
fig.set_size_inches(27, 18)

#ax1.scatter(all_class[:,0],all_results[:,0], color='Blue', alpha=0.2)
sb.regplot(x=water_valid[:,0],y=water_valid[:,5], color='#0071DC', scatter_kws={'alpha':0.2}, ax=ax1)
ax1.text(.05, .93, '$R^2$ = '+str(np.corrcoef(water_valid[:,0],water_valid[:,5])[0,1]**2)[:4], fontsize=24)
ax1.text(.05, .88, 'MAE = '+str(100*mean_absolute_error(water_valid[:,0],water_valid[:,5]))[:4]+'%', fontsize=24)
ax1.text(.05, .83, 'RMSE = '+str(100*np.sqrt(mean_squared_error(water_valid[:,0],water_valid[:,5])))[:4]+'%', fontsize=24)
ax1.text(.05, .78, 'Bias = '+str(100*np.mean(water_valid[:,5]-water_valid[:,0]))[:4]+'%', fontsize=24)
ax1.plot([0,1],[0,1], color='Black', linewidth=.5)
ax1.set_xlim(0,1)
ax1.set_ylim(0,1)
ax1.set_title('Water fractional cover',fontsize=32)
ax1.set_xlabel('Observed',fontsize=24)
ax1.set_ylabel('Predicted',fontsize=24)

#ax2.scatter(all_class[:,1],all_results[:,1], color='Grey', alpha=0.2)
sb.regplot(x=barren_valid[:,1],y=barren_valid[:,6], color='Grey', scatter_kws={'alpha':0.2}, ax=ax2)
ax2.text(.05, .93, '$R^2$ = '+str(np.corrcoef(barren_valid[:,1],barren_valid[:,6])[0,1]**2)[:4], fontsize=24)
ax2.text(.05, .88, 'MAE = '+str(100*mean_absolute_error(barren_valid[:,1],barren_valid[:,6]))[:4]+'%', fontsize=24)
ax2.text(.05, .83, 'RMSE = '+str(100*np.sqrt(mean_squared_error(barren_valid[:,1],barren_valid[:,6])))[:4]+'%', fontsize=24)
ax2.text(.05, .78, 'Bias = '+str(100*np.mean(barren_valid[:,6]-barren_valid[:,1]))[:4]+'%', fontsize=24)
ax2.plot([0,1],[0,1], color='Black', linewidth=.5)
ax2.set_xlim(0,1)
ax2.set_ylim(0,1)
ax2.set_title('Barren fractional cover',fontsize=32)
ax2.set_xlabel('Observed',fontsize=24)
ax2.set_ylabel('Predicted',fontsize=24)

#ax3.scatter(all_class[:,2],all_results[:,2], color='YellowGreen', alpha=0.2)
sb.regplot(x=herb_valid[:,2],y=herb_valid[:,7], color='YellowGreen', scatter_kws={'alpha':0.2}, ax=ax3)
ax3.text(.05, .93, '$R^2$ = '+str(np.corrcoef(herb_valid[:,2],herb_valid[:,7])[0,1]**2)[:4], fontsize=24)
ax3.text(.05, .88, 'MAE = '+str(100*mean_absolute_error(herb_valid[:,2],herb_valid[:,7]))[:4]+'%', fontsize=24)
ax3.text(.05, .83, 'RMSE = '+str(100*np.sqrt(mean_squared_error(herb_valid[:,2],herb_valid[:,7])))[:4]+'%', fontsize=24)
ax3.text(.05, .78, 'Bias = '+str(100*np.mean(herb_valid[:,7]-herb_valid[:,2]))[:4]+'%', fontsize=24)
ax3.plot([0,1],[0,1], color='Black', linewidth=.5)
ax3.set_xlim(0,1)
ax3.set_ylim(0,1)
ax3.set_title('Herbaceous fractional cover',fontsize=32)
ax3.set_xlabel('Observed',fontsize=24)
ax3.set_ylabel('Predicted',fontsize=24)

#ax4.scatter(all_class[:,3],all_results[:,3], color='Green', alpha=0.2)
sb.regplot(x=shrub_valid[:,3],y=shrub_valid[:,8], color='Green', scatter_kws={'alpha':0.2}, ax=ax4)
ax4.text(.05, .93, '$R^2$ = '+str(np.corrcoef(shrub_valid[:,3],shrub_valid[:,8])[0,1]**2)[:4], fontsize=24)
ax4.text(.05, .88, 'MAE = '+str(100*mean_absolute_error(shrub_valid[:,3],shrub_valid[:,8]))[:4], fontsize=24)
ax4.text(.05, .83, 'RMSE = '+str(100*np.sqrt(mean_squared_error(shrub_valid[:,3],shrub_valid[:,8])))[:4], fontsize=24)
ax4.text(.05, .78, 'Bias = '+str(100*np.mean(shrub_valid[:,8]-shrub_valid[:,3]))[:4]+'%', fontsize=24)
ax4.plot([0,1],[0,1], color='Black', linewidth=.5)
ax4.set_xlim(0,1)
ax4.set_ylim(0,1)
ax4.set_title('Shrub fractional cover',fontsize=32)
ax4.set_xlabel('Observed',fontsize=24)
ax4.set_ylabel('Predicted',fontsize=24)

#ax5.scatter(all_class[:,3],all_results[:,3], color='Green', alpha=0.2)
sb.regplot(x=trees_valid[:,4],y=trees_valid[:,9], color='darkgreen', scatter_kws={'alpha':0.2}, ax=ax5)
ax5.text(.05, .93, '$R^2$ = '+str(np.corrcoef(trees_valid[:,4],trees_valid[:,9])[0,1]**2)[:4], fontsize=24)
ax5.text(.05, .88, 'MAE = '+str(100*mean_absolute_error(trees_valid[:,4],trees_valid[:,9]))[:4], fontsize=24)
ax5.text(.05, .83, 'RMSE = '+str(100*np.sqrt(mean_squared_error(trees_valid[:,4],trees_valid[:,9])))[:4], fontsize=24)
ax5.text(.05, .78, 'Bias = '+str(100*np.mean(trees_valid[:,9]-trees_valid[:,4]))[:4]+'%', fontsize=24)
ax5.plot([0,1],[0,1], color='Black', linewidth=.5)
ax5.set_xlim(0,1)
ax5.set_ylim(0,1)
ax5.set_title('Trees fractional cover',fontsize=32)
ax5.set_xlabel('Observed',fontsize=24)
ax5.set_ylabel('Predicted',fontsize=24)




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