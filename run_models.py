import sys
import time
import pickle
import numpy as np
import pandas as pd
import rasterio
from sklearn.ensemble import RandomForestRegressor

start_time = time.time()

tile = sys.argv[1]
year = sys.argv[2]
model = '_'+str(sys.argv[3]) if len(sys.argv) > 3 else ''

#### PARAMETERS TO BE PASSED IN ####
outdir = '/projectnb/modislc/users/seamorez/HLS_FCover/output/'
model_traindir = '/projectnb/modislc/users/seamorez/HLS_FCover/model_training/'
model_outdir = '/projectnb/modislc/users/seamorez/HLS_FCover/model_outputs/'

################################### STEP 1 ###################################
# Prepare models and data
# Read in models
trees_regressor = pickle.load(open(model_traindir+"trees_rfr"+model+".pickle", "rb"))
shrub_regressor = pickle.load(open(model_traindir+"shrub_rfr"+model+".pickle", "rb"))
herb_regressor = pickle.load(open(model_traindir+"herb_rfr"+model+".pickle", "rb"))
barren_regressor = pickle.load(open(model_traindir+"barren_rfr"+model+".pickle", "rb"))
water_regressor = pickle.load(open(model_traindir+"water_rfr"+model+".pickle", "rb"))

# Read in raster
full_stack = rasterio.open(outdir+year+'/'+tile+'/feats_full_'+year+'.tif','r')
meta = full_stack.meta
full_stack_img = full_stack.read()
nsamples, nx, ny = full_stack_img.shape 
full_stack_img_2d = full_stack_img.reshape((nsamples,nx*ny))
full_stack_img_2d = full_stack_img_2d.T

#water_stack_img_2d = full_stack_img_2d[:,[3,4,5,7,8,13,14,15,17,18,23,24,25,27,28,30,31,32]]

################################### STEP 2 ###################################
# Predict fractional cover
water_predict = water_regressor.predict(full_stack_img_2d)
barren_predict = barren_regressor.predict(full_stack_img_2d)
herb_predict = herb_regressor.predict(full_stack_img_2d)
shrub_predict = shrub_regressor.predict(full_stack_img_2d)
trees_predict = trees_regressor.predict(full_stack_img_2d)

################################### STEP 3 ###################################
# Correct bias
# Read in bias models 
water_bias = pickle.load(open(model_traindir+"water_bias"+model+".pickle", "rb"))
barren_bias = pickle.load(open(model_traindir+"barren_bias"+model+".pickle", "rb"))
herb_bias = pickle.load(open(model_traindir+"herb_bias"+model+".pickle", "rb"))
shrub_bias = pickle.load(open(model_traindir+"shrub_bias"+model+".pickle", "rb"))
trees_bias = pickle.load(open(model_traindir+"trees_bias"+model+".pickle", "rb"))

# Correct for each class
water_corrected = water_predict.reshape(nx*ny) - water_bias.predict(water_predict.reshape(nx*ny,1))
barren_corrected = barren_predict.reshape(nx*ny) - barren_bias.predict(barren_predict.reshape(nx*ny,1))
herb_corrected = herb_predict.reshape(nx*ny) - herb_bias.predict(herb_predict.reshape(nx*ny,1))
shrub_corrected = shrub_predict.reshape(nx*ny) - shrub_bias.predict(shrub_predict.reshape(nx*ny,1))
trees_corrected = trees_predict.reshape(nx*ny) - trees_bias.predict(trees_predict.reshape(nx*ny,1))

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
water_corrected[totals_nonwater.reshape(1, nx*ny)>=.95] = 0
# Remove barren fraction if fractions from other modeled cover is already >95%
nonbarren = np.vstack((water_corrected,herb_corrected,shrub_corrected,trees_corrected))
totals_nonbarren = nonbarren.sum(axis=0)
barren_corrected[(totals_nonbarren>=0.97) & (barren_corrected>=.04)] = 0
# Remove tree fraction if fractions from other modeled cover is already >95%
nontree = np.vstack((water_corrected,barren_corrected,herb_corrected,shrub_corrected))
totals_nontree = nontree.sum(axis=0)
trees_corrected[(totals_nontree>=0.95) & (trees_corrected>=.10)] = 0

# Phenology masking
try:
    with rasterio.open(outdir+year+'/'+tile+'/'+tile+'_'+year+'_pheno_final.tif','r') as src0:
        pheno = src0.read(1).reshape(1,6000*6000)
except:
    print('No pheno file yet for this tile!')
    pheno = np.zeros_like(water_corrected)
with rasterio.open(outdir+'2016/'+tile+'/'+tile+'_water_final.tif','r') as src1:
    water_presence = src1.read().reshape(1,6000*6000)
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
water_corrected[water_corrected>=.95] = 1
barren_corrected[water_corrected>=.95] = 0
herb_corrected[water_corrected>=.95] = 0
shrub_corrected[water_corrected>=.95] = 0
trees_corrected[water_corrected>=.95] = 0
barren_corrected[barren_corrected>=.95] = 1
water_corrected[barren_corrected>=.95] = 0
herb_corrected[barren_corrected>=.95] = 0
shrub_corrected[barren_corrected>=.95] = 0
trees_corrected[barren_corrected>=.95] = 0

# Trees post-processing step: remove fractional trees <= 2% cover (low values lead to inaccuracies since so sparse)
trees_corrected[trees_corrected<=.02] = 0

################################### STEP 4 ###################################
# Normalize
# First step to normalize results: joining the fractional cover
all_corrected = np.concatenate([water_corrected,barren_corrected,herb_corrected,shrub_corrected, trees_corrected], axis=0)
# Sums, which need to be normalized to 1
totals_corrected = all_corrected.sum(axis=0)
# Normalized fractional cover for each class
all_norm = all_corrected/totals_corrected[:, np.newaxis].T

# Separating back out to classes after normalization
water_normf = all_norm[0].reshape(1, nx*ny)
barren_normf = all_norm[1].reshape(1, nx*ny)
herb_normf = all_norm[2].reshape(1, nx*ny)
shrub_normf = all_norm[3].reshape(1, nx*ny)
trees_normf = all_norm[4].reshape(1, nx*ny)

# Removing no-data values and scaling (SOME MAY SUM TO >100 THIS WAY)
water_normf = water_normf*100
barren_normf = barren_normf*100
herb_normf = herb_normf*100
shrub_normf = shrub_normf*100
trees_normf = trees_normf*100
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

with rasterio.open(model_outdir+year+'/FCover_'+tile+'_'+year+'_rfr.tif', 'w', **meta2) as dst:
    for id,row in enumerate(results_list, start=1):
        dst.write_band(id, row)


end_time = time.time()
elapsed_time = end_time - start_time
# Convert to hours, minutes, and seconds
hours = int(elapsed_time // 3600)
minutes = int((elapsed_time % 3600) // 60)
seconds = int(elapsed_time % 60)

# Print to log file
print(f"Job completed in {hours}h {minutes}m {seconds}s.")

# ## TESTING
# file1 = open("/projectnb/modislc/users/seamorez/HLS_FCover/scripts/"+tile+'_'+year+".txt","w")
# L = [tile,"\n",year,"\n"]
 
# # \n is placed to indicate EOL (End of Line)
# file1.writelines(L)
# file1.close() #to change file access modes