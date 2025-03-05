import glob
import os
import numpy as np
import rasterio

#### PARAMETERS TO BE PASSED IN ####
outdir = '/projectnb/modislc/users/seamorez/HLS_FCover/output/'
s1_outdir = '/projectnb/modislc/users/seamorez/HLS_FCover/S1_input/retiled/'
indir = '/projectnb/modislc/users/seamorez/HLS_FCover/input/HLS30/'
start_yr = 2016
end_yr = 2023
phenometrics = ['50PCGI','Peak','50PCGD']
metrics = ['dem', 'water', 'slope', 'aspect']
do_s1 = True  # Bool for testing S1

#### PARAMETERS ####
years = np.arange(start_yr, end_yr+1, 1)

################################### STEP 1 ###################################
def add_feats(img):
    # Vegetation indices 
    evi2 = 2.5*(img[3]*.0001 - img[2]*.0001) / (img[3]*.0001 + 2.4*img[2]*.0001 + 1)
    tcg = -0.2941*img[0] + -0.243*img[1] + -0.5424*img[2] + 0.7276*img[3] + 0.0713*img[4] + -0.1608*img[5]
    tcw = 0.1511*img[0] + 0.1973*img[1] + 0.3283*img[2] + 0.3407*img[3] + -0.7117*img[4] + -0.4559*img[5]
    tcb = 0.3029*img[0] + 0.2786*img[1] + 0.4733*img[2] + 0.5599*img[3] + 0.508*img[4] + 0.1872*img[5]

    # Scaling, rounding, and nd masking
    evi2 = np.around(evi2*10000)
    evi2[img[3]==32767]=32767
    tcg[img[3]==32767]=32767; tcw[img[3]==32767]=32767; tcb[img[3]==32767]=32767
    tcg = np.around(tcg); tcw = np.around(tcw);tcb = np.around(tcb)

    # Stack bands
    evi2 = evi2.reshape(1,6000,6000); tcg = tcg.reshape(1,6000,6000); tcw = tcw.reshape(1,6000,6000); tcb = tcb.reshape(1,6000,6000)
    feats_full = np.concatenate((img, evi2,tcg,tcw,tcb), axis=0)
    
    return feats_full


# Iterate through years 
for year in years:  # CHANGE THIS WHEN READY
    # Iterate through ABoVE tile dirs
    tiledirs = glob.glob(outdir+str(year)+'/Bh*')
    print(year, len(tiledirs))
    for tiledir in tiledirs:
        # If full feats already exist, print statement and continue
        if os.path.exists(tiledir+'/feats_full_'+str(year)+'.tif'):
            if not do_s1:
                print('Full feats already exists!')
                continue
            else:
                tile = tiledir.split('/')[8]
                if not os.path.exists(s1_outdir+tile[1:]+'_summer_'+str(year)+'.tif'):
                    continue
        
        # Iterate through phenometric sythetic+composite spectra
        tile = tiledir.split('/')[8]
        print(tile)
        # 50PCGI
        sos_r = rasterio.open(tiledir+'/'+tile+'_'+str(year)+'_'+'50PCGI_final.tif', 'r')
        sos = sos_r.read()
        # Peak
        peak_r = rasterio.open(tiledir+'/'+tile+'_'+str(year)+'_'+'Peak_final.tif', 'r')
        peak = peak_r.read()
        # 50PCGD
        eos_r = rasterio.open(tiledir+'/'+tile+'_'+str(year)+'_'+'50PCGD_final.tif', 'r')
        eos = eos_r.read()
        
        # Solve for TCG, TCW, TCI, and EVI2
        sos_full = add_feats(sos)
        peak_full = add_feats(peak)
        eos_full = add_feats(eos)
        
        # Topo features
        dem = rasterio.open(outdir+'2016/'+tile+'/'+tile+'_'+'dem_final.tif', 'r').read()
        slope = rasterio.open(outdir+'2016/'+tile+'/'+tile+'_'+'slope_final.tif', 'r').read()
        slope[slope==65535] = 32767   # replacing uint16 nd value with int16 nd value
        aspect = rasterio.open(outdir+'2016/'+tile+'/'+tile+'_'+'aspect_final.tif', 'r').read()
        aspect = aspect - 32768       # shifting to fit in int16 (original in uint16)
        dem = dem.reshape(1,6000,6000); slope = slope.reshape(1,6000,6000); aspect = aspect.reshape(1,6000,6000)

        # Update metadata
        meta = peak_r.meta
        meta.update(count=33)

        # Stack all features for this year, tile
        full_stack = np.concatenate((sos_full,peak_full,eos_full, dem,slope,aspect), axis=0)
        print(full_stack.shape)
        
        # Save out new full features raster with spectra (6x3), indices (4x3), and topo (3)
        with rasterio.open(tiledir+'/'+'feats_full_'+str(year)+'.tif', 'w', **meta) as dst:
            for idx,band in enumerate(np.linspace(0,full_stack.shape[0]-1, full_stack.shape[0], dtype=int), start=1):
                dst.write(full_stack[band], idx)
                
        # If we're adding S1 features, make a separate set of full features
        if do_s1 & os.path.exists(s1_outdir+tile[1:]+'_summer_'+str(year)+'.tif'):
            print('Adding Sentinel 1 for tile: '+tile)
            try:
                s1_summer = rasterio.open(s1_outdir+tile[1:]+'_summer_'+str(year)+'.tif', 'r').read()
                s1_winter = rasterio.open(s1_outdir+tile[1:]+'_winter_'+str(year)+'.tif', 'r').read()
                
                # Update metadata and stack
                meta.update(count=37)
                full_stack = np.concatenate((full_stack, s1_summer,s1_winter), axis=0)
            except:
                print('Warning: Missing summer or winter S1 data for tile '+tile+'. S1 data not added to features!')
            
            # Save out new full features raster with spectra (6x3), indices (4x3), topo (3), and S1 (2x2)
            with rasterio.open(tiledir+'/'+'feats_full_s1_'+str(year)+'.tif', 'w', **meta) as dst:
                for idx,band in enumerate(np.linspace(0,full_stack.shape[0]-1, full_stack.shape[0], dtype=int), start=1):
                    dst.write(full_stack[band], idx)