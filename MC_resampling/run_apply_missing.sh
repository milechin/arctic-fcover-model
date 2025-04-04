#!/bin/bash -l
#$ -j y

#SCC
#Load required modules
module load jq
module load miniconda
module load geos/3.10.2
module load gdal/3.4.3
module load sqlite3/3.37.2
# module load grass
# module load R/4.2.1
# module load rstudio/2022.07.2-576

#Start environment
conda activate landsat_lsp_env

parameters="/projectnb/modislc/users/seamorez/HLS_FCover/scripts/MC_resampling/model_parameters.json"

tile=$1
start_year=$2
end_year=$3
model=$4
workDir=$( jq --raw-output .SCC.workDir $parameters )
pyScript=$( jq --raw-output .SCC.pyScript $parameters )

#Run the script
######################
# enable Numpy calls to maybe use more cores
export OPENBLAS_NUM_THREADS=$NSLOTS
export GDAL_NUM_THREADS=$NSLOTS
python "/projectnb/modislc/users/seamorez/HLS_FCover/scripts/mc_run_missing.py" $tile $start_year $end_year $model
