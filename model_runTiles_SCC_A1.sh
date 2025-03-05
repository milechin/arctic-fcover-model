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

parameters="/projectnb/modislc/users/seamorez/HLS_FCover/scripts/model_parameters_A1.json"

tile=$1
year=$2
workDir=$( jq --raw-output .SCC.workDir $parameters )
pyScript=$( jq --raw-output .SCC.pyScript $parameters )

#Run the script
######################
python "$pyScript" $tile $year
