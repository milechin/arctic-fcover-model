#!/bin/bash 

module load jq

parameters="/projectnb/modislc/users/seamorez/HLS_FCover/scripts/MC_resampling/model_parameters.json"

numCores=$( jq .SCC.numCores $parameters )

workDir=$( jq --raw-output .SCC.workDir $parameters )
jobTime="$(date +%Y_%m_%d_%H_%M_%S)"
logDir=$( jq --raw-output .SCC.logDir $parameters ) 
mkdir -p $logDir

startYr=$( jq --raw-output .setup.startYr $parameters )
endYr=$( jq --raw-output .setup.endYr $parameters )
numModels=$( jq --raw-output .setup.numModels $parameters )
s1=$( jq --raw-output .setup.s1 $parameters )

nodeArgs="-l h_rt=12:00:00 -pe omp ${numCores}"

for ((i=21; i<=20+$numModels; i++))
do
   echo "i: $i"
   
    nameArg="-N Py_${i}"
	logArg="-o ${logDir}Model_${i}_${jobTime}.txt"
    qsub $nameArg $logArg $nodeArgs run_train_models.sh $i $s1
    
done
