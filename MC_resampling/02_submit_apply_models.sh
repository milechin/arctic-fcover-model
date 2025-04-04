#!/bin/bash 

module load jq

parameters="/projectnb/modislc/users/seamorez/HLS_FCover/scripts/MC_resampling/model_parameters.json"
tileList="/projectnb/modislc/users/seamorez/HLS_FCover/scripts/techmanuscript_tiles.txt" 

numCores=$( jq .SCC.numCores $parameters )

workDir=$( jq --raw-output .SCC.workDir $parameters )
jobTime="$(date +%Y_%m_%d_%H_%M_%S)"
logDir=$( jq --raw-output .SCC.logDir $parameters ) 
mkdir -p $logDir

startYr=$( jq --raw-output .setup.startYr $parameters )
endYr=$( jq --raw-output .setup.endYr $parameters )
numModels=$( jq --raw-output .setup.numModels $parameters )
s1=$( jq --raw-output .setup.s1 $parameters )

nodeArgs="-l h_rt=28:00:00 -pe omp ${numCores}"

for ((model=21; model<=20+$numModels; model++))
do
    echo "model: $model"
    
    while read -r tile
     do
   	   #modelDir="${workDir}${i}/${tile}/"
   	   #mkdir -p $modelDir
       nameArg="-N Py_${model}_${i}_${tile}"
       logArg="-o ${logDir}Run_${model}_${i}_${tile}_${jobTime}.txt"
       qsub $nameArg $logArg $nodeArgs run_apply_models.sh $tile $startYr $endYr $model
           
     done < $tileList
    
done
