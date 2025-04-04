#!/bin/bash 

module load jq

parameters="/projectnb/modislc/users/seamorez/HLS_FCover/scripts/model_parameters.json"
tileList="/projectnb/modislc/users/seamorez/HLS_FCover/scripts/add_tiles.txt" 
#tileList="/projectnb/modislc/users/seamorez/HLS_FCover/scripts/add_tiles.txt"  #CHANGE AFTER TEST TO ABOVE LINE

numCores=$( jq .SCC.numCores $parameters )

workDir=$( jq --raw-output .SCC.workDir $parameters )
jobTime="$(date +%Y_%m_%d_%H_%M_%S)"
logDir=$( jq --raw-output .SCC.logDir $parameters ) 
mkdir -p $logDir

startYr=$( jq --raw-output .setup.startYr $parameters )
endYr=$( jq --raw-output .setup.endYr $parameters )

nodeArgs="-l h_rt=8:00:00 -pe omp ${numCores}"

for ((i=$startYr; i<=$endYr; i++))
do
   echo "i: $i"

   while read -r tile
    do
	#modelDir="${workDir}${i}/${tile}/"
	#mkdir -p $modelDir
        nameArg="-N Py_${i}_${tile}"
	logArg="-o ${logDir}Run_${i}_${tile}_${jobTime}.txt"
        qsub $nameArg $logArg $nodeArgs model_runTiles_SCC.sh $tile $i
        
    done < $tileList

done



