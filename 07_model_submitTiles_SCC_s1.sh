#!/bin/bash 

module load jq

parameters="/projectnb/modislc/users/seamorez/HLS_FCover/scripts/model_parameters_s1.json"
tileList="/projectnb/modislc/users/seamorez/HLS_FCover/scripts/s1_tiles.txt" 
#tileList="/projectnb/modislc/users/seamorez/HLS_FCover/scripts/add_tiles.txt"  #CHANGE AFTER TEST TO ABOVE LINE

numCores=$( jq .SCC.numCores $parameters )

workDir=$( jq --raw-output .SCC.workDir $parameters )
jobTime="$(date +%Y_%m_%d_%H_%M_%S)"
logDir=$( jq --raw-output .SCC.logDir $parameters ) 
mkdir -p $logDir

startYr=$( jq --raw-output .setup.startYr $parameters )
endYr=$( jq --raw-output .setup.endYr $parameters )

nodeArgs="-l h_rt=8:00:00 -pe omp ${numCores}"

# Read tile and year from the file
while IFS=',' read -r tile year
do
    year=$(echo "$year" | tr -d ' ')  # Remove any spaces around the year
    echo "Processing Tile: $tile for Year: $year"

    nameArg="-N Py_${year}_${tile}"
    logArg="-o ${logDir}Run_${year}_${tile}_${jobTime}.txt"
    
    qsub $nameArg $logArg $nodeArgs model_runTiles_SCC_s1.sh $tile $year

done < "$tileList"
