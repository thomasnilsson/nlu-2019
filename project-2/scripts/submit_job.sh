#!/bin/sh
SCRIPT=$1
EXTRA=$2
TIMESTAMP=$(date +%y%m%d%H%M%S)
bsub -n 6 -N -W 4:00 -R "rusage[mem=2048, ngpus_excl_p=1]" -o nlu_job_${TIMESTAMP}.out $EXTRA < $SCRIPT
