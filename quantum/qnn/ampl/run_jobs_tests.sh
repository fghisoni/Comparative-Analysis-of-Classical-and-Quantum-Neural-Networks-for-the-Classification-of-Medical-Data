#!/bin/bash

for seed in {1..1..1}
 do
    sbatch job_script.sh 1 $seed
 done

