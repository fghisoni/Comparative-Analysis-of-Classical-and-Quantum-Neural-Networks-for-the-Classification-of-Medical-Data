#!/bin/bash

for layer in {11..12..1}
   do
   for seed in {1..10..1}
      do 
      sbatch job_script.sh $layer $seed
   done
done
