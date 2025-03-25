#!/bin/bash

for t_size in {3..7..2}
   do
   for seed in {1..10..1}
      do 
      sbatch job_script_qnn.sh $t_size $seed
   done
done
