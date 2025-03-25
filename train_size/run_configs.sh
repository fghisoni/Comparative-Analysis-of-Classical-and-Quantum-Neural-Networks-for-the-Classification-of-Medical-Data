#!/bin/bash

INPUT_FILE="test_configs.txt"

while IFS= read -r line
do
    for tsize in {1..10}
    do
        for param in {1..10}
        do
        sbatch job_script_cnn.sh "$line" "$param" "$tsize"
        done
    done
done < "$INPUT_FILE"

