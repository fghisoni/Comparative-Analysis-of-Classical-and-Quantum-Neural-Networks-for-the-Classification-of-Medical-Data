#!/bin/bash

INPUT_FILE="test_configs.txt"

while IFS= read -r line
do
    for param in {1..1}
    do
        sbatch job_script_configs.sh "$line" "$param"
    done
done < "$INPUT_FILE"

