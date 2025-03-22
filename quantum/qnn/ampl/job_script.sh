#!/bin/bash
#SBATCH --job-name=ce_amp
#SBATCH --partition ulow
#SBATCH -D /home/ghisoni/quantum_computing/medical_qnn/final/quantum/qnn/ampl
#SBATCH -c 1
#SBATCH --mem 128G
#SBATCH -t 2-23:59:59
#SBATCH --error=errorQML
#SBATCH --output=outputQML
#SBATCH --mail-type=ALL
#SBATCH --mail-user=francesco.ghisoni01@universitadipavia.it

# ----------- actual script to run ---------------
python ce_ampl.py $1 $2
