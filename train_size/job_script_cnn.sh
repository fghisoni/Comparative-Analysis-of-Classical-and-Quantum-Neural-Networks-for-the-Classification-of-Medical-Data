#!/bin/bash
#SBATCH --job-name=cnn
#SBATCH --partition ulow
#SBATCH -D /home/ghisoni/quantum_computing/medical_qnn/final/train_size
#SBATCH -c 1
#SBATCH --mem 1G
#SBATCH -t 2-23:59:59
#SBATCH --error=errorQML
#SBATCH --output=outputQML
#SBATCH --mail-type=ALL
#SBATCH --mail-user=francesco.ghisoni01@universitadipavia.it

# ----------- actual script to run ---------------
python classical_network_cluster_exp_4_configs.py $1 $2 $3
