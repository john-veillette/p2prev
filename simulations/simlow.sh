#!/bin/bash
#SBATCH --job-name=simlow
#SBATCH --output=out/simlow_%A_%a.out
#SBATCH --error=out/simlow_%A_a%.err
#SBATCH --time=06:00:00
#SBATCH --partition=caslake
#SBATCH --account=pi-hcn1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --array=3-60:3

cd ~/p2prev/simulations
source activate pymc_env

# tell pytensor to compile to a different directory for each job
export PYTENSOR_FLAGS="compiledir_format=compiledir_${SLURM_ARRAY_TASK_ID}_low,base_compiledir=/scratch/midway3/${USER}/compiledir"

python simulate_classification.py $SLURM_ARRAY_TASK_ID low


