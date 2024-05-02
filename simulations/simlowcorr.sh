#!/bin/bash
#SBATCH --job-name=simlowcorr
#SBATCH --output=out/simlowcorr_%A_%a.out
#SBATCH --error=out/simlowcorr_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --partition=caslake
#SBATCH --account=pi-hcn1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --array=3-60:3

cd ~/p2prev/simulations
source activate pymc_env

# tell pytensor to compile to a different directory for each job
export PYTENSOR_FLAGS="compiledir_format=compiledir_${SLURM_ARRAY_TASK_ID}_lowcorr,base_compiledir=/scratch/midway3/${USER}/compiledir"

python simulate_correlation.py $SLURM_ARRAY_TASK_ID low


