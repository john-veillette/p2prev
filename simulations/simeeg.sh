#!/bin/bash
#SBATCH --job-name=simeeg
#SBATCH --output=out/simeeg_%A_%a.out
#SBATCH --error=out/simeeg_%A_%a.err
#SBATCH --time=00:06:00
#SBATCH --partition=caslake
#SBATCH --account=pi-hcn1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --array=1-1000

cd ~/p2prev/simulations
source activate pymc_env

# tell pytensor to compile to a different directory for each job
export PYTENSOR_FLAGS="compiledir_format=compiledir_${SLURM_ARRAY_TASK_ID}_eeg,base_compiledir=/scratch/midway3/${USER}/compiledir"

python simulate_eeg.py $SLURM_ARRAY_TASK_ID