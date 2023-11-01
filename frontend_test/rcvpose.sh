#!/bin/bash -l
# SLURM SUBMIT SCRIPT
#SBATCH --account=ingenuitylabs
#SBATCH --partition=Aurora
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=output/%j.out
#SBATCH --error=output/%j.err
#SBATCH --time=05:00:00

python test.py --frontend accumulator --root_dataset /ingenuity_NAS/datasets/public/RCVLab/Bluewrist/16yw113/