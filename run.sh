#!/bin/sh
#SBATCH --account=pinaki.sarder
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32gb
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --output=logs/model400_bs8.out
#SBATCH --job-name="intima_seg"
echo  "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST="$SLURM_JOB_NODELIST
echo "SLURM_NNODES="$SLURM_NNODEs
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR
 
ulimit -s unlimited
module load conda
pwd
ml
date

# Add your userID here:
USER=anish.tatke

# Add the name of project
PROJECT=intima-seg

module load conda
conda activate PTEnv
 
CUDA_LAUNCH_BLOCKING=0
 
python3 train.py --epochs=300 --batch_size=8 --folds=5