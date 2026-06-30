#!/bin/bash

#SBATCH --job-name=hybrid_training
#SBATCH -p Teaching
#SBATCH --gres=gpu:1

. /home/htang2/toolchain-20251006/toolchain.rc
. venv/bin/activate

set -e

SCRATCH=/disk/scratch/$USER/$SLURM_JOB_ID
OUTPUTS_DIR=$SCRATCH/outputs

echo "[$(date '+%y-%m-%d %H:%M:%S')] Unzipping dataset to scratch..."
unzip -q ./data.zip -d $SCRATCH/
echo "[$(date '+%y-%m-%d %H:%M:%S')] Verify unzip..."
ls -lh .

echo "[$(date '+%y-%m-%d %H:%M:%S')] Starting training..."
python3 ~/Data_Augmented_Training_hybrid.py --data $SCRATCH/dataset.csv --models $OUTPUTS_DIR

echo "[$(date '+%y-%m-%d %H:%M:%S')] Copy results to home directory..."
cd $SCRTACH
zip -r outputs.zip outputs
cp outputs.zip ~

rm -rf $SCRATCH