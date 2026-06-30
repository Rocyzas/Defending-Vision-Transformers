#!/bin/bash

#SBATCH --job-name=vit_training
#SBATCH -p Teaching
#SBATCH --gres=gpu:1

. /home/htang2/toolchain-20251006/toolchain.rc
. venv/bin/activate

set -e

SCRATCH=/disk/scratch/$USER/$SLURM_JOB_ID
OUTPUTS_DIR=$SCRATCH/outputs
mkdir -p $SCRATCH
mkdir -p $OUTPUTS_DIR

echo "[$(date '+%y-%m-%d %H:%M:%S')] Unzipping dataset to scratch..."
unzip -q ./data.zip -d $SCRATCH/

echo "[$(date '+%y-%m-%d %H:%M:%S')] Copying python file to scratch..."
cp ./src/Data_Augmented_Training_vit.py $SCRATCH/Data_Augmented_Training_vit.py

# I think this is the fix but I cant exactly remember the error msg
echo "[$(date '+%y-%m-%d %H:%M:%S')] Copying source directory to scratch..."
cp -r ./src $SCRATCH

echo "[$(date '+%y-%m-%d %H:%M:%S')] Verify setup..."
ls -lh $SCRATCH

echo "[$(date '+%y-%m-%d %H:%M:%S')] Starting training..."

cd $SCRATCH
python3 Data_Augmented_Training_vit.py \
    --data data/dataset.csv \
    --models outputs
    
echo "[$(date '+%y-%m-%d %H:%M:%S')] Copy results to home directory..."
cd $SCRATCH
zip -r outputs.zip outputs
cp outputs.zip ~

rm -rf $SCRATCH