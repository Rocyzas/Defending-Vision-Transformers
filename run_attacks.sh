#!/bin/bash

#SBATCH --job-name=attack
#SBATCH -p Teaching
#SBATCH --gres=gpu:1

. /home/htang2/toolchain-20251006/toolchain.rc
. venv/bin/activate

set -e

SCRATCH=/disk/scratch/$USER/$SLURM_JOB_ID
OUTPUTS_DIR=$SCRATCH/outputs
mkdir -p $SCRATCH
mkdir -p $OUTPUTS_DIR

# MODELTYPE=hybrid
# TARGET_MODEL_NAME=best_vit_model_96x96_grid_patch_16x16_9_snp_fillup_0.1

# # ---- Defaults (used if the flag isn't passed) ----
# MODELTYPE=hybrid
# TARGET_MODEL_NAME=best_vit_model_96x96_grid_patch_16x16_9_snp_fillup_0.1

# ---- Override from command-line flags ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --modelType)
            MODELTYPE="$2"
            shift 2
            ;;
        --TargetModelName)
            TARGET_MODEL_NAME="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

TARGET_MODEL=$SCRATCH/best_${MODELTYPE}/${TARGET_MODEL_NAME}.pth

echo "[$(date '+%y-%m-%d %H:%M:%S')] Unzipping Testing to scratch..."
unzip -q ./Testing_new.zip -d $SCRATCH/

# echo "[$(date '+%y-%m-%d %H:%M:%S')] Copying python file to scratch..."
# cp ./src/a.py $SCRATCH/Data_Augmented_Training_hybrid.py

# I think this is the fix but I cant exactly remember the error msg
echo "[$(date '+%y-%m-%d %H:%M:%S')] Copying source directory to scratch..."
cp -r ./src $SCRATCH

echo "[$(date '+%y-%m-%d %H:%M:%S')] Copying attack suite file to scratch..."
cp -r ./attack_suite.py $SCRATCH

echo "[$(date '+%y-%m-%d %H:%M:%S')] Copying GreedyPixel..."
cp -r ./greedypixel $SCRATCH

# COPYING MODELS
echo "[$(date '+%y-%m-%d %H:%M:%S')] Copying Models..."
unzip -q ./best_$MODELTYPE.zip -d $SCRATCH/


echo "[$(date '+%y-%m-%d %H:%M:%S')] Verify setup..."
ls -lh $SCRATCH

echo "[$(date '+%y-%m-%d %H:%M:%S')] Starting attacking a single model - $TARGET_MODEL..."
#python3 $SCRATCH/Data_Augmented_Training_hybrid.py --data $SCRATCH/data/dataset.csv --models $OUTPUTS_DIR
cd $SCRATCH
python3 attack_suite.py \
    $TARGET_MODEL \
    $SCRATCH/Testing \
    --max-samples 1
    # --output-csv ~/ViT_defense/Defending-Vision-Transformers/attack_results.csv


echo "[$(date '+%y-%m-%d %H:%M:%S')] Copying attack_results from scratch to home directory..."
cp -r $SCRATCH/attack_results.csv ~/ViT_defense/Defending-Vision-Transformers/attack_results_${MODELTYPE}_${TARGET_MODEL_NAME}.csv
# echo "[$(date '+%y-%m-%d %H:%M:%S')] Copy results to home directory..."
# cd $SCRATCH
# zip -r outputs.zip outputs
# cp outputs.zip ~

rm -rf $SCRATCH