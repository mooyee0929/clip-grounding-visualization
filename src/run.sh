#!/bin/bash
#SBATCH --job-name=clip_hw6
#SBATCH --output=hw6_%j.out
#SBATCH --error=hw6_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END,FAIL


source ~/miniconda3/etc/profile.d/conda.sh
conda activate HW6



jupyter nbconvert --to notebook --execute clip_interpretation.ipynb \
    --output clip_interpretation_output.ipynb \
    --ExecutePreprocessor.timeout=3600

echo "Notebook last modified time:"
stat clip_interpretation.ipynb


echo "============================================"
echo "Job finished: $(date)"
echo "============================================"