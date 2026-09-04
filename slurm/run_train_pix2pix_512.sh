#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=168:0:00
#SBATCH --account=def-example
#SBATCH --mail-user=example@example.ca
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:v100:1

# 512x512 pix2pix training run.
# Usage: sbatch run_train_pix2pix_512.sh <dataset_name>

NAME=${1:-facades}

source ~/Example/bin/activate

cd /home/$USER/projects/def-example/pytorch-CycleGAN-and-pix2pix

#cp -r datasets/$NAME $SLURM_TMPDIR/datasets
#cp -r checkpoints/ $SLURM_TMPDIR/checkpoints

# Changes vs. the 256x256 defaults in train.sh:
#   --load_size 572 --crop_size 512   512 crops, keeping the paper's 1.117x random-jitter ratio (286/256)
#   --batch_size 1                    memory scales quadratically with resolution; 2-4 may still fit on 32GB
#   --netD n_layers --n_layers_D 4    142x142 receptive field instead of the basic PatchGAN's 70x70
#   --num_threads 8                   match --cpus-per-task, since decoding larger images costs more CPU
# The generator stays at pix2pix's default unet_256: it is fully convolutional, so 512 inputs
# simply bottleneck at 2x2 instead of 1x1.
srun python train.py \
    --dataroot ./datasets/$NAME \
    --name pix2pix_${NAME}_512 \
    --model pix2pix \
    --direction AtoB \
    --preprocess resize_and_crop \
    --load_size 572 \
    --crop_size 512 \
    --batch_size 1 \
    --netD n_layers \
    --n_layers_D 4 \
    --n_epochs 50 \
    --n_epochs_decay 50 \
    --save_epoch_freq 20 \
    --num_threads 8 \
    --display_id -1

srun python test.py \
    --dataroot ./datasets/$NAME \
    --name pix2pix_${NAME}_512 \
    --model pix2pix \
    --preprocess resize_and_crop \
    --load_size 512 \
    --crop_size 512
