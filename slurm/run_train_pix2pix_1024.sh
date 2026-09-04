#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=168:0:00
#SBATCH --account=def-example
#SBATCH --mail-user=example@example.ca
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:v100:1
#SBATCH --constraint=cascade,v100

# 1024x1024 pix2pix training run.
# Usage: sbatch run_train_pix2pix_1024.sh <dataset_name>
#
# Needs a 32GB GPU. The --constraint line above pins the V100-32GB nodes; swap it for
# "#SBATCH --gres=gpu:a100:1" (and drop --constraint) on a cluster with A100s.

NAME=${1:-facades}

source ~/Example/bin/activate

cd /home/$USER/projects/def-example/pytorch-CycleGAN-and-pix2pix

#cp -r datasets/$NAME $SLURM_TMPDIR/datasets
#cp -r checkpoints/ $SLURM_TMPDIR/checkpoints

# Changes vs. the 256x256 defaults in train.sh:
#   --load_size 1144 --crop_size 1024  1024 crops, keeping the paper's 1.117x random-jitter ratio (286/256)
#   --batch_size 1                     required: activation memory is ~16x the 256x256 run
#   --netD n_layers --n_layers_D 5     286x286 receptive field; the basic PatchGAN's 70x70 covers <7% of a 1024 image
#   --num_threads 8                    match --cpus-per-task, since decoding larger images costs more CPU
#   --save_latest_freq 2000            checkpoints are expensive to redo at this resolution
# The generator stays at pix2pix's default unet_256: it is fully convolutional, so 1024 inputs
# simply bottleneck at 4x4 instead of 1x1. resnet_9blocks only downsamples twice, which would
# leave a 256x256 latent and blow up VRAM.
srun python train.py \
    --dataroot ./datasets/$NAME \
    --name pix2pix_${NAME}_1024 \
    --model pix2pix \
    --direction AtoB \
    --preprocess resize_and_crop \
    --load_size 1144 \
    --crop_size 1024 \
    --batch_size 1 \
    --netD n_layers \
    --n_layers_D 5 \
    --n_epochs 50 \
    --n_epochs_decay 50 \
    --save_epoch_freq 20 \
    --save_latest_freq 2000 \
    --num_threads 8 \
    --display_id -1

srun python test.py \
    --dataroot ./datasets/$NAME \
    --name pix2pix_${NAME}_1024 \
    --model pix2pix \
    --preprocess resize_and_crop \
    --load_size 1024 \
    --crop_size 1024
