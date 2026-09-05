# High-Resolution pix2pix Slurm Runs

Two Slurm batch scripts for training pix2pix above the repository's default 256x256 resolution,
written to follow `slurm/STYLE.md`:

- `slurm/run_train_pix2pix_1024.sh`
- `slurm/run_train_pix2pix_512.sh`

Both take the dataset name as `$1` (defaulting to `facades`) and mirror `train.sh`'s train-then-test
flow with the same 50/50 epoch schedule and `--save_epoch_freq 20`.

```bash
sbatch slurm/run_train_pix2pix_1024.sh my_dataset
sbatch slurm/run_train_pix2pix_512.sh my_dataset
```

## Flags changed vs. the 256x256 defaults

| Flag | Default | 512 | 1024 | Why |
|---|---|---|---|---|
| `--load_size` / `--crop_size` | 286 / 256 | 572 / 512 | 1144 / 1024 | Keeps the paper's 1.117x random-jitter ratio rather than `load_size == crop_size`, so the crop augmentation is not lost |
| `--batch_size` | 1 | 1 | 1 | Already 1; 512 can probably take 2-4 |
| `--netD` / `--n_layers_D` | `basic` / 3 (70x70 RF) | `n_layers` / 4 (142x142) | `n_layers` / 5 (286x286) | Per `ARCHITECTURE.md` section 3 - a 70x70 patch is <7% of a 1024 image, which limits the model to local texture |
| `--num_threads` | 4 | 8 | 8 | Matches `--cpus-per-task=8`; decoding large images is CPU-bound |
| `--save_latest_freq` | 5000 | - | 2000 | Re-running lost iterations is expensive at this resolution |

Defaults confirmed in `options/base_options.py:32-49` and `options/train_options.py`.

## Decisions worth flagging

### Generator stays `unet_256`

Both scripts keep pix2pix's own default generator. It is fully convolutional, so 512 inputs simply
bottleneck at 2x2 and 1024 at 4x4 instead of 1x1.

`ARCHITECTURE.md`'s high-resolution section is written around `resnet_9blocks`, but that architecture
only downsamples twice - at 1024 it carries a 256x256 latent through 9 residual blocks and will OOM
well before the U-Net does. There is no `unet_512` registered in `models/networks.py:148`, so
`unet_256` is the practical choice at both resolutions.

### Resources

- **512**: the `slurm/STYLE.md` standard V100 plus 30G host RAM.
- **1024**: 60G host RAM and `--constraint=cascade,v100` to pin the 32GB V100 nodes, with a commented
  A100 alternative.

The account, email, virtualenv path, and project path are `slurm/STYLE.md`'s `def-example`
placeholders. Substitute real values before submitting.

## Caveat: aspect ratio

`resize_and_crop` resizes to a square `load_size x load_size`, distorting aspect ratio. This is the
same behavior as the 256 default (see `get_transform` in `data/base_dataset.py`), but it is far more
visible at high resolution. If the image pairs are not square, `--preprocess scale_width_and_crop` is
the better switch.
