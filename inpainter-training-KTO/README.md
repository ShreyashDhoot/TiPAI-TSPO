# Inpainter Training KTO

Training pipeline for fine-tuning a Stable Diffusion inpainting UNet with a KTO-style preference loss. The project supports optional LoRA adapters, latent-space datasets stored as Parquet files, mixed-precision training, checkpointing, visual evaluation, and Weights & Biases logging.

## What This Repository Does

This codebase trains an inpainting model from a pretrained Stable Diffusion inpainting checkpoint and updates the UNet using a preference signal. The training loop:

- Loads latent-space training and validation data from a Hugging Face dataset repository or a local Parquet directory.
- Builds a Stable Diffusion inpainting pipeline from `runwayml/stable-diffusion-inpainting` by default.
- Optionally wraps the UNet with LoRA adapters.
- Compares the trainable UNet against a frozen reference UNet.
- Optimizes a KTO loss that uses sample labels to encourage better predictions on preferred examples.
- Logs metrics to Weights & Biases.
- Periodically saves checkpoints and visual evaluation outputs.

## Repository Layout

- `configs/inpaint.yaml`: Main training configuration.
- `data/dataset.py`: Parquet-backed latent inpainting dataset loader.
- `data/collate.py`: Batch collation helper.
- `engine/train_one_epoch.py`: Main training loop.
- `engine/evaluate.py`: Visual evaluation helpers.
- `engine/checkpoint.py`: Checkpoint save/load utilities.
- `losses/kto_loss.py`: Preference-based KTO loss.
- `models/unet_wrapper.py`: UNet forward wrapper for inpainting inputs.
- `models/diffusion_utils.py`: Diffusion noise helpers.
- `scripts/train.py`: Entry point for training.
- `utils/logging.py`: Weights & Biases setup and logging.
- `utils/seed.py`: Reproducibility helpers.

## Requirements

Install the Python dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

The project expects a CUDA-capable GPU for the default training script, since `scripts/train.py` sets the device to `cuda` and uses fp16 mixed precision.

## Dataset Format

The dataset loader expects Parquet files with at least these fields per row:

- `z0`: clean latent tensor
- `masked_latent`: masked latent tensor
- `mask_latent`: inpainting mask in latent space
- `input_ids`: tokenized prompt ids
- `label`: preference label, typically `1.0` for positive and `0.0` for negative

The loader can read from:

- a Hugging Face dataset repository via `repo_id`
- a local directory via `local_dir`

It also supports split subdirectories such as `train` and `val`.

## Configuration

Training is controlled by `configs/inpaint.yaml`.

Important sections:

- `model.base_model`: pretrained inpainting model checkpoint
- `model.hf_dataset_repo`: Hugging Face dataset repo to download
- `model.use_lora`: enable or disable LoRA adapters
- `data.train_subdir` and `data.val_subdir`: dataset split names
- `data.cache_dir`: local dataset cache directory
- `training.batch_size`, `training.lr`, `training.max_steps`, `training.grad_accum_steps`
- `training.beta`: KTO loss temperature
- `training.save_every`, `training.log_every`, `training.grad_clip_norm`
- `output.checkpoint_dir` and `output.eval_dir`: output locations
- `output.wandb_project` and related W&B settings

Example defaults are already provided in the config file.

## Training

Run training from the repository root:

```bash
python scripts/train.py
```

The script:

1. Loads `configs/inpaint.yaml`.
2. Seeds Python, NumPy, and PyTorch with `42`.
3. Downloads or opens the latent dataset.
4. Builds the Stable Diffusion inpainting pipeline.
5. Optionally applies LoRA to the UNet.
6. Trains until `training.max_steps` is reached.

### Notes

- The script assumes a CUDA environment.
- Weights & Biases is enabled by default in the sample config.
- The config path is hardcoded as `configs/inpaint.yaml`, so run the command from the repository root unless you update the script.

## Outputs

During training, the project writes artifacts to the directories configured in `configs/inpaint.yaml`.

Typical outputs:

- `checkpoints/step_<N>.pt`: serialized training state
- `checkpoints/lora_step_<N>/`: LoRA adapter weights when LoRA is enabled
- `checkpoint--<N>/`: adapter export written by `unet.save_pretrained(...)`
- `eval_outputs/`: saved inpainting previews from validation samples

## Checkpointing

The checkpoint helper saves the UNet state, optimizer, scheduler, scaler, global step, and epoch.

If LoRA is enabled, the adapter weights are saved separately and the checkpoint file stores metadata plus optimizer state.

The repository also includes a `load_checkpoint` helper in `engine/checkpoint.py`, but the training script does not currently wire resume logic into the main entry point.

## Visual Evaluation

Validation previews are generated periodically with the current pipeline and written to `eval_outputs/`.

If you extend the dataset format, make sure validation samples contain whatever prompt text the visual evaluation path expects. The current training script collects the first few validation samples for preview generation.

## Logging

Training metrics are logged through Weights & Biases using the project name configured in `output.wandb_project`.

Common logged values include:

- training loss
- reward gap between positive and negative samples
- gradient norm
- learning rate
- epoch number

## Troubleshooting

- If the dataset cannot be found, check the Hugging Face repo id, cache directory, and split names.
- If no Parquet files are found, verify the dataset directory layout and that each split contains `.parquet` files.
- If training fails on startup, confirm that CUDA is available and that the base model can be downloaded.
- If W&B logging is not desired, disable it in `configs/inpaint.yaml` under `output.use_wandb` or set `output.wandb_mode` to `disabled`.

## W&B Sync and Cleanup

If local runs are not visible in W&B yet, sync all local run folders:

```bash
python scripts/wandb_sync_and_clean.py --sync
```

Sync and then clean local W&B files before pushing to GitHub:

```bash
python scripts/wandb_sync_and_clean.py --sync --clean --yes
```

Preview all actions without making changes:

```bash
python scripts/wandb_sync_and_clean.py --sync --clean --dry-run
```

## License

Add a license here if one applies to this repository.