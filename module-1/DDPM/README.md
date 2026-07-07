# DDPM

Educational Denoising Diffusion Probabilistic Model implementations using `uv`:

- `src/mlx_ddpm`: [MLX](https://ml-explore.github.io/mlx/) implementation for Apple silicon.
- `src/cuda_ddpm`: PyTorch/CUDA implementation for NVIDIA GPUs.

Reference paper: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239), Ho, Jain, Abbeel, 2020.

## What This Implements

- Forward process `q(x_t | x_0)` with a linear beta schedule.
- Noise-prediction objective `E[||epsilon - epsilon_theta(x_t, t)||^2]` from the simplified DDPM loss.
- Reverse denoising sampler using the learned mean parameterization.
- Small U-Net epsilon predictor with sinusoidal timestep embeddings.
- MLX uses NHWC image tensors because MLX convolutions use channels-last layout.
- CUDA/PyTorch uses NCHW image tensors because PyTorch convolutions use channels-first layout.

This is intentionally compact for Module 1 study work, not a production image generator.

## Setup

```bash
uv sync
```

The package supports Python 3.11+. The checked-in `.python-version` uses Python 3.13 for local study work, while the CUDA Docker image below uses Python 3.11 to match the RunPod CUDA guide.

## Smoke Test

```bash
uv run mlx-ddpm smoke
uv run cuda-ddpm smoke --device auto
```

Use `mlx-ddpm` on Apple silicon. Use `cuda-ddpm` on NVIDIA GPU machines.

## Train

Point `--data-dir` at a folder containing images. Images are center-cropped, resized, converted to `[-1, 1]`, and loaded recursively.

```bash
uv run mlx-ddpm train \
  --data-dir /path/to/images \
  --image-size 32 \
  --channels 3 \
  --batch-size 32 \
  --steps 10000 \
  --save-every 1000 \
  --output-dir runs/ddpm
```

CUDA equivalent:

```bash
uv run cuda-ddpm train \
  --data-dir /path/to/images \
  --image-size 32 \
  --channels 3 \
  --batch-size 32 \
  --steps 10000 \
  --save-every 1000 \
  --output-dir runs/cuda-ddpm \
  --device cuda
```

The MLX final checkpoint is saved to `runs/ddpm/ddpm.safetensors`. The CUDA final checkpoint is saved to `runs/cuda-ddpm/ddpm.pt`. Intermediate checkpoints are saved under `checkpoints/` when `--save-every` is non-zero. Intermediate samples are saved as `sample_*.png` when `--sample-every` is non-zero.

## CIFAR-10

The training CLI can also download CIFAR-10 automatically and convert the official Python batches to PNG images:

```bash
uv run mlx-ddpm train \
  --dataset cifar-10 \
  --image-size 32 \
  --channels 3 \
  --base-channels 64 \
  --batch-size 32 \
  --timesteps 500 \
  --steps 10000 \
  --save-every 1000 \
  --output-dir runs/cifar10-ddpm
```

For CUDA, use the same arguments with `cuda-ddpm` and add `--device cuda`.

By default this saves the archive and converted images under `data/cifar-10`. Override that with `--data-dir` if needed.

## Oxford Flowers 102

The training CLI can download and extract Oxford Flowers 102 automatically:

```bash
uv run mlx-ddpm train \
  --dataset oxford-flowers \
  --data-dir data/oxford-flowers \
  --image-size 64 \
  --channels 3 \
  --base-channels 64 \
  --batch-size 16 \
  --timesteps 500 \
  --save-every 1000 \
  --steps 10000
```

For CUDA, use the same arguments with `cuda-ddpm` and add `--device cuda`.

The dataset archive is saved as `data/oxford-flowers/102flowers.tgz` and images are extracted to `data/oxford-flowers/jpg`.

Generate from a saved checkpoint:

```bash
uv run mlx-ddpm sample \
  --checkpoint runs/ddpm/checkpoints/ddpm_step_001000.safetensors \
  --image-size 64 \
  --channels 3 \
  --timesteps 500 \
  --output runs/ddpm/test_samples.png
```

CUDA checkpoints use `.pt` files:

```bash
uv run cuda-ddpm sample \
  --checkpoint runs/cuda-ddpm/checkpoints/ddpm_step_001000.pt \
  --image-size 64 \
  --channels 3 \
  --timesteps 500 \
  --output runs/cuda-ddpm/test_samples.png \
  --device cuda
```

## Sample

Use the same model-shape arguments used during training.

```bash
uv run mlx-ddpm sample \
  --checkpoint runs/ddpm/ddpm.safetensors \
  --image-size 32 \
  --channels 3 \
  --timesteps 1000 \
  --output runs/ddpm/samples.png
```

CUDA equivalent:

```bash
uv run cuda-ddpm sample \
  --checkpoint runs/cuda-ddpm/ddpm.pt \
  --image-size 32 \
  --channels 3 \
  --timesteps 1000 \
  --output runs/cuda-ddpm/samples.png \
  --device cuda
```

## Useful Smaller Experiment

For quick iteration on a small dataset:

```bash
uv run mlx-ddpm train \
  --data-dir /path/to/images \
  --image-size 32 \
  --base-channels 32 \
  --timesteps 200 \
  --batch-size 16 \
  --steps 1000
```

For CUDA quick iteration, replace `mlx-ddpm` with `cuda-ddpm` and add `--device auto` or `--device cuda`.

## RunPod CUDA Training

Use the Docker image in this directory when training `cuda_ddpm` on RunPod. It uses a CUDA 12.4 RunPod base image for wider host-driver compatibility:

- Base image: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`.
- Python: 3.11 from the RunPod PyTorch image.
- PyTorch: CUDA 12.4 from the RunPod PyTorch image.
- Installer: `uv` for the project package and small runtime dependencies.
- Working directory: `/workspace`, which is the recommended RunPod persistent volume mount path.
- Default command: `sleep infinity`, so the pod stays alive while you connect and launch training manually.

The Dockerfile installs into the image's system Python instead of running `uv sync`. That keeps RunPod's preinstalled CUDA PyTorch visible and avoids creating a separate `.venv` that may install a different Torch build.

Build the image from the DDPM project directory:

```bash
cd module-1/DDPM
docker build -t cuda-ddpm-runpod:cu124 .
```

On Apple silicon, the Dockerfile still builds a `linux/amd64` image because RunPod NVIDIA GPU pods are `amd64`. CUDA images are large; if Docker Desktop fails with `no space left on device`, increase Docker Desktop's disk limit or free Docker storage before rebuilding.

The build includes an import check for `cuda_ddpm`. If `cuda-ddpm smoke` fails on RunPod with `ModuleNotFoundError: No module named 'cuda_ddpm'`, rebuild and push the image again so the latest Dockerfile copies `src/` and sets `PYTHONPATH=/app/src`.

If your RunPod host supports CUDA 12.8, you can change the Dockerfile base image to a CUDA 12.8 RunPod image. If the pod fails with `unsatisfied condition: cuda>=12.8`, keep the CUDA 12.4 image or choose a RunPod machine with a newer NVIDIA driver.

Validate CUDA locally on a NVIDIA Docker host:

```bash
docker run --rm --gpus all cuda-ddpm-runpod:cu124 nvidia-smi
docker run --rm --gpus all cuda-ddpm-runpod:cu124 \
  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count()); print(torch.cuda.get_device_name(0))"
docker run --rm --gpus all cuda-ddpm-runpod:cu124 cuda-ddpm smoke --device cuda
```

Tag and push the image to a registry that RunPod can pull from:

```bash
docker tag cuda-ddpm-runpod:cu124 docker.io/<dockerhub-user>/cuda-ddpm-runpod:cu124
docker push docker.io/<dockerhub-user>/cuda-ddpm-runpod:cu124
```

Create the RunPod pod:

1. Create a Secure Cloud Network Volume if you want checkpoints and datasets to survive pod termination.
2. Deploy a GPU pod from a custom template using `docker.io/<dockerhub-user>/cuda-ddpm-runpod:cu124` as the image.
3. Mount the network volume at `/workspace`.
4. Use enough container disk for the image, then keep datasets and training outputs under `/workspace`.
5. Connect with the RunPod Web Terminal after the pod reaches `Running`.

Inside the pod, verify the GPU and run a smoke test:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
cuda-ddpm smoke --device cuda
```

Train on Oxford Flowers 102 in the cloud. The dataset and checkpoints are written to the persistent volume:

```bash
cuda-ddpm train \
  --dataset oxford-flowers \
  --data-dir /workspace/data/oxford-flowers \
  --image-size 64 \
  --channels 3 \
  --base-channels 64 \
  --batch-size 16 \
  --timesteps 1000 \
  --steps 50000 \
  --save-every 1000 \
  --sample-every 1000 \
  --output-dir /workspace/runs/flowers-cuda-ddpm \
  --device cuda
```

Optional CIFAR-10 run:

```bash
cuda-ddpm train \
  --dataset cifar-10 \
  --data-dir /workspace/data/cifar-10 \
  --image-size 32 \
  --channels 3 \
  --base-channels 64 \
  --batch-size 32 \
  --timesteps 1000 \
  --steps 50000 \
  --save-every 1000 \
  --sample-every 1000 \
  --output-dir /workspace/runs/cifar10-cuda-ddpm \
  --device cuda
```

Generate samples from a saved cloud checkpoint:

```bash
cuda-ddpm sample \
  --checkpoint /workspace/runs/flowers-cuda-ddpm/checkpoints/ddpm_step_001000.pt \
  --image-size 64 \
  --channels 3 \
  --timesteps 500 \
  --output /workspace/runs/flowers-cuda-ddpm/samples.png \
  --device cuda
```

RunPod persistence notes:

- Prefer Secure Cloud with a Network Volume for longer training jobs.
- Keep `--data-dir` and `--output-dir` under `/workspace` so datasets, checkpoints, and samples persist.
- Community Cloud storage is less persistent; download checkpoints before terminating the pod or sync `/workspace/runs` to external storage.
- Rebuild and push the Docker image whenever you change code under `src/`.

### Download Weights Through The RunPod S3 API

Use RunPod's S3-compatible API when SSH shows **No support for SCP & SFTP**. This API exposes the Network Volume directly, so a pod file like:

```text
/workspace/runs/flowers-cuda-ddpm/ddpm.pt
```

maps to:

```text
s3://<network-volume-id>/runs/flowers-cuda-ddpm/ddpm.pt
```

Create the Network Volume in a supported datacenter, then note:

- Network Volume ID, for example `6agzcxkxwf`.
- Datacenter/region, for example `eu-ro-1`.
- S3 endpoint, for example `https://s3api-eu-ro-1.runpod.io/`.
- S3 API access key and secret from RunPod settings.

Configure credentials on your local computer:

```bash
aws configure --profile runpod
```

When prompted, use the RunPod S3 API key values. For **AWS Access Key ID**, use the RunPod user ID/access key shown for the S3 API key, usually starting with `user_`. For **AWS Secret Access Key**, use the S3 API key secret, usually starting with `rps_`. The default region and output format can be left blank.

Verify the Network Volume contents:

```bash
aws s3 ls \
  --profile runpod \
  --region eu-ro-1 \
  --endpoint-url https://s3api-eu-ro-1.runpod.io/ \
  s3://6agzcxkxwf/
```

### Upload Datasets Through The RunPod S3 API

Download a dataset locally, then upload it directly to the Network Volume with the AWS CLI. These examples use the same example Network Volume ID, region, and endpoint from above.

Download and upload Oxford Flowers 102:

```bash
mkdir -p data/oxford-flowers
curl -L \
  https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz \
  -o data/oxford-flowers/102flowers.tgz
tar -xzf data/oxford-flowers/102flowers.tgz -C data/oxford-flowers

aws s3 sync data/oxford-flowers/jpg/ s3://6agzcxkxwf/data/oxford-flowers/jpg/ \
  --profile runpod \
  --region eu-ro-1 \
  --endpoint-url https://s3api-eu-ro-1.runpod.io/
```

Download, convert, and upload CIFAR-10 PNG images:

```bash
mkdir -p data/cifar-10
curl -L \
  https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz \
  -o data/cifar-10/cifar-10-python.tar.gz
uv run cuda-ddpm train \
  --dataset cifar-10 \
  --data-dir data/cifar-10 \
  --steps 0 \
  --output-dir /tmp/cifar-10-prepare

aws s3 sync data/cifar-10/images/ s3://6agzcxkxwf/data/cifar-10/images/ \
  --profile runpod \
  --region eu-ro-1 \
  --endpoint-url https://s3api-eu-ro-1.runpod.io/
```

Verify the uploaded dataset images:

```bash
aws s3 ls s3://6agzcxkxwf/data/ \
  --recursive \
  --profile runpod \
  --region eu-ro-1 \
  --endpoint-url https://s3api-eu-ro-1.runpod.io/
```

Train from the uploaded image folders inside the pod:

```bash
cuda-ddpm train \
  --data-dir /workspace/data/oxford-flowers/jpg \
  --image-size 64 \
  --channels 3 \
  --base-channels 64 \
  --batch-size 16 \
  --timesteps 1000 \
  --steps 10000 \
  --save-every 1000 \
  --sample-every 1000 \
  --output-dir /workspace/runs/flowers-cuda-ddpm \
  --device cuda

cuda-ddpm train \
  --data-dir /workspace/data/cifar-10/images \
  --image-size 32 \
  --channels 3 \
  --base-channels 64 \
  --batch-size 32 \
  --timesteps 1000 \
  --steps 50000 \
  --save-every 1000 \
  --sample-every 1000 \
  --output-dir /workspace/runs/cifar10-cuda-ddpm \
  --device cuda
```

List the Flowers run directory:

```bash
aws s3 ls s3://6agzcxkxwf/runs/flowers-cuda-ddpm/ \
  --recursive \
  --profile runpod \
  --region eu-ro-1 \
  --endpoint-url https://s3api-eu-ro-1.runpod.io/
```

Download the final checkpoint to your computer:

```bash
aws s3 cp s3://6agzcxkxwf/runs/flowers-cuda-ddpm/ddpm.pt ./ddpm.pt \
  --profile runpod \
  --region eu-ro-1 \
  --endpoint-url https://s3api-eu-ro-1.runpod.io/
```

Download an intermediate checkpoint to your computer:

```bash
aws s3 cp s3://6agzcxkxwf/runs/flowers-cuda-ddpm/checkpoints/ddpm_step_001000.pt ./ddpm_step_001000.pt \
  --profile runpod \
  --region eu-ro-1 \
  --endpoint-url https://s3api-eu-ro-1.runpod.io/
```

Download the full run directory to your computer:

```bash
aws s3 sync s3://6agzcxkxwf/runs/flowers-cuda-ddpm ./flowers-cuda-ddpm \
  --profile runpod \
  --region eu-ro-1 \
  --endpoint-url https://s3api-eu-ro-1.runpod.io/
```

If `sync` has problems, use recursive `cp` instead:

```bash
aws s3 cp s3://6agzcxkxwf/runs/flowers-cuda-ddpm ./flowers-cuda-ddpm \
  --recursive \
  --profile runpod \
  --region eu-ro-1 \
  --endpoint-url https://s3api-eu-ro-1.runpod.io/
```

Upload a local file to the Network Volume if needed:

```bash
aws s3 cp ./local-file.txt s3://6agzcxkxwf/local-file.txt \
  --profile runpod \
  --region eu-ro-1 \
  --endpoint-url https://s3api-eu-ro-1.runpod.io/
```

If `aws s3 ls` returns no files under `s3://6agzcxkxwf/runs/flowers-cuda-ddpm/`, check inside the pod that training wrote files to the mounted Network Volume:

```bash
find /workspace/runs/flowers-cuda-ddpm -type f
```

If `/workspace/runs/flowers-cuda-ddpm` is empty, training did not save weights there yet. The S3 API cannot show files that do not exist on the Network Volume.

If `aws s3 ls` fails with `SignatureDoesNotMatch`, check that the access key, secret key, Network Volume ID, datacenter/region, endpoint URL, and selected `--profile` all match the same RunPod account and volume.

If `aws s3 cp ./ddpm.pt ...` fails with `The user-provided path ./ddpm.pt does not exist`, the source file is not in your current local directory. For downloads, put the `s3://...` path first and the local destination second.
