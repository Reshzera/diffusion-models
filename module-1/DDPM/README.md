# DDPM

Educational Denoising Diffusion Probabilistic Model implementations using `uv`:

- `src/mlx_ddpm`: [MLX](https://ml-explore.github.io/mlx/) implementation for Apple silicon.
- `src/cuda_ddpm`: PyTorch/CUDA implementation for NVIDIA GPUs such as AWS H100/P5.

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

CUDA/H100 equivalent:

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

For CUDA/H100, use the same arguments with `cuda-ddpm` and add `--device cuda`.

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

For CUDA/H100, use the same arguments with `cuda-ddpm` and add `--device cuda`.

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

## AWS H100 Training Automation

This repository includes Terraform and Bash automation for launching an AWS EC2 GPU instance, cloning this repo, installing dependencies with `uv`, running a configurable training command, streaming logs to your local terminal, and stopping or destroying the infrastructure safely.

The default instance type is `p5.4xlarge` because the workflow is intended for H100/P5 usage. Availability varies by region and AWS account. If AWS reports that `p5.4xlarge` is unavailable, change `INSTANCE_TYPE` in `.env` to a P5 instance type available in your region and account.

Use `cuda-ddpm` for AWS H100 training. The MLX command remains available for local Apple silicon experiments, but it will not use NVIDIA CUDA acceleration.

### Prerequisites

- AWS CLI installed and configured locally with `aws configure` or an equivalent profile/SSO setup.
- Terraform installed locally.
- An existing EC2 key pair in the selected AWS region.
- The matching private key file on your local machine.
- P5/H100 EC2 quota in the selected AWS region.
- A GitHub repository URL that the EC2 instance can clone.
- Awareness that H100/P5 instances are expensive and may require a quota increase.

Do not commit AWS credentials, GitHub tokens, private SSH keys, Terraform state, or `.env` files. The included `.gitignore` excludes local config and Terraform state.

### Configure

```bash
cp .env.example .env
```

Edit `.env`:

```bash
AWS_REGION=us-east-1
INSTANCE_TYPE=p5.4xlarge
KEY_NAME=my-aws-key
SSH_PRIVATE_KEY_PATH=~/.ssh/my-aws-key.pem
ALLOWED_SSH_CIDR=
REPO_URL=https://github.com/USER/REPO.git
REPO_DIR=ddpm-training
TRAIN_COMMAND="uv run cuda-ddpm train --dataset oxford-flowers --data-dir data/oxford-flowers --image-size 64 --channels 3 --base-channels 64 --batch-size 16 --timesteps 500 --save-every 1000 --steps 10000 --device cuda"
ROOT_VOLUME_SIZE=200
AUTO_STOP_AFTER_TRAINING=true
```

Leave `ALLOWED_SSH_CIDR` empty to let `scripts/setup_aws.sh` detect your current public IP and use `<your-ip>/32` for SSH. Set it manually if your network blocks public IP detection or if you need a specific CIDR.

For this repository, a H100 command might look like this after you adapt paths and arguments:

```bash
TRAIN_COMMAND="uv run cuda-ddpm train --dataset oxford-flowers --data-dir data/oxford-flowers --image-size 64 --channels 3 --base-channels 64 --batch-size 16 --timesteps 500 --save-every 1000 --steps 10000 --device cuda"
```

### Recommended Workflow

```bash
./scripts/setup_aws.sh
./scripts/train_aws.sh --auto-stop
```

Optional commands:

```bash
./scripts/ssh_aws.sh
./scripts/tail_logs.sh
./scripts/download_aws_runs.sh
./scripts/stop_aws.sh
./scripts/destroy_aws.sh
```

Use `--auto-stop` by default to reduce the chance of accidentally leaving an expensive GPU instance running. When completely finished, run `./scripts/destroy_aws.sh` to remove the Terraform-managed infrastructure.

### What The Scripts Do

`scripts/setup_aws.sh` checks for AWS CLI and Terraform, verifies AWS authentication with `aws sts get-caller-identity`, detects your current public IP when `ALLOWED_SSH_CIDR` is empty, runs `terraform init`, runs `terraform apply`, and prints the instance IP plus SSH command.

`scripts/train_aws.sh` reads Terraform outputs and `.env`, connects by SSH, installs `git` and `uv` if needed, clones `REPO_URL` into `REPO_DIR`, runs `uv sync`, executes `TRAIN_COMMAND`, streams output to your local terminal, and saves remote logs to `train.log`.

`scripts/train_aws.sh --auto-stop` stops the EC2 instance locally with `aws ec2 stop-instances` after training exits, even if the training command fails. It also installs a cleanup trap that attempts to stop the instance if the local script is interrupted.

`scripts/train_aws.sh --no-auto-stop` leaves the instance running after training exits. Use this only when you intentionally need the instance to remain available.

`scripts/train_aws.sh --use-tmux` runs the remote training command inside a `tmux` session named `ddpm_train` and attaches your terminal to it. Detach with `Ctrl-b` then `d`. If you want training to survive an SSH drop, prefer `./scripts/train_aws.sh --use-tmux --no-auto-stop`; otherwise the local auto-stop cleanup may stop the instance when the local SSH process exits.

`scripts/tail_logs.sh` connects to the instance and runs `tail -f train.log` inside `REPO_DIR`, so you can follow logs without restarting training.

`scripts/download_aws_runs.sh` downloads generated `.png` images from a remote run directory on the instance. It defaults to `runs/cuda-ddpm`; pass another directory like `runs/ddpm` if you trained with a different output path.

`scripts/ssh_aws.sh` opens an interactive SSH session using Terraform outputs.

`scripts/stop_aws.sh` stops the instance by ID. Stopping interrupts GPU/CPU instance billing, but EBS volumes, S3 objects, snapshots, and Elastic IPs may continue generating costs.

`scripts/destroy_aws.sh` asks you to type `destroy`, then runs `terraform destroy`. Save important checkpoints and logs to S3 or your local machine before destroying.

### Terraform Resources

The Terraform configuration under `infra/` creates:

- AWS provider configuration for `aws_region`.
- Default VPC lookup and default subnet selection.
- A security group that allows SSH only from `allowed_ssh_cidr`.
- An EC2 instance using the latest matching AWS Deep Learning GPU AMI for Ubuntu 22.04.
- A configurable root EBS volume, 200 GB by default, with `delete_on_termination = true`.
- Tags including `Project`, `ManagedBy`, and `Purpose`.

Terraform outputs:

- `instance_id`
- `public_ip`
- `public_dns`
- `ssh_command`

### Stop, Destroy, And Terminate

`stop` stops the EC2 instance but keeps resources such as the EBS root volume. Use this when you may resume work later. GPU/CPU instance billing stops, but storage and other resources may still cost money.

`destroy` removes infrastructure created by Terraform. Use this when you are completely finished. The root volume is configured to be deleted on termination, but save important data elsewhere before destroying.

`terminate` directly terminates an instance through AWS. Avoid manual termination unless you know what you are doing, because Terraform state can become stale and manually managed resources may be left behind.

### Private GitHub Repositories

For a public repository, no GitHub credentials are required on the EC2 instance.

For a private repository, do not hardcode tokens in the repo. Use one of these secure approaches:

- Add a read-only deploy key to the GitHub repository and configure the EC2 instance to use it.
- Use SSH agent forwarding with care and only from trusted instances.
- Pass a short-lived GitHub token through your local environment or a secrets manager, then avoid writing it to shell history, logs, or Terraform state.

### Troubleshooting

Insufficient quota: request a P5 quota increase in the AWS Service Quotas console for the target region. H100/P5 capacity is limited and quota approval can take time.

Region without `p5.4xlarge`: set `INSTANCE_TYPE` to a P5 type available in your selected region, or change `AWS_REGION`. Run `aws ec2 describe-instance-type-offerings --location-type availability-zone --filters Name=instance-type,Values=p5* --region <region>` to inspect regional offerings.

SSH error: confirm `KEY_NAME` exists in the same region, `SSH_PRIVATE_KEY_PATH` points to the matching private key, and `ALLOWED_SSH_CIDR` includes your current public IP. If your IP changed, rerun `./scripts/setup_aws.sh` with an updated CIDR.

Incorrect SSH key permissions: run `chmod 600 ~/.ssh/my-aws-key.pem` locally, replacing the path with your key path.

`uv` not found: reconnect and check `~/.local/bin/uv`. The scripts install uv with the official installer and prepend `~/.local/bin` to `PATH` during training.

CUDA/PyTorch does not detect the GPU: SSH into the instance and run `nvidia-smi`. Confirm the AMI booted correctly, the selected instance has an NVIDIA GPU, and your Python dependencies install CUDA-enabled PyTorch or another NVIDIA-compatible stack.

Training stopped unexpectedly: check `train.log` with `./scripts/tail_logs.sh` or SSH into the instance and inspect the repository directory. If `--auto-stop` was enabled and the local script was interrupted, the cleanup trap may have stopped the instance intentionally.
