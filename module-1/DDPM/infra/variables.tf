variable "aws_region" {
  description = "AWS region where the GPU instance will be created."
  type        = string
  default     = "us-east-1"
}

variable "instance_type" {
  description = "EC2 instance type. Use a P5 family type for H100 GPUs."
  type        = string
  default     = "p5.4xlarge"
}

variable "key_name" {
  description = "Existing AWS EC2 key pair name."
  type        = string
}

variable "ssh_private_key_path" {
  description = "Local path to the private key matching key_name. Used only in outputs."
  type        = string
}

variable "allowed_ssh_cidr" {
  description = "CIDR allowed to SSH to the instance. Prefer your current public IP with /32."
  type        = string
}

variable "repo_url" {
  description = "GitHub repository URL to clone on the instance."
  type        = string
}

variable "repo_dir" {
  description = "Directory name to use when cloning the repository on the instance."
  type        = string
  default     = "ddpm-training"
}

variable "train_command" {
  description = "Training command executed inside the cloned repository."
  type        = string
  default     = "uv run cuda-ddpm train --dataset oxford-flowers --data-dir data/oxford-flowers --image-size 64 --channels 3 --base-channels 64 --batch-size 16 --timesteps 500 --save-every 1000 --steps 10000 --device cuda"
}

variable "root_volume_size" {
  description = "Root EBS volume size in GB."
  type        = number
  default     = 200
}

variable "auto_stop_after_training" {
  description = "Default auto-stop behavior used by the helper scripts."
  type        = bool
  default     = true
}
