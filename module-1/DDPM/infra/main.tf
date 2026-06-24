provider "aws" {
  region = var.aws_region
}

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

data "aws_ami" "deep_learning" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name = "name"
    values = [
      "Deep Learning AMI GPU PyTorch*Ubuntu 22.04*",
      "Deep Learning Base OSS Nvidia Driver GPU AMI*Ubuntu 22.04*"
    ]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

resource "aws_security_group" "ddpm_training" {
  name_prefix = "ddpm-training-ssh-"
  description = "SSH access for DDPM GPU training"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "SSH from configured client CIDR"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ssh_cidr]
  }

  egress {
    description = "Allow outbound internet access"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name      = "ddpm-training-ssh"
    Project   = "DDPM"
    ManagedBy = "Terraform"
    Purpose   = "H100 training SSH access"
  }
}

resource "aws_instance" "ddpm_training" {
  ami                         = data.aws_ami.deep_learning.id
  instance_type               = var.instance_type
  key_name                    = var.key_name
  subnet_id                   = data.aws_subnets.default.ids[0]
  vpc_security_group_ids      = [aws_security_group.ddpm_training.id]
  associate_public_ip_address = true

  root_block_device {
    volume_size           = var.root_volume_size
    volume_type           = "gp3"
    delete_on_termination = true
  }

  metadata_options {
    http_tokens = "required"
  }

  tags = {
    Name      = "ddpm-h100-training"
    Project   = "DDPM"
    ManagedBy = "Terraform"
    Purpose   = "H100 GPU DDPM training"
    RepoURL   = var.repo_url
    RepoDir   = var.repo_dir
    AutoStop  = tostring(var.auto_stop_after_training)
  }
}
