#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INFRA_DIR="$ROOT_DIR/infra"
ENV_FILE="$ROOT_DIR/.env"

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

load_env() {
  if [ ! -f "$ENV_FILE" ]; then
    echo "Missing $ENV_FILE. Create it first: cp .env.example .env" >&2
    exit 1
  fi

  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
}

detect_public_ip() {
  curl -fsS https://checkip.amazonaws.com 2>/dev/null | tr -d '[:space:]'
}

load_env
require_command aws
require_command terraform
require_command curl

aws sts get-caller-identity >/dev/null

if [ -z "${ALLOWED_SSH_CIDR:-}" ]; then
  PUBLIC_IP="$(detect_public_ip || true)"
  if [ -z "$PUBLIC_IP" ]; then
    echo "Could not detect your public IP. Set ALLOWED_SSH_CIDR in .env, for example 203.0.113.10/32." >&2
    exit 1
  fi
  export TF_VAR_allowed_ssh_cidr="$PUBLIC_IP/32"
  echo "Using detected SSH CIDR: $TF_VAR_allowed_ssh_cidr"
else
  export TF_VAR_allowed_ssh_cidr="$ALLOWED_SSH_CIDR"
fi

export TF_VAR_aws_region="${AWS_REGION:?Set AWS_REGION in .env}"
export TF_VAR_instance_type="${INSTANCE_TYPE:?Set INSTANCE_TYPE in .env}"
export TF_VAR_key_name="${KEY_NAME:?Set KEY_NAME in .env}"
export TF_VAR_ssh_private_key_path="${SSH_PRIVATE_KEY_PATH:?Set SSH_PRIVATE_KEY_PATH in .env}"
export TF_VAR_repo_url="${REPO_URL:?Set REPO_URL in .env}"
export TF_VAR_repo_dir="${REPO_DIR:?Set REPO_DIR in .env}"
export TF_VAR_train_command="${TRAIN_COMMAND:?Set TRAIN_COMMAND in .env}"
export TF_VAR_root_volume_size="${ROOT_VOLUME_SIZE:?Set ROOT_VOLUME_SIZE in .env}"
export TF_VAR_auto_stop_after_training="${AUTO_STOP_AFTER_TRAINING:-true}"

echo "Initializing Terraform in $INFRA_DIR"
terraform -chdir="$INFRA_DIR" init

echo "Applying Terraform. Review the plan carefully before approving."
terraform -chdir="$INFRA_DIR" apply

echo "Instance public IP: $(terraform -chdir="$INFRA_DIR" output -raw public_ip)"
echo "SSH command: $(terraform -chdir="$INFRA_DIR" output -raw ssh_command)"
