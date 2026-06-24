#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INFRA_DIR="$ROOT_DIR/infra"
ENV_FILE="$ROOT_DIR/.env"

if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

echo "Destroying removes infrastructure created by Terraform, including the EC2 instance and its Terraform-managed root volume."
echo "Save important checkpoints and logs to S3 or your local machine before continuing."
printf 'Type "destroy" to continue: '
read -r CONFIRMATION

if [ "$CONFIRMATION" != "destroy" ]; then
  echo "Destroy cancelled."
  exit 0
fi

export TF_VAR_aws_region="${AWS_REGION:?Set AWS_REGION in .env}"
export TF_VAR_instance_type="${INSTANCE_TYPE:?Set INSTANCE_TYPE in .env}"
export TF_VAR_key_name="${KEY_NAME:?Set KEY_NAME in .env}"
export TF_VAR_ssh_private_key_path="${SSH_PRIVATE_KEY_PATH:?Set SSH_PRIVATE_KEY_PATH in .env}"
export TF_VAR_allowed_ssh_cidr="${ALLOWED_SSH_CIDR:-0.0.0.0/0}"
export TF_VAR_repo_url="${REPO_URL:?Set REPO_URL in .env}"
export TF_VAR_repo_dir="${REPO_DIR:?Set REPO_DIR in .env}"
export TF_VAR_train_command="${TRAIN_COMMAND:?Set TRAIN_COMMAND in .env}"
export TF_VAR_root_volume_size="${ROOT_VOLUME_SIZE:?Set ROOT_VOLUME_SIZE in .env}"
export TF_VAR_auto_stop_after_training="${AUTO_STOP_AFTER_TRAINING:-true}"

terraform -chdir="$INFRA_DIR" destroy
