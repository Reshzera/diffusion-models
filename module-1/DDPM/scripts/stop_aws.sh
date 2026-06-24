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

command -v aws >/dev/null 2>&1 || { echo "Missing required command: aws" >&2; exit 1; }
command -v terraform >/dev/null 2>&1 || { echo "Missing required command: terraform" >&2; exit 1; }

AWS_REGION="${AWS_REGION:?Set AWS_REGION in .env}"
INSTANCE_ID="$(terraform -chdir="$INFRA_DIR" output -raw instance_id)"

aws ec2 stop-instances --region "$AWS_REGION" --instance-ids "$INSTANCE_ID" >/dev/null
echo "Stop requested for $INSTANCE_ID. This interrupts GPU/CPU instance billing."
echo "EBS volumes, S3 objects, snapshots, and Elastic IPs may continue generating costs."
