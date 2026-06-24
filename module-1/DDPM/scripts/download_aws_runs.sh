#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INFRA_DIR="$ROOT_DIR/infra"
ENV_FILE="$ROOT_DIR/.env"
REMOTE_RUNS_DIR="${1:-runs/cuda-ddpm}"
LOCAL_OUTPUT_DIR="${2:-$ROOT_DIR/downloads/aws-runs}"

usage() {
  cat <<'USAGE'
Usage: ./scripts/download_aws_runs.sh [remote-runs-dir] [local-output-dir]

Downloads generated image files from the EC2 training run directory.

Examples:
  ./scripts/download_aws_runs.sh
  ./scripts/download_aws_runs.sh runs/ddpm
  ./scripts/download_aws_runs.sh runs/cuda-ddpm downloads/aws-ddpm
USAGE
}

expand_path() {
  case "$1" in
    "~"/*) printf '%s/%s\n' "$HOME" "${1#~/}" ;;
    *) printf '%s\n' "$1" ;;
  esac
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

if [ ! -f "$ENV_FILE" ]; then
  echo "Missing $ENV_FILE. Create it first: cp .env.example .env" >&2
  exit 1
fi

command -v terraform >/dev/null 2>&1 || { echo "Missing required command: terraform" >&2; exit 1; }
command -v ssh >/dev/null 2>&1 || { echo "Missing required command: ssh" >&2; exit 1; }
command -v scp >/dev/null 2>&1 || { echo "Missing required command: scp" >&2; exit 1; }

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

PUBLIC_IP="$(terraform -chdir="$INFRA_DIR" output -raw public_ip)"
KEY_PATH="$(expand_path "${SSH_PRIVATE_KEY_PATH:?Set SSH_PRIVATE_KEY_PATH in .env}")"
REPO_DIR="${REPO_DIR:?Set REPO_DIR in .env}"
REMOTE_PATH="$REPO_DIR/$REMOTE_RUNS_DIR"
LOCAL_PATH="$LOCAL_OUTPUT_DIR/$(basename "$REMOTE_RUNS_DIR")"

mkdir -p "$LOCAL_OUTPUT_DIR"

echo "Checking remote path: ubuntu@$PUBLIC_IP:$REMOTE_PATH"
ssh -i "$KEY_PATH" ubuntu@"$PUBLIC_IP" test -d "$REMOTE_PATH"

echo "Downloading PNG images to: $LOCAL_PATH"
mkdir -p "$LOCAL_PATH"
scp -i "$KEY_PATH" ubuntu@"$PUBLIC_IP":"$REMOTE_PATH"/*.png "$LOCAL_PATH/"

echo "Downloaded images from $REMOTE_RUNS_DIR to $LOCAL_PATH"
