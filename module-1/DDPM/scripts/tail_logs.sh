#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INFRA_DIR="$ROOT_DIR/infra"
ENV_FILE="$ROOT_DIR/.env"

if [ ! -f "$ENV_FILE" ]; then
  echo "Missing $ENV_FILE. Create it first: cp .env.example .env" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

expand_path() {
  case "$1" in
    "~"/*) printf '%s/%s\n' "$HOME" "${1#~/}" ;;
    *) printf '%s\n' "$1" ;;
  esac
}

PUBLIC_IP="$(terraform -chdir="$INFRA_DIR" output -raw public_ip)"
KEY_PATH="$(expand_path "${SSH_PRIVATE_KEY_PATH:?Set SSH_PRIVATE_KEY_PATH in .env}")"
REPO_DIR="${REPO_DIR:?Set REPO_DIR in .env}"

exec ssh -i "$KEY_PATH" ubuntu@"$PUBLIC_IP" bash -s -- "$REPO_DIR" <<'REMOTE'
set -euo pipefail
cd "$1"
touch train.log
tail -f train.log
REMOTE
