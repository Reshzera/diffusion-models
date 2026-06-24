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

expand_path() {
  case "$1" in
    "~"/*) printf '%s/%s\n' "$HOME" "${1#~/}" ;;
    *) printf '%s\n' "$1" ;;
  esac
}

PUBLIC_IP="$(terraform -chdir="$INFRA_DIR" output -raw public_ip)"
KEY_PATH="$(expand_path "${SSH_PRIVATE_KEY_PATH:?Set SSH_PRIVATE_KEY_PATH in .env}")"

exec ssh -i "$KEY_PATH" ubuntu@"$PUBLIC_IP" "$@"
