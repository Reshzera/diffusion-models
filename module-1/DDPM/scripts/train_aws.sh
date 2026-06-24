#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INFRA_DIR="$ROOT_DIR/infra"
ENV_FILE="$ROOT_DIR/.env"
AUTO_STOP=""
USE_TMUX=false
STOP_REQUESTED=false

usage() {
  cat <<'USAGE'
Usage: ./scripts/train_aws.sh [--auto-stop|--no-auto-stop] [--use-tmux]

Runs the configured training command on the EC2 instance and mirrors output locally.
USAGE
}

expand_path() {
  case "$1" in
    "~"/*) printf '%s/%s\n' "$HOME" "${1#~/}" ;;
    *) printf '%s\n' "$1" ;;
  esac
}

stop_instance() {
  if [ "$STOP_REQUESTED" = true ]; then
    return
  fi
  STOP_REQUESTED=true
  echo "Auto-stop active. Stopping EC2 instance $INSTANCE_ID to interrupt GPU/CPU billing."
  aws ec2 stop-instances --region "$AWS_REGION" --instance-ids "$INSTANCE_ID" >/dev/null || true
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --auto-stop) AUTO_STOP=true ;;
    --no-auto-stop) AUTO_STOP=false ;;
    --use-tmux) USE_TMUX=true ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown flag: $1" >&2; usage; exit 1 ;;
  esac
  shift
done

if [ ! -f "$ENV_FILE" ]; then
  echo "Missing $ENV_FILE. Create it first: cp .env.example .env" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

if [ -z "$AUTO_STOP" ]; then
  AUTO_STOP="${AUTO_STOP_AFTER_TRAINING:-true}"
fi

command -v aws >/dev/null 2>&1 || { echo "Missing required command: aws" >&2; exit 1; }
command -v terraform >/dev/null 2>&1 || { echo "Missing required command: terraform" >&2; exit 1; }

AWS_REGION="${AWS_REGION:?Set AWS_REGION in .env}"
REPO_URL="${REPO_URL:?Set REPO_URL in .env}"
REPO_DIR="${REPO_DIR:?Set REPO_DIR in .env}"
TRAIN_COMMAND="${TRAIN_COMMAND:?Set TRAIN_COMMAND in .env}"
KEY_PATH="$(expand_path "${SSH_PRIVATE_KEY_PATH:?Set SSH_PRIVATE_KEY_PATH in .env}")"
PUBLIC_IP="$(terraform -chdir="$INFRA_DIR" output -raw public_ip)"
INSTANCE_ID="$(terraform -chdir="$INFRA_DIR" output -raw instance_id)"

if [ "$AUTO_STOP" = true ]; then
  trap stop_instance EXIT INT TERM
fi

echo "Connecting to ubuntu@$PUBLIC_IP and running training. Logs will be saved to ~/$(basename "$REPO_DIR")/train.log."

ssh -i "$KEY_PATH" -o ServerAliveInterval=30 -o ServerAliveCountMax=10 ubuntu@"$PUBLIC_IP" \
  bash -s -- "$REPO_URL" "$REPO_DIR" "$TRAIN_COMMAND" "$USE_TMUX" <<'REMOTE'
set -euo pipefail

REPO_URL="$1"
REPO_DIR="$2"
TRAIN_COMMAND="$3"
USE_TMUX="$4"

install_git() {
  if command -v git >/dev/null 2>&1; then
    return
  fi
  sudo apt-get update
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y git
}

install_uv() {
  if command -v uv >/dev/null 2>&1; then
    return
  fi
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
}

install_tmux() {
  if command -v tmux >/dev/null 2>&1; then
    return
  fi
  sudo apt-get update
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y tmux
}

install_git
install_uv

if [ ! -d "$REPO_DIR/.git" ]; then
  git clone "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"
export PATH="$HOME/.local/bin:$PATH"
uv sync

if [ "$USE_TMUX" = true ]; then
  install_tmux
  SESSION="ddpm_train"
  cat > .aws_train_tmux.sh <<SCRIPT
#!/usr/bin/env bash
set -euo pipefail
export PATH="\$HOME/.local/bin:\$PATH"
$TRAIN_COMMAND 2>&1 | tee train.log
exit \${PIPESTATUS[0]}
SCRIPT
  chmod +x .aws_train_tmux.sh
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists. Attaching now."
  else
    tmux new-session -d -s "$SESSION" -c "$PWD" "./.aws_train_tmux.sh"
    echo "Started tmux session '$SESSION'. Detach with Ctrl-b then d."
  fi
  tmux attach-session -t "$SESSION"
else
  bash -lc "$TRAIN_COMMAND" 2>&1 | tee train.log
  exit ${PIPESTATUS[0]}
fi
REMOTE

if [ "$AUTO_STOP" = true ]; then
  stop_instance
  trap - EXIT INT TERM
fi
