#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

printf '\033[36mChemVerify installer\033[0m\n'

if ! command -v uv >/dev/null 2>&1; then
  printf '\033[31muv is required. Install it first:\033[0m\n'
  printf 'curl -LsSf https://astral.sh/uv/install.sh | sh\n'
  exit 1
fi

uv python install 3.12
uv venv --python 3.12 --allow-existing .venv

# shellcheck disable=SC1091
source .venv/bin/activate

uv pip install -e . --torch-backend=auto
./chemverify init
./chemverify doctor

printf '\n\033[32mDone. Edit .env, then run:\033[0m\n'
printf './chemverify demo-chem --max-papers 5\n'
printf './chemverify index\n'
printf './chemverify web\n'
