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

node_ok() {
  command -v node >/dev/null 2>&1 \
    && node -e 'process.exit(Number(process.versions.node.split(".")[0]) >= 20 ? 0 : 1)' >/dev/null 2>&1
}

install_local_node() {
  local version="${CHEMVERIFY_NODE_VERSION:-22.13.1}"
  local os
  local arch
  case "$(uname -s)" in
    Linux) os="linux" ;;
    Darwin) os="darwin" ;;
    *)
      printf '\033[31mUnsupported OS for automatic Node.js install: %s\033[0m\n' "$(uname -s)" >&2
      exit 1
      ;;
  esac
  case "$(uname -m)" in
    x86_64|amd64) arch="x64" ;;
    arm64|aarch64) arch="arm64" ;;
    *)
      printf '\033[31mUnsupported CPU for automatic Node.js install: %s\033[0m\n' "$(uname -m)" >&2
      exit 1
      ;;
  esac

  local node_root="$ROOT/.local/node"
  local current="$node_root/current"
  local node_bin="$current/bin/node"
  if [[ -x "$node_bin" ]] && "$node_bin" -e 'process.exit(Number(process.versions.node.split(".")[0]) >= 20 ? 0 : 1)' >/dev/null 2>&1; then
    export PATH="$current/bin:$PATH"
    printf 'Using local Node.js %s\n' "$("$node_bin" -v)"
    return
  fi

  local name="node-v${version}-${os}-${arch}"
  local archive="${name}.tar.xz"
  local url="https://nodejs.org/dist/v${version}/${archive}"
  mkdir -p "$node_root"
  printf 'Node.js >=20 not found. Installing local Node.js %s...\n' "$version"
  curl -L "$url" -o "$node_root/$archive"
  rm -rf "$node_root/$name" "$current"
  tar -xJf "$node_root/$archive" -C "$node_root"
  mv "$node_root/$name" "$current"
  rm -f "$node_root/$archive"
  export PATH="$current/bin:$PATH"
}

if node_ok; then
  printf 'Using system Node.js %s\n' "$(node -v)"
else
  install_local_node
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
