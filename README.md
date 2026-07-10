# ChemSearch

![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![uv](https://img.shields.io/badge/env-uv-4B32C3)
![Platforms](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey)

ChemSearch is an interactive system for search and question answering over local chemistry paper libraries.

Put PDFs in a folder, build a local index, search the collection, and ask cited questions about a selected paper.

![ChemSearch search interface](docs/assets/chemsearch-search-results.png)

## Requirements

- `uv`
- An OpenAI-compatible API key
- CUDA or Apple MPS is optional. CPU works, but indexing is slower.

The installer uses a system Node.js 20+ if one exists. Otherwise it downloads a local Node.js into `.local/node/`.

## Quick Start

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/xukefaker/ChemSearch.git
cd ChemSearch
./scripts/install.sh
```

Edit `.env`:

```env
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
CHEMSEARCH_DEVICE=auto
```

Run a small chemistry demo:

```bash
./chemsearch demo-chem --max-papers 5
./chemsearch index
./chemsearch web
```

Open `http://127.0.0.1:4000`.

<details>
<summary>Windows PowerShell</summary>

Install `uv`, clone the repo, then run the installer:

```powershell
winget install --id=astral-sh.uv -e
git clone https://github.com/xukefaker/ChemSearch.git
cd ChemSearch
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\install.ps1
```

Edit `.env`:

```powershell
notepad .env
```

Run the demo:

```powershell
.\chemsearch.cmd demo-chem --max-papers 5
.\chemsearch.cmd index
.\chemsearch.cmd web
```

Open `http://127.0.0.1:4000`.

</details>

## Use Your PDFs

```bash
mkdir -p pdfs
# Put PDFs in ./pdfs

./chemsearch add-pdfs ./pdfs
./chemsearch index
./chemsearch web
```

During indexing, press `q` to cancel. ChemSearch removes staged files from that run and keeps the previous working index.

## Configuration

The installer creates `.venv/`, installs ChemSearch with an automatically selected PyTorch backend, creates `.env`, and runs `chemsearch doctor`.

The only required setting is:

```env
OPENAI_API_KEY=sk-...
```

Useful defaults:

```env
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
CHEMSEARCH_DATA_DIR=./data
CHEMSEARCH_DEVICE=auto
CHEMSEARCH_APP_NAME=ChemSearch
```

`CHEMSEARCH_DEVICE=auto` prefers CUDA or Apple MPS when PyTorch can use it. If no accelerator is available, ChemSearch warns and continues on CPU.

## Troubleshooting

```bash
./chemsearch doctor
```

- `CUDA available=False`: CPU still works, but indexing is slower. If you expected an NVIDIA GPU, reinstall after checking your driver.
- `OPENAI_API_KEY=missing`: edit `.env` and set your key.
- PowerShell blocks scripts: use the installer command shown in the Windows section. Its bypass applies only to that command.
- First `web` run is slow: frontend dependencies are installed under `apps/web/node_modules/`.
