# ChemVerify

![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![uv](https://img.shields.io/badge/env-uv-4B32C3)
![Platforms](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey)

ChemVerify is a local evidence-verified chemistry paper search system for PDF libraries.

Put PDFs in a folder, build a local index, then search and read papers through a web interface with cited evidence.

## Requirements

- `uv`
- Node.js 20 or newer
- An OpenAI-compatible API key
- CUDA or Apple MPS is optional. CPU works, but indexing is slower.

## Quick Start

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/xukefaker/ChemVerify.git
cd ChemVerify
./scripts/install.sh
```

Edit `.env`:

```env
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
CHEMVERIFY_DEVICE=auto
```

Run a small chemistry demo:

```bash
./chemverify demo-chem --max-papers 5
./chemverify index
./chemverify web
```

Open `http://127.0.0.1:4000`.

<details>
<summary>Windows PowerShell</summary>

Install `uv`, clone the repo, then run the installer:

```powershell
winget install --id=astral-sh.uv -e
git clone https://github.com/xukefaker/ChemVerify.git
cd ChemVerify
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\install.ps1
```

Edit `.env`:

```powershell
notepad .env
```

Run the demo:

```powershell
.\chemverify.cmd demo-chem --max-papers 5
.\chemverify.cmd index
.\chemverify.cmd web
```

Open `http://127.0.0.1:4000`.

</details>

## Use Your PDFs

```bash
mkdir -p pdfs
# Put PDFs in ./pdfs

./chemverify add-pdfs ./pdfs
./chemverify index
./chemverify web
```

During indexing, press `q` to cancel. ChemVerify removes staged files from that run and keeps the previous working index.

## Configuration

The installer creates `.venv/`, installs ChemVerify with an automatically selected PyTorch backend, creates `.env`, and runs `chemverify doctor`.

The only required setting is:

```env
OPENAI_API_KEY=sk-...
```

Useful defaults:

```env
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
CHEMVERIFY_DATA_DIR=./data
CHEMVERIFY_DEVICE=auto
CHEMVERIFY_APP_NAME=ChemVerify
```

`CHEMVERIFY_DEVICE=auto` prefers CUDA or Apple MPS when PyTorch can use it. If no accelerator is available, ChemVerify warns and continues on CPU.

## Troubleshooting

```bash
./chemverify doctor
```

- `CUDA available=False`: CPU still works, but indexing is slower. If you expected an NVIDIA GPU, reinstall after checking your driver.
- `OPENAI_API_KEY=missing`: edit `.env` and set your key.
- PowerShell blocks scripts: use the installer command shown in the Windows section. Its bypass applies only to that command.
- First `web` run is slow: frontend dependencies are installed under `apps/web/node_modules/`.
