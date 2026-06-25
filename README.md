# PaperSearchAgent

English | [简体中文](README.zh-CN.md)

Local paper search from your own PDFs. Add PDFs, build an index, open the web UI, ask questions.

## Quick Start

Requirements: Python 3.11/3.12, Node.js 20+, and an OpenAI-compatible API key.

```bash
git clone -b public-release --single-branch https://github.com/xukefaker/PaperSearchAgent.git
cd PaperSearchAgent

python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .

paper-search-agent init

# edit .env:
# OPENAI_API_KEY=sk-...
# OPENAI_MODEL=your-model-name

mkdir -p pdfs
# put your PDFs in ./pdfs

paper-search-agent add-pdfs ./pdfs
paper-search-agent index
paper-search-agent web
```

Open `http://127.0.0.1:4000`.

## Try Demo Papers

No PDFs yet? Download 100 ACL 2025 long papers:

```bash
paper-search-agent demo-acl --max-papers 100
paper-search-agent index
paper-search-agent web
```

## Notes

- PDFs and indexes stay under `data/` by default.
- Questions are sent to your configured OpenAI-compatible API.
- CPU works for small collections; CUDA is recommended for larger PDF sets.
- The first `paper-search-agent web` run installs frontend dependencies in `apps/web` if needed.
