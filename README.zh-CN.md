# PaperSearchAgent

[English](README.md) | 简体中文

用你自己的 PDF 做本地论文检索。加入 PDF，构建索引，打开网页，然后直接提问。

## 快速开始

环境要求：Python 3.11/3.12、Node.js 20+、OpenAI 兼容 API key。

```bash
git clone -b public-release --single-branch https://github.com/xukefaker/PaperSearchAgent.git
cd PaperSearchAgent

python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .

paper-search-agent init

# 编辑 .env:
# OPENAI_API_KEY=sk-...
# OPENAI_MODEL=你的模型名

mkdir -p pdfs
# 把你的 PDF 放进 ./pdfs

paper-search-agent add-pdfs ./pdfs
paper-search-agent index
paper-search-agent web
```

然后打开 `http://127.0.0.1:4000`。

## 体验示例论文

暂时没有自己的 PDF？可以下载 100 篇 ACL 2025 long track 论文：

```bash
paper-search-agent demo-acl --max-papers 100
paper-search-agent index
paper-search-agent web
```

## 说明

- PDF 和索引默认保存在 `data/`。
- 问题会发送到你配置的 OpenAI 兼容 API。
- 小规模论文库可以用 CPU；大量 PDF 建议用 CUDA。
- 第一次运行 `paper-search-agent web` 时，如果需要，会自动在 `apps/web` 安装前端依赖。
