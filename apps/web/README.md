# ChemVerify Web

This is the Next.js frontend for ChemVerify. It is normally started from the repository root:

```bash
uv run chemverify web
```

For frontend-only development:

```bash
npm install
CHEMVERIFY_API_BASE_URL=http://127.0.0.1:4001/api npm run dev -- --hostname 127.0.0.1 --port 4000
```

The frontend proxies API routes to the FastAPI backend through `CHEMVERIFY_API_BASE_URL`.
