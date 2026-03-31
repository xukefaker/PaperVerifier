import { readFile } from "node:fs/promises";
import path from "node:path";

const PDF_WORKER_PATH = path.join(process.cwd(), "node_modules", "pdfjs-dist", "legacy", "build", "pdf.worker.mjs");

export async function GET() {
  try {
    const payload = await readFile(PDF_WORKER_PATH, "utf-8");
    return new Response(payload, {
      status: 200,
      headers: {
        "Content-Type": "text/javascript; charset=utf-8",
        "Cache-Control": "public, max-age=3600",
      },
    });
  } catch (error) {
    return Response.json(
      {
        detail: error instanceof Error ? error.message : "Failed to load pdfjs worker.",
      },
      { status: 500 },
    );
  }
}
