import { readFileSync } from 'node:fs';
import { NextResponse } from 'next/server';
import { paperPdfPath } from '@/lib/workbench-store';

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ paper_id: string }> },
) {
  const { paper_id } = await params;
  const pdfPath = paperPdfPath(paper_id);
  if (!pdfPath) {
    return NextResponse.json({ detail: 'PDF file not found for this paper.' }, { status: 404 });
  }
  return new NextResponse(readFileSync(pdfPath), {
    headers: {
      'content-type': 'application/pdf',
      'cache-control': 'no-store',
    },
  });
}
