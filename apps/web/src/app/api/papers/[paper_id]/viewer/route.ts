import { NextResponse } from 'next/server';
import { paperViewer } from '@/lib/workbench-store';

export async function GET(_request: Request, context: { params: Promise<{ paper_id: string }> }) {
  const { paper_id: paperId } = await context.params;
  const viewer = paperViewer(paperId);
  if (!viewer) {
    return NextResponse.json({ detail: 'Paper not found.' }, { status: 404 });
  }
  return NextResponse.json(viewer);
}
