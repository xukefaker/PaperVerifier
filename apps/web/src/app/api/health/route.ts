import { NextResponse } from 'next/server';
import { getSettings, listPapers } from '@/lib/workbench-store';

export function GET() {
  const papers = listPapers();
  return NextResponse.json({
    status: 'ready',
    mode: 'local-workbench',
    counts: {
      papers: papers.length,
      ready: papers.filter((paper) => paper.status === 'ready').length,
      indexing: papers.filter((paper) => paper.status === 'indexing').length,
      failed: papers.filter((paper) => paper.status === 'failed').length,
    },
    settings: getSettings(),
  });
}
