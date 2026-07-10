import { NextResponse } from 'next/server';
import { listPapers } from '@/lib/workbench-store';

export function GET() {
  return NextResponse.json({ papers: listPapers() });
}
