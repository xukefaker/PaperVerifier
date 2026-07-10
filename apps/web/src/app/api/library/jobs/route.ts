import { NextResponse } from 'next/server';
import { listLibraryJobs } from '@/lib/workbench-store';

export function GET() {
  return NextResponse.json({ jobs: listLibraryJobs() });
}
