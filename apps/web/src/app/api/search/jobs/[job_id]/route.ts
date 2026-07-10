import { NextResponse } from 'next/server';
import { getSearchJob, searchJobStatus } from '@/lib/workbench-store';

export async function GET(_request: Request, context: { params: Promise<{ job_id: string }> }) {
  const { job_id: jobId } = await context.params;
  const job = getSearchJob(jobId);
  if (!job) {
    return NextResponse.json({ detail: 'Search job not found.' }, { status: 404 });
  }
  return NextResponse.json(searchJobStatus(job));
}
