import { NextResponse } from 'next/server';
import { demoRankedResults, getSearchJob, rankedResults, retrievalLabel, writeSearchResults } from '@/lib/workbench-store';

export async function GET(_request: Request, context: { params: Promise<{ job_id: string }> }) {
  const { job_id: jobId } = await context.params;
  const job = getSearchJob(jobId);
  if (!job) {
    return NextResponse.json({ detail: 'Search job not found.' }, { status: 404 });
  }
  let results = rankedResults(job);
  if (!results.length) {
    results = writeSearchResults(job, demoRankedResults(job));
  }
  return NextResponse.json({
    job_id: job.job_id,
    query: job.query,
    retrieval_method: job.retrieval_method,
    retrieval_label: retrievalLabel(job.retrieval_method),
    qa_model: job.qa_model,
    qa_model_label: job.qa_model,
    results,
  });
}
