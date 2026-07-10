import { NextRequest, NextResponse } from 'next/server';
import { createSearchJob, searchJobStatus, type QaModel, type RetrievalMethod } from '@/lib/workbench-store';

export async function POST(request: NextRequest) {
  const payload = (await request.json().catch(() => ({}))) as {
    query?: string;
    retrieval_method?: RetrievalMethod;
    qa_model?: QaModel;
    corpus_scope?: string;
  };
  if (!payload.query?.trim()) {
    return NextResponse.json({ detail: 'Query is required.' }, { status: 400 });
  }
  const job = createSearchJob({
    query: payload.query.trim(),
    retrieval_method: payload.retrieval_method,
    qa_model: payload.qa_model,
    corpus_scope: payload.corpus_scope,
  });
  return NextResponse.json(searchJobStatus(job));
}
