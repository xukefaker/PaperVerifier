import { appendFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { NextRequest, NextResponse } from 'next/server';

import { appError, errorMessage } from '@/lib/api-error';
import { workbenchDataRoot } from '@/lib/workbench-store';

type FeedbackVote = 'up' | 'down';
type FeedbackReason = 'incorrect' | 'missing_evidence' | 'not_clear' | 'other';

const validReasons = new Set<FeedbackReason>(['incorrect', 'missing_evidence', 'not_clear', 'other']);

function cleanString(value: unknown, maxLength: number) {
  return typeof value === 'string' ? value.trim().slice(0, maxLength) : '';
}

export async function POST(request: NextRequest) {
  const payload = (await request.json().catch(() => ({}))) as {
    paper_id?: unknown;
    answer_id?: unknown;
    question?: unknown;
    vote?: unknown;
    reason?: unknown;
    note?: unknown;
  };

  const paperId = cleanString(payload.paper_id, 256);
  const answerId = cleanString(payload.answer_id, 256);
  const question = cleanString(payload.question, 4000);
  const vote = payload.vote === 'up' || payload.vote === 'down' ? (payload.vote as FeedbackVote) : null;
  const reason = typeof payload.reason === 'string' && validReasons.has(payload.reason as FeedbackReason)
    ? (payload.reason as FeedbackReason)
    : null;
  const note = cleanString(payload.note, 2000);

  if (!paperId || !answerId || !question || !vote) {
    return appError('bad_request', 'paper_id, answer_id, question, and vote are required.');
  }
  if (vote === 'down' && !reason) {
    return appError('bad_request', 'A feedback reason is required for thumbs down.');
  }

  const feedback = {
    paper_id: paperId,
    answer_id: answerId,
    question,
    vote,
    reason: vote === 'down' ? reason : null,
    note: vote === 'down' ? note : '',
    created_at: new Date().toISOString(),
    submitted_at: new Date().toISOString(),
  };

  try {
    const feedbackDir = join(workbenchDataRoot(), 'feedback');
    mkdirSync(feedbackDir, { recursive: true });
    appendFileSync(join(feedbackDir, 'paper_qa_feedback.jsonl'), `${JSON.stringify(feedback)}\n`);
    return NextResponse.json({ ok: true, feedback });
  } catch (error) {
    return appError('internal_error', errorMessage(error, 'Feedback could not be saved.'));
  }
}
