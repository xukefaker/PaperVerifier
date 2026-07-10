import { NextRequest, NextResponse } from 'next/server';
import { callChatCompletion } from '@/lib/openai-compatible';
import { effectiveSettings, type WorkbenchSettings } from '@/lib/workbench-store';

export async function POST(request: NextRequest) {
  const payload = (await request.json().catch(() => ({}))) as Partial<WorkbenchSettings>;
  const settings = effectiveSettings();
  try {
    const answer = await callChatCompletion(
      {
        qa_base_url: payload.qa_base_url ?? settings.qa_base_url,
        qa_api_key: payload.qa_api_key || settings.qa_api_key,
        qa_model: payload.qa_model ?? settings.qa_model,
      },
      [
        { role: 'system', content: 'Reply with exactly: ok' },
        { role: 'user', content: 'Connection test.' },
      ],
      8,
    );
    return NextResponse.json({ ok: true, answer });
  } catch (error) {
    return NextResponse.json({ ok: false, detail: error instanceof Error ? error.message : 'Connection test failed.' }, { status: 502 });
  }
}
