import { NextRequest, NextResponse } from 'next/server';
import { createUploadJob } from '@/lib/workbench-store';

export async function POST(request: NextRequest) {
  const form = await request.formData();
  const file = form.get('file');
  if (!(file instanceof File)) {
    return NextResponse.json({ detail: 'PDF file is required.' }, { status: 400 });
  }
  const data = Buffer.from(await file.arrayBuffer());
  return NextResponse.json({ job: createUploadJob({ fileName: file.name, data }) });
}
