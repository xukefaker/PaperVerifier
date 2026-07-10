import { NextRequest, NextResponse } from 'next/server';
import { publicSettings, saveSettingsPatch, type WorkbenchSettings } from '@/lib/workbench-store';

export function GET() {
  return NextResponse.json(publicSettings());
}

export async function PATCH(request: NextRequest) {
  const payload = (await request.json().catch(() => ({}))) as Partial<WorkbenchSettings>;
  return NextResponse.json(saveSettingsPatch(payload));
}
