import { readFileSync } from 'node:fs';
import { extname } from 'node:path';
import { NextResponse } from 'next/server';
import { paperImagePath } from '@/lib/workbench-store';

export async function GET(
  _request: Request,
  {
    params,
  }: {
    params: Promise<{ paper_id: string; image_path: string[] }>;
  },
) {
  const { paper_id, image_path } = await params;
  const imageName = image_path.join('/');
  const localPath = paperImagePath(paper_id, imageName);
  if (!localPath) {
    return NextResponse.json({ detail: 'Image not found for this paper.' }, { status: 404 });
  }
  const ext = extname(localPath).toLowerCase();
  const contentType = ext === '.jpg' || ext === '.jpeg' ? 'image/jpeg' : ext === '.webp' ? 'image/webp' : 'image/png';
  return new NextResponse(readFileSync(localPath), {
    headers: {
      'content-type': contentType,
      'cache-control': 'no-store',
    },
  });
}
