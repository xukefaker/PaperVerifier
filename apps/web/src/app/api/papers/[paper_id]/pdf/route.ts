import { proxyToBackend } from "@/lib/backend-proxy";

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ paper_id: string }> },
) {
  const { paper_id } = await params;
  return proxyToBackend(`/papers/${encodeURIComponent(paper_id)}/pdf`);
}
