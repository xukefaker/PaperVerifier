import { proxyToBackend } from "@/lib/backend-proxy";

type RouteContext = {
  params: Promise<{
    project_id: string;
    paper_id: string;
  }>;
};

export async function GET(_: Request, context: RouteContext) {
  const { project_id, paper_id } = await context.params;
  return proxyToBackend(
    `/projects/${encodeURIComponent(project_id)}/papers/${encodeURIComponent(paper_id)}/session`,
  );
}

export async function PUT(request: Request, context: RouteContext) {
  const { project_id, paper_id } = await context.params;
  return proxyToBackend(
    `/projects/${encodeURIComponent(project_id)}/papers/${encodeURIComponent(paper_id)}/session`,
    {
      method: "PUT",
      headers: {
        "Content-Type": request.headers.get("content-type") ?? "application/json",
      },
      body: await request.text(),
    },
  );
}
