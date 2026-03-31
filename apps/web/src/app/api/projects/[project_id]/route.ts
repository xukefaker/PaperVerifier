import { proxyToBackend } from "@/lib/backend-proxy";

type RouteContext = {
  params: Promise<{
    project_id: string;
  }>;
};

export async function GET(_: Request, context: RouteContext) {
  const { project_id } = await context.params;
  return proxyToBackend(`/projects/${encodeURIComponent(project_id)}`);
}

export async function DELETE(_: Request, context: RouteContext) {
  const { project_id } = await context.params;
  return proxyToBackend(`/projects/${encodeURIComponent(project_id)}`, {
    method: "DELETE",
  });
}

export async function PATCH(request: Request, context: RouteContext) {
  const { project_id } = await context.params;
  return proxyToBackend(`/projects/${encodeURIComponent(project_id)}`, {
    method: "PATCH",
    headers: {
      "Content-Type": request.headers.get("content-type") ?? "application/json",
    },
    body: await request.text(),
  });
}
