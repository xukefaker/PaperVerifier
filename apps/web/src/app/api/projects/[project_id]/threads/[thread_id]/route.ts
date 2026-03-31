import { proxyToBackend } from "@/lib/backend-proxy";

type RouteContext = {
  params: Promise<{
    project_id: string;
    thread_id: string;
  }>;
};

export async function GET(_: Request, context: RouteContext) {
  const { project_id, thread_id } = await context.params;
  return proxyToBackend(
    `/projects/${encodeURIComponent(project_id)}/threads/${encodeURIComponent(thread_id)}`,
  );
}

export async function PUT(request: Request, context: RouteContext) {
  const { project_id, thread_id } = await context.params;
  return proxyToBackend(
    `/projects/${encodeURIComponent(project_id)}/threads/${encodeURIComponent(thread_id)}`,
    {
      method: "PUT",
      headers: {
        "Content-Type": request.headers.get("content-type") ?? "application/json",
      },
      body: await request.text(),
    },
  );
}
