import { proxyToBackend } from "@/lib/backend-proxy";

type RouteContext = {
  params: Promise<{
    project_id: string;
  }>;
};

export async function POST(_: Request, context: RouteContext) {
  const { project_id } = await context.params;
  return proxyToBackend(`/projects/${encodeURIComponent(project_id)}/clear`, {
    method: "POST",
  });
}
