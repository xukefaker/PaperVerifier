import type {
  PaperChatRequest,
  PaperChatResponse,
  ProjectDetailResponse,
  ProjectListResponse,
  ProjectMutationResponse,
  ProjectPaperSession,
  ProjectSearchThread,
  PaperViewerResponse,
  SearchJobResult,
  SearchJobStatus,
  SearchTrace,
} from '@/lib/types';

type JsonInit = {
  method?: 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH';
  body?: unknown;
};

async function requestJson<T>(path: string, init: JsonInit = {}): Promise<T> {
  const response = await fetch(path, {
    method: init.method ?? 'GET',
    headers: init.body === undefined ? undefined : { 'Content-Type': 'application/json' },
    body: init.body === undefined ? undefined : JSON.stringify(init.body),
    cache: 'no-store',
  });

  if (!response.ok) {
    let message = `Request failed with status ${response.status}.`;
    try {
      const payload = (await response.json()) as { detail?: string };
      if (payload.detail) message = payload.detail;
    } catch {
      const text = await response.text();
      if (text) message = text;
    }
    throw new Error(message);
  }

  return (await response.json()) as T;
}

export function createSearchJob(payload: { query: string; top_k: number; display_k?: number }) {
  return requestJson<SearchJobStatus>('/api/search/jobs', {
    method: 'POST',
    body: payload,
  });
}

export function fetchSearchJob(jobId: string) {
  return requestJson<SearchJobStatus>(`/api/search/jobs/${encodeURIComponent(jobId)}`);
}

export function fetchSearchJobResult(jobId: string) {
  return requestJson<SearchJobResult>(`/api/search/jobs/${encodeURIComponent(jobId)}/result`);
}

export function fetchTrace(traceId: string) {
  return requestJson<SearchTrace>(`/api/traces/${encodeURIComponent(traceId)}`);
}

export function chatWithPaper(payload: PaperChatRequest) {
  return requestJson<PaperChatResponse>('/api/chat/paper', {
    method: 'POST',
    body: payload,
  });
}

export async function fetchPaperContentList(paperId: string): Promise<unknown> {
  return requestJson<unknown>(`/api/papers/${encodeURIComponent(paperId)}/content_list`);
}

export function fetchPaperViewer(paperId: string) {
  return requestJson<PaperViewerResponse>(`/api/papers/${encodeURIComponent(paperId)}/viewer`);
}

export function paperZoteroPageUrl(paperId: string) {
  return `/api/papers/${encodeURIComponent(paperId)}/zotero`;
}

export function paperBibtexExportUrl(paperId: string) {
  return `/api/papers/${encodeURIComponent(paperId)}/export.bib`;
}

export function paperRisExportUrl(paperId: string) {
  return `/api/papers/${encodeURIComponent(paperId)}/export.ris`;
}

export function listProjects() {
  return requestJson<ProjectListResponse>('/api/projects');
}

export function createProject(payload: { title: string }) {
  return requestJson<{ project_id: string; title: string; created_at: string; updated_at: string; search_thread_count: number; paper_session_count: number }>(
    '/api/projects',
    {
      method: 'POST',
      body: payload,
    },
  );
}

export function fetchProject(projectId: string) {
  return requestJson<ProjectDetailResponse>(`/api/projects/${encodeURIComponent(projectId)}`);
}

export function clearProject(projectId: string) {
  return requestJson<ProjectMutationResponse>(`/api/projects/${encodeURIComponent(projectId)}/clear`, {
    method: 'POST',
  });
}

export function deleteProject(projectId: string) {
  return requestJson<ProjectMutationResponse>(`/api/projects/${encodeURIComponent(projectId)}`, {
    method: 'DELETE',
  });
}

export function renameProject(projectId: string, payload: { title: string }) {
  return requestJson<{ project_id: string; title: string; created_at: string; updated_at: string; search_thread_count: number; paper_session_count: number }>(
    `/api/projects/${encodeURIComponent(projectId)}`,
    {
      method: 'PATCH',
      body: payload,
    },
  );
}

export function upsertProjectThread(
  projectId: string,
  threadId: string,
  payload: {
    query: string;
    trace_id?: string | null;
    result_counts: Record<string, number>;
    paper_ids: string[];
  },
) {
  return requestJson<ProjectSearchThread>(
    `/api/projects/${encodeURIComponent(projectId)}/threads/${encodeURIComponent(threadId)}`,
    {
      method: 'PUT',
      body: payload,
    },
  );
}

export function upsertProjectPaperSession(
  projectId: string,
  paperId: string,
  payload: {
    paper_title?: string | null;
    source_thread_id?: string | null;
    chat_history: {
      role: 'user' | 'assistant';
      content: string;
      citations: {
        evidence_id: string;
        page_start: number;
        page_end: number;
        section_path: string[];
        snippet: string;
      }[];
    }[];
    last_active_evidence_id?: string | null;
  },
) {
  return requestJson<ProjectPaperSession>(
    `/api/projects/${encodeURIComponent(projectId)}/papers/${encodeURIComponent(paperId)}/session`,
    {
      method: 'PUT',
      body: payload,
    },
  );
}
