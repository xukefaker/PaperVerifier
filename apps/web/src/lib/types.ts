export type HealthResponse = {
  status: string;
  data_dir: string;
  index_state: Record<string, unknown>;
  counts: {
    papers: number;
    chunks: number;
    traces: number;
  };
  jobs: {
    total_jobs: number;
    queued: number;
    running: number;
    completed: number;
    failed: number;
  };
};

export type HealthSummary =
  | { kind: 'ready'; data: HealthResponse }
  | { kind: 'error'; message: string };

export type SearchJobProgress = {
  stage_index: number;
  stage_total: number;
  stage_progress: number;
  overall_progress: number;
  completed_items?: number | null;
  total_items?: number | null;
};

export type SearchJobStatus = {
  job_id: string;
  status: 'queued' | 'running' | 'completed' | 'failed';
  stage: string;
  message: string;
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  elapsed_ms: number;
  trace_id?: string | null;
  error?: string | null;
  progress?: SearchJobProgress | null;
};

export type ScopeConstraints = {
  venues: string[];
  years: number[];
  tracks?: string[];
  paper_ids?: string[];
};

export type QueryAspect = {
  aspect_id: string;
  query: string;
  weight: number;
};

export type VerifierRubric = {
  must_satisfy: string[];
  should_satisfy: string[];
  rejection_rules: string[];
};

export type QueryEvidenceBucket = {
  bucket_id: string;
  description: string;
  queries: string[];
  target_chunks: number;
};

export type QueryPlan = {
  mode: string;
  user_query: string;
  global_query: string;
  scope_constraints: ScopeConstraints;
  entity_terms: string[];
  exact_phrases: string[];
  aspect_queries: QueryAspect[];
  verifier_rubric: VerifierRubric;
  evidence_buckets: QueryEvidenceBucket[];
};

export type RecallItem = {
  item_id: string;
  source: string;
  score: number;
  rank: number;
  aspect_id?: string | null;
};

export type EvidenceChunk = {
  paper_id: string;
  bucket_id: string;
  chunk_id: string;
  score: number;
  source_query: string;
  heading: string;
  section_path: string[];
  page_start: number;
  page_end: number;
  text: string;
};

export type PaperResult = {
  paper_id: string;
  title: string;
  score: number;
  coarse_score: number;
  verifier_score: number;
  venue: string;
  year: number;
  track?: string | null;
  verdict: string;
  entity_role?: string | null;
  satisfied_constraints: string[];
  missing_constraints: string[];
  confidence: number;
  rationale: string;
  rationale_structured?: {
    main_reason: string | null;
    matching_points: string[];
  } | null;
  matched_sections?: string[];
  matched_sections_summary?: Record<string, number>;
  evidence_chunks: Record<string, EvidenceChunk[]>;
  main_image_url?: string | null;
  abstract?: string | null;
  authors?: string[] | null;
  affiliations?: string[] | null;
  authors_structured?: {
    name: string;
    affiliation: string | null;
  }[] | null;
  structured_summary?: {
    methodology: string | null;
    benchmarks: string[];
    key_findings: string[];
  } | null;
  enriched_metadata?: {
    citations: number | null;
    code_url: string | null;
  } | null;
};

export type SearchJobResult = {
  job_id: string;
  query: string;
  trace_id: string;
  mode: string;
  counts: {
    satisfied: number;
    partial: number;
    rejected: number;
  };
  display_results: PaperResult[];
  satisfied: PaperResult[];
  partial: PaperResult[];
  rejected: PaperResult[];
};

export type PaperChatRequest = {
  paper_id: string;
  query: string;
  history: { role: 'user' | 'assistant'; content: string }[];
};

export type PaperChatCitation = {
  evidence_id: string;
  page_start: number;
  page_end: number;
  section_path: string[];
  snippet: string;
  html?: string | null;
};

export type PaperChatEvidence = {
  evidence_id: string;
  evidence_type: string;
  heading: string;
  section_path: string[];
  page_start: number;
  page_end: number;
  snippet: string;
  html?: string | null;
};

export type PaperChatVerifier = {
  support_verdict: string;
  alignment_verdict: string;
  interpretation_verdict: string;
  competition_verdict: string;
  strongest_competitor_id?: string | null;
  confidence: number;
  verified_evidence_ids: string[];
  failure_reason?: string | null;
};

export type PaperChatResponse = {
  paper_id: string;
  decision?: 'ask_clarification' | 'answer' | 'unsupported';
  answer: string;
  rewritten_query?: string | null;
  uncertainty_note?: string | null;
  citations: PaperChatCitation[];
  evidence?: PaperChatEvidence[];
  verifier?: PaperChatVerifier | null;
};

export type PaperViewerBlock = {
  block_id: string;
  block_type: 'section_heading' | 'text_block' | 'figure_block' | 'table_block' | 'equation_block' | 'list_block';
  text: string;
  caption?: string | null;
  footnote?: string | null;
  image_url?: string | null;
  page_start: number;
  page_end: number;
  section_path: string[];
  table?: {
    rows: {
      cells: {
        text: string;
        colspan?: number | null;
        rowspan?: number | null;
        is_header?: boolean | null;
      }[];
    }[];
  } | null;
};

export type PaperViewerReference = {
  ordinal: number;
  raw_text: string;
  year?: number | null;
  doi?: string | null;
  arxiv_id?: string | null;
  url?: string | null;
};

export type ViewerMode = 'manuscript' | 'pdf';

export type EvidenceNavigationMarkdownTarget = {
  block_id: string;
  section_block_id?: string | null;
};

export type EvidenceNavigationPdfRect = {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
};

export type EvidenceNavigationPdfPageTarget = {
  page: number;
  width: number;
  height: number;
  bboxes: EvidenceNavigationPdfRect[];
};

export type EvidenceNavigationPdfTarget = {
  primary_page: number;
  pages: EvidenceNavigationPdfPageTarget[];
};

export type EvidenceNavigationTarget = {
  evidence_id: string;
  markdown_target?: EvidenceNavigationMarkdownTarget | null;
  pdf_target?: EvidenceNavigationPdfTarget | null;
};

export type PaperViewerResponse = {
  paper_id: string;
  title: string;
  pdf_url?: string | null;
  display_header?: {
    authors_structured: {
      name: string;
      affiliation: string | null;
    }[];
    affiliations: string[];
  } | null;
  blocks: PaperViewerBlock[];
  evidence_navigation_map: Record<string, EvidenceNavigationTarget>;
  references?: PaperViewerReference[];
};

export type SearchTraceFilterSummary = Record<string, unknown> & {
  allowed_paper_ids?: number;
  candidate_pool_count?: number;
  source_sizes?: Record<string, number>;
  section_narrowing?: Record<string, unknown>;
};

export type SearchTraceVerifierSummary = Record<string, unknown> & {
  candidate_pool_count?: number;
  satisfied_count?: number;
  partial_count?: number;
  rejected_count?: number;
  reranker_backend?: string;
};

export type TokenUsage = {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  cost_estimate_usd?: number | null;
};

export type SearchTrace = {
  trace_id: string;
  created_at: string;
  mode: string;
  user_query: string;
  query_plan: QueryPlan;
  paper_recall: RecallItem[];
  evidence_packs: Record<string, Record<string, EvidenceChunk[]>>;
  filter_summary: SearchTraceFilterSummary;
  verifier_summary: SearchTraceVerifierSummary;
  final_results: Record<string, PaperResult[]>;
  timings_ms: Record<string, number>;
  token_usage: TokenUsage;
};

export type ProjectChatCitation = {
  evidence_id: string;
  page_start: number;
  page_end: number;
  section_path: string[];
  snippet: string;
};

export type ProjectChatMessage = {
  role: 'user' | 'assistant';
  content: string;
  citations: ProjectChatCitation[];
};

export type ProjectSearchThread = {
  project_id: string;
  thread_id: string;
  query: string;
  trace_id?: string | null;
  created_at: string;
  updated_at: string;
  result_counts: Record<string, number>;
  paper_ids: string[];
};

export type ProjectPaperSession = {
  project_id: string;
  paper_id: string;
  paper_title?: string | null;
  source_thread_id?: string | null;
  created_at: string;
  updated_at: string;
  chat_history: ProjectChatMessage[];
  last_active_evidence_id?: string | null;
};

export type ProjectSummary = {
  project_id: string;
  title: string;
  created_at: string;
  updated_at: string;
  search_thread_count: number;
  paper_session_count: number;
};

export type ProjectListResponse = {
  projects: ProjectSummary[];
};

export type ProjectDetailResponse = {
  project: ProjectSummary;
  threads: ProjectSearchThread[];
  paper_sessions: ProjectPaperSession[];
};

export type ProjectMutationResponse = {
  project_id: string;
  status: 'cleared' | 'deleted';
};
