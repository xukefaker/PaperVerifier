'use client';

import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { AlertTriangle, ChevronDown, ChevronUp, FolderOpen, Loader2, Plus, Search } from 'lucide-react';

import { usePresence } from '@/lib/animation/use-presence';
import { useSequencedReveal } from '@/lib/animation/use-sequenced-reveal';
import { runViewTransition } from '@/lib/animation/view-transition';
import { ActivityMark, useTransientActivityMode } from '@/components/activity-mark';
import { GlobalSearchPalette } from '@/components/global-search-palette';
import { PaperResultCard } from '@/components/paper-result-card';
import { ProjectWorkspacePanel } from '@/components/project-workspace-panel';
import { QuickPeekPanel } from '@/components/quick-peek-panel';
import { SplitPaneWorkspace } from '@/components/split-pane-workspace';
import {
  clearProject,
  createProject,
  createSearchJob,
  deleteProject,
  fetchCorpusCatalog,
  fetchProject,
  fetchSearchJob,
  fetchSearchJobResult,
  fetchTrace,
  listProjects,
  updateProject,
  upsertProjectPaperSession,
  upsertProjectThread,
} from '@/lib/client-api';
import { clearCurrentProjectId, loadCurrentProjectId, saveCurrentProjectId } from '@/lib/project-ui-state';
import { APP_NAME, APP_TAGLINE } from '@/lib/branding';
import type {
  CorpusCatalogEntry,
  PaperChatCitation,
  PaperResult,
  ProjectDetailResponse,
  ProjectPaperSession,
  ProjectSearchThread,
  ProjectSummary,
  SearchJobProgress,
  SearchJobStatus,
  SearchTrace,
} from '@/lib/types';

type SearchMessage = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  type?: 'text' | 'search_results';
  results?: PaperResult[];
  status?: 'loading' | 'completed' | 'error';
  currentStage?: string;
  stageMessage?: string;
  progress?: SearchJobProgress | null;
  traceId?: string | null;
  trace?: SearchTrace | null;
  traceState?: 'idle' | 'loading' | 'loaded' | 'error';
  traceError?: string | null;
  traceOpen?: boolean;
};

export type ChatMessage = {
  role: 'user' | 'assistant';
  content: string;
  citations?: PaperChatCitation[];
};

type WorkspaceActionIntent = 'clear' | 'delete';

type ActiveSearchRun = {
  assistantId: string;
  query: string;
  status: SearchJobStatus['status'];
  currentStage?: string;
  stageMessage?: string;
  progress?: SearchJobProgress | null;
  startedAt: number;
  stageStartedAt: number;
};

function buildDisplayResultsFromTrace(thread: ProjectSearchThread, trace: SearchTrace | null): PaperResult[] {
  if (!trace) {
    return [];
  }
  const satisfied = trace.final_results.satisfied ?? [];
  const partial = trace.final_results.partial ?? [];
  const allowedIds = new Set(thread.paper_ids ?? []);
  const pool = [...satisfied, ...partial];
  if (allowedIds.size === 0) {
    return pool.slice(0, 10);
  }
  const resultById = new Map(pool.map((paper) => [paper.paper_id, paper]));
  return thread.paper_ids
    .map((paperId) => resultById.get(paperId))
    .filter((paper): paper is PaperResult => paper != null)
    .slice(0, 10);
}

function buildRestoredSearchMessages(thread: ProjectSearchThread, trace: SearchTrace | null): SearchMessage[] {
  const displayResults = buildDisplayResultsFromTrace(thread, trace);
  return [
    {
      id: `${thread.thread_id}-user`,
      role: 'user',
      content: thread.query,
    },
    {
      id: `${thread.thread_id}-assistant`,
      role: 'assistant',
      content:
        displayResults.length > 0
          ? `Restored search. ${displayResults.length} papers are ready for review.`
          : 'Restored search thread. Trace-backed paper results are not available right now.',
      status: 'completed',
      type: 'search_results',
      results: displayResults,
      progress: null,
      traceId: thread.trace_id ?? null,
      trace,
      traceState: trace ? 'loaded' : 'idle',
      traceError: null,
      traceOpen: false,
    },
  ];
}

function projectSessionToChatMessages(session: ProjectPaperSession): ChatMessage[] {
  return session.chat_history.map((message) => ({
    role: message.role,
    content: message.content,
    citations: message.citations.map((citation) => ({
      evidence_id: citation.evidence_id,
      page_start: citation.page_start,
      page_end: citation.page_end,
      section_path: citation.section_path,
      snippet: citation.snippet,
      html: null,
    })),
  }));
}

function buildChatSessionsMap(sessions: ProjectPaperSession[]): Record<string, ChatMessage[]> {
  return Object.fromEntries(sessions.map((session) => [session.paper_id, projectSessionToChatMessages(session)]));
}

function serializeProjectChatHistory(messages: ChatMessage[]) {
  return messages.map((message) => ({
    role: message.role,
    content: message.content,
    citations: (message.citations ?? []).map((citation) => ({
      evidence_id: citation.evidence_id,
      page_start: citation.page_start,
      page_end: citation.page_end,
      section_path: citation.section_path,
      snippet: citation.snippet,
    })),
  }));
}

function buildProjectSessionSignature(payload: {
  paperId: string;
  paperTitle: string | null;
  sourceThreadId: string | null;
  chatHistory: ChatMessage[];
}): string {
  return JSON.stringify({
    paperId: payload.paperId,
    paperTitle: payload.paperTitle,
    sourceThreadId: payload.sourceThreadId,
    chatHistory: serializeProjectChatHistory(payload.chatHistory),
  });
}

const SAMPLE_QUERIES = [
  'Find papers on donor-acceptor covalent organic frameworks for coupled CO2 reduction and water oxidation.',
  'Find papers on covalent organic framework photocatalysts for selective two-electron oxygen reduction to hydrogen peroxide.',
  'Find papers on oxygen-vacancy-rich TiO2 for hydroperoxy-mediated selective benzene oxidation to phenol.',
];

const QUERY_SUGGESTION_GROUPS = [
  {
    label: 'Photocatalysis',
    items: [
      'Find papers on donor-acceptor covalent organic frameworks for coupled CO2 reduction and water oxidation.',
      'Find papers on covalent organic framework photocatalysts for selective two-electron oxygen reduction to hydrogen peroxide.',
      'Find papers on internal electric fields in MOF/COF heterojunctions for photocatalytic water splitting.',
    ],
  },
  {
    label: 'Reaction / Materials',
    items: [
      'Find papers on atomically dispersed cobalt or transition-metal sites in photocatalysts for nitrogen fixation.',
      'Find papers on oxygen-vacancy-rich TiO2 for hydroperoxy-mediated selective benzene oxidation to phenol.',
      'Find papers on light-triggered depolymerization of polypinacols for closed-loop chemical recycling.',
    ],
  },
] as const;

const SEARCH_STAGE_SEQUENCE = [
  { id: 'loading_index', short: 'Index', label: 'Load offline indexes' },
  { id: 'planning_query', short: 'Plan', label: 'Parse the query' },
  { id: 'candidate_generation', short: 'Recall', label: 'Generate candidate papers' },
  { id: 'section_narrowing', short: 'Sections', label: 'Narrow to relevant sections' },
  { id: 'evidence_assembly', short: 'Evidence', label: 'Assemble evidence packs' },
  { id: 'final_verifier', short: 'Verify', label: 'Run the final verifier' },
  { id: 'saving_trace', short: 'Save', label: 'Persist the trace' },
] as const;

const SEARCH_STAGE_META = Object.fromEntries(
  SEARCH_STAGE_SEQUENCE.map((stage) => [stage.id, stage]),
) as Record<string, (typeof SEARCH_STAGE_SEQUENCE)[number]>;

function clampPercentage(value: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }
  return Math.max(0, Math.min(100, value));
}

function progressSignature(progress: SearchJobProgress | null | undefined): string {
  if (!progress) {
    return 'none';
  }

  return [
    progress.stage_index,
    progress.stage_total,
    progress.stage_progress,
    progress.overall_progress,
    progress.completed_items ?? '',
    progress.total_items ?? '',
  ].join(':');
}

function getStageMeta(stage: string | undefined) {
  if (!stage) {
    return null;
  }
  return SEARCH_STAGE_META[stage] ?? null;
}

function compactInteger(value: number): string {
  return new Intl.NumberFormat('en', {
    notation: value >= 1000 ? 'compact' : 'standard',
    maximumFractionDigits: value >= 1000 ? 1 : 0,
  }).format(value);
}

function titleCaseToken(value: string): string {
  return value
    .split('_')
    .map((part) => (part.length <= 3 ? part.toUpperCase() : part[0]!.toUpperCase() + part.slice(1)))
    .join(' ');
}

function formatTimingMs(value: number | undefined): string {
  if (!value || !Number.isFinite(value) || value <= 0) {
    return '0 ms';
  }
  if (value < 1000) {
    return `${Math.round(value)} ms`;
  }
  return `${(value / 1000).toFixed(value >= 10_000 ? 0 : 1)} s`;
}

function formatElapsedCompact(valueMs: number): string {
  const totalSeconds = Math.max(0, Math.floor(valueMs / 1000));
  if (totalSeconds < 60) {
    return `${totalSeconds}s`;
  }
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}m ${seconds}s`;
}

function formatProgressDetail(stage: string | undefined, progress: SearchJobProgress | null | undefined): string | null {
  if (!progress) {
    return null;
  }

  const completed = progress.completed_items;
  const total = progress.total_items;
  if (completed != null && total != null && total > 0) {
    if (stage === 'section_narrowing') {
      return `${completed} / ${total} candidate papers narrowed to relevant sections`;
    }
    if (stage === 'evidence_assembly') {
      return `${completed} / ${total} candidate papers assembled into evidence packs`;
    }
    if (stage === 'final_verifier') {
      return `${completed} / ${total} candidate papers verified by the final model`;
    }
  }
  return null;
}

const SearchProgressPanel = memo(function SearchProgressPanel({
  message,
  stageElapsedMs = 0,
  totalElapsedMs = 0,
  compact = false,
}: {
  message: SearchMessage;
  stageElapsedMs?: number;
  totalElapsedMs?: number;
  compact?: boolean;
}) {
  const progress = message.progress ?? null;
  const stageMeta = getStageMeta(message.currentStage);
  const isCompleted = message.status === 'completed';
  const overallProgressPercent = clampPercentage(Math.round(((isCompleted ? 1 : (progress?.overall_progress ?? 0)) * 100)));
  const detail = formatProgressDetail(message.currentStage, progress);
  const activeStageIndex = progress?.stage_index ?? 0;
  const activeStageShort = isCompleted ? 'Ready' : stageMeta?.short ?? 'Run';
  const longRunning = !isCompleted && stageElapsedMs >= 8000;
  const steadyNote = isCompleted
    ? `Search wrapped in ${formatElapsedCompact(totalElapsedMs)}. Preparing the review view.`
    : longRunning
      ? `Still working in ${stageMeta?.label ?? 'the current stage'} • ${formatElapsedCompact(stageElapsedMs)} in this stage`
      : null;

  return (
    <div className={`search-progress-panel mt-2 w-full overflow-hidden rounded-[1.7rem] border border-slate-200/80 bg-[linear-gradient(180deg,rgba(255,255,255,0.99),rgba(249,250,252,0.97))] ${compact ? 'search-progress-panel--compact max-w-[760px] p-5 shadow-[0_10px_24px_rgba(15,23,42,0.04)]' : 'max-w-[620px] p-5 shadow-[0_18px_42px_rgba(15,23,42,0.06)]'}`}>
      <div className="search-progress-panel__rail" />
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0 flex items-start gap-3">
          <ActivityMark mode={isCompleted ? 'done' : 'active'} label={null} layout="inline" size="md" minimal={compact} />
          <div className="min-w-0 flex-1 pt-0.5 text-left">
            <div className="truncate text-[1rem] font-semibold tracking-tight text-slate-950">
              {isCompleted ? 'Search complete' : stageMeta?.label ?? 'Preparing the search pipeline'}
            </div>
            <div className="mt-1 text-[0.8rem] leading-6 text-slate-500">
              {isCompleted ? 'The evidence pass has finished and the result view is being prepared.' : message.stageMessage ?? 'Searching across the current corpus.'}
            </div>
          </div>
        </div>
        <div className="shrink-0 text-right">
          <div className="text-[1.28rem] font-black tracking-tight text-slate-950">{overallProgressPercent}%</div>
          <div className="text-[0.62rem] font-bold uppercase tracking-[0.2em] text-slate-400">{activeStageShort}</div>
        </div>
      </div>

      <div className="mt-4 h-2 overflow-hidden rounded-full bg-slate-100">
        <div
          className="search-progress-fill h-full rounded-full bg-[linear-gradient(90deg,#4f46e5_0%,#6366f1_55%,#60a5fa_100%)] transition-[width] duration-500"
          style={{ width: `${overallProgressPercent}%` }}
        />
      </div>

      {compact ? (
        <div className="mt-4 flex items-center gap-3">
          <div className="search-stage-chip search-stage-chip--compact search-stage-chip--active inline-flex items-center gap-2 rounded-full border border-slate-300 bg-slate-950 px-3 py-2 text-[0.64rem] font-bold uppercase tracking-[0.16em] text-white">
            <span className="h-1.5 w-1.5 rounded-full bg-cyan-300" />
            {activeStageShort}
          </div>
          <div className="rounded-full border border-slate-200 bg-slate-50 px-3 py-2 text-[0.62rem] font-bold uppercase tracking-[0.16em] text-slate-400">
            Stage {progress?.stage_index ?? 0} of {progress?.stage_total ?? SEARCH_STAGE_SEQUENCE.length}
          </div>
        </div>
      ) : (
        <div className="mt-4 flex flex-wrap gap-2">
          {SEARCH_STAGE_SEQUENCE.map((stage, index) => {
            const stageNumber = index + 1;
            const isCompleted = activeStageIndex > stageNumber || message.status === 'completed';
            const isActive = message.currentStage === stage.id;
            return (
              <div
                key={stage.id}
                className={`search-stage-chip inline-flex items-center gap-2 rounded-full border px-3 py-2 text-[0.64rem] font-bold uppercase tracking-[0.16em] transition-colors ${
                  isCompleted
                    ? 'search-stage-chip--completed border-indigo-200 bg-indigo-50 text-indigo-700'
                    : isActive
                      ? 'search-stage-chip--active border-slate-300 bg-slate-950 text-white'
                      : 'border-slate-200 bg-slate-50 text-slate-400'
                }`}
              >
                <span
                  className={`h-1.5 w-1.5 rounded-full ${
                    isCompleted ? 'bg-indigo-500' : isActive ? 'bg-cyan-300' : 'bg-slate-300'
                  }`}
                />
                {stage.short}
              </div>
            );
          })}
        </div>
      )}

      {compact ? null : (
        <>
          {detail ? <div className="mt-3 text-[0.8rem] leading-6 text-slate-500">{detail}</div> : null}
          {steadyNote ? <div className="mt-2 text-[0.75rem] font-medium leading-6 text-slate-400">{steadyNote}</div> : null}
        </>
      )}
    </div>
  );
});

function SearchSuggestionPanel({
  onSelect,
  variant,
}: {
  onSelect: (query: string) => void;
  variant: 'hero' | 'dock';
}) {
  const compact = variant === 'dock';

  return (
    <div className={`rounded-[1.65rem] border border-white/85 bg-white/92 shadow-scholar-lg backdrop-blur-xl ${
      compact ? 'p-4' : 'p-5'
    }`}>
      <div className='mb-3 flex items-center justify-between gap-3'>
        <div className='text-[0.68rem] font-bold uppercase tracking-[0.22em] text-slate-400'>Query suggestions</div>
        <div className='text-[0.68rem] font-medium text-slate-400'>Click to search</div>
      </div>

      <div className={`grid gap-3 ${compact ? 'lg:grid-cols-2' : 'md:grid-cols-2'}`}>
        {QUERY_SUGGESTION_GROUPS.map((group) => (
          <div key={group.label} className='rounded-[1.3rem] border border-slate-100 bg-slate-50/80 p-3.5'>
            <div className='mb-3 text-[0.66rem] font-bold uppercase tracking-[0.22em] text-slate-400'>{group.label}</div>
            <div className='flex flex-wrap gap-2'>
              {group.items.map((query) => (
                <button
                  key={query}
                  type='button'
                  onMouseDown={(event) => {
                    event.preventDefault();
                  }}
                  onClick={() => onSelect(query)}
                  className='rounded-full border border-white/80 bg-white px-3.5 py-2 text-left text-[0.77rem] font-medium leading-5 text-slate-600 shadow-scholar-sm transition hover:border-indigo-200 hover:text-indigo-600'
                >
                  {query}
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

const SearchRunningSkeletonGrid = memo(function SearchRunningSkeletonGrid() {
  return (
    <div className='grid grid-cols-1 gap-5 lg:grid-cols-3'>
      {Array.from({ length: 3 }, (_, index) => (
        <article
          key={`running-skeleton-${index}`}
          className='search-running-skeleton-card relative overflow-hidden rounded-[1.8rem] border border-slate-200/80 bg-white/96 p-5 shadow-[0_10px_24px_rgba(15,23,42,0.04)]'
        >
          <div className='search-running-skeleton-bar h-5 w-24 rounded-full' />
          <div className='mt-6 space-y-3'>
            <div className='search-running-skeleton-bar h-4 w-[88%] rounded-full' />
            <div className='search-running-skeleton-bar h-4 w-[72%] rounded-full' />
            <div className='search-running-skeleton-bar h-4 w-[64%] rounded-full' />
          </div>
          <div className='mt-8 space-y-2'>
            <div className='search-running-skeleton-bar h-3.5 w-full rounded-full' />
            <div className='search-running-skeleton-bar h-3.5 w-[82%] rounded-full' />
          </div>
          <div className='mt-8 border-t border-slate-100 pt-4'>
            <div className='search-running-skeleton-bar h-10 w-full rounded-[1.1rem]' />
          </div>
        </article>
      ))}
    </div>
  );
});

const SearchRunningPreview = memo(function SearchRunningPreview({
  run,
}: {
  run: ActiveSearchRun;
}) {
  const [clockMs, setClockMs] = useState(() => Date.now());

  useEffect(() => {
    const timerId = window.setInterval(() => {
      setClockMs(Date.now());
    }, 1000);
    return () => window.clearInterval(timerId);
  }, []);

  const progressMessage: SearchMessage = {
    id: run.assistantId,
    role: 'assistant',
    content: 'Running evidence-aware scholarly retrieval.',
    status: run.status === 'completed' ? 'completed' : 'loading',
    currentStage: run.currentStage,
    stageMessage: run.stageMessage,
    progress: run.progress ?? null,
  };
  const stageElapsedMs = Math.max(0, clockMs - run.stageStartedAt);
  const totalElapsedMs = Math.max(0, clockMs - run.startedAt);

  return (
    <main className="relative z-30 flex flex-1 items-start justify-center px-5 pb-14 pt-10 sm:px-8 sm:pt-14">
      <div className="w-full max-w-[1040px]">
        <div className="mx-auto flex max-w-[860px] flex-col items-center text-center">
          <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-4 py-2 text-[0.68rem] font-bold uppercase tracking-[0.22em] text-slate-500">
            <span className="h-2 w-2 rounded-full bg-indigo-500" />
            Search in progress
          </div>
          <div className="search-query-bubble w-full max-w-[46rem] rounded-[2rem] border border-slate-200/80 bg-white px-7 py-6 shadow-[0_12px_32px_rgba(15,23,42,0.04)]">
            <div className="mb-3 text-[0.68rem] font-bold uppercase tracking-[0.24em] text-slate-400">Query</div>
            <div className="font-scholar text-[1.12rem] italic leading-9 text-slate-700">{run.query}</div>
          </div>
          <div className="mt-6 w-full flex justify-center">
            <SearchProgressPanel message={progressMessage} stageElapsedMs={stageElapsedMs} totalElapsedMs={totalElapsedMs} compact />
          </div>
        </div>

        <div className="mx-auto mt-10 max-w-[980px]">
          <div className="mb-4 text-center text-[0.68rem] font-bold uppercase tracking-[0.22em] text-slate-400">
            Preparing result cards
          </div>
          <SearchRunningSkeletonGrid />
        </div>
      </div>
    </main>
  );
});

function getSearchResultHeadline(message: SearchMessage): string {
  if ((message.content ?? '').toLowerCase().startsWith('restored search')) {
    return 'Workspace restored';
  }
  return 'Search complete';
}

function buildSearchStatusSnapshot(status: SearchJobStatus) {
  return {
    status: status.status,
    currentStage: status.stage,
    stageMessage: status.message,
    progress: status.progress ?? null,
  } satisfies Pick<ActiveSearchRun, 'status' | 'currentStage' | 'stageMessage' | 'progress'>;
}

function searchStatusSnapshotSignature(snapshot: Pick<ActiveSearchRun, 'status' | 'currentStage' | 'stageMessage' | 'progress'>): string {
  return [snapshot.status, snapshot.currentStage ?? '', snapshot.stageMessage ?? '', progressSignature(snapshot.progress)].join('|');
}

function formatScopeChipLabel(corpusKey: string): string {
  if (corpusKey === 'chemqa500_simple/2026/all') {
    return 'ChemPaperSearch';
  }
  const [venue, year, track] = corpusKey.split('/');
  if (!venue || !year || !track) {
    return corpusKey;
  }
  return `${venue.toUpperCase()} ${year} ${track}`;
}

function summarizeWorkspaceScope(selectedCorpora: string[], catalog: CorpusCatalogEntry[]): string {
  const catalogKeys = new Set(catalog.map((entry) => entry.corpus_key));
  const available = selectedCorpora.filter((corpus) => catalogKeys.has(corpus));
  const unavailableCount = selectedCorpora.length - available.length;
  if (available.length === 0 && unavailableCount === 0) {
    return 'No corpus selected';
  }
  if (available.length === 1 && unavailableCount === 0) {
    return formatScopeChipLabel(available[0]!);
  }
  if (unavailableCount > 0) {
    return `${available.length} active • ${unavailableCount} unavailable`;
  }
  return `${available.length} corpora selected`;
}

function SearchResultRevealGrid({
  papers,
  traceId,
  onOpenPaper,
  onQuickPeek,
}: {
  papers: PaperResult[];
  traceId: string | null;
  onOpenPaper: (paper: PaperResult, traceId?: string | null) => void;
  onQuickPeek: (paper: PaperResult, traceId?: string | null) => void;
}) {
  const scope = useSequencedReveal([papers.map((paper) => paper.paper_id).join('|')]);

  return (
    <div ref={scope} className="grid grid-cols-1 gap-6 lg:grid-cols-2 2xl:grid-cols-3">
      {papers.map((paper, index) => (
        <div
          key={paper.paper_id}
          data-reveal-item
          className="search-result-reveal"
          style={{ animationDelay: `${Math.min(index, 5) * 70}ms` }}
        >
          <PaperResultCard
            paper={paper}
            onOpenPaper={(nextPaper) => onOpenPaper(nextPaper, traceId)}
            onQuickPeek={(nextPaper) => onQuickPeek(nextPaper, traceId)}
          />
        </div>
      ))}
    </div>
  );
}

function SearchTracePanel({
  message,
  onToggle,
}: {
  message: SearchMessage;
  onToggle: () => void;
}) {
  if (!message.traceId) {
    return null;
  }

  const isOpen = message.traceOpen ?? false;
  const tracePresence = usePresence(isOpen, 180);
  const trace = message.trace ?? null;
  const sourceSizes = (trace?.filter_summary?.source_sizes ?? {}) as Record<string, number>;
  const verifierSummary = (trace?.verifier_summary ?? {}) as Record<string, unknown>;
  const timingEntries = Object.entries(trace?.timings_ms ?? {}).filter(([, value]) => Number.isFinite(value)).slice(0, 6);

  return (
    <div className='mt-5 rounded-[1.6rem] border border-slate-200/90 bg-white/90 p-5 shadow-scholar-sm'>
      <button
        type='button'
        onClick={onToggle}
        className='flex w-full items-center justify-between gap-4 text-left'
      >
        <div>
          <div className='text-[0.68rem] font-bold uppercase tracking-[0.22em] text-slate-400'>Search trace</div>
          <div className='mt-1 text-[0.96rem] font-semibold tracking-tight text-slate-900'>
            Inspect how this result was planned, recalled, and verified.
          </div>
        </div>
        <div className='inline-flex items-center gap-2 rounded-full border border-slate-200 bg-slate-50 px-3 py-2 text-[0.68rem] font-bold uppercase tracking-[0.18em] text-slate-500'>
          {isOpen ? 'Hide' : 'Show'}
          {isOpen ? <ChevronUp className='h-3.5 w-3.5' /> : <ChevronDown className='h-3.5 w-3.5' />}
        </div>
      </button>

      {tracePresence.mounted ? (
        <div className='psa-collapse overflow-hidden' data-state={isOpen ? 'open' : 'closed'}>
          <div>
            {message.traceState === 'loading' ? (
              <div className='mt-4 flex items-center gap-2 text-[0.82rem] font-semibold uppercase tracking-[0.16em] text-slate-500'>
                <Loader2 className='h-3.5 w-3.5 animate-spin' />
                Loading trace details
              </div>
            ) : message.traceState === 'error' ? (
              <div className='mt-4 rounded-[1.2rem] border border-rose-200 bg-rose-50 px-4 py-3 text-[0.92rem] leading-7 text-rose-700'>
                {message.traceError ?? 'Trace details could not be loaded.'}
              </div>
            ) : trace ? (
              <div className='mt-5 grid gap-4 lg:grid-cols-2'>
                <div className='rounded-[1.3rem] border border-slate-200 bg-slate-50/80 p-4'>
                  <div className='text-[0.66rem] font-bold uppercase tracking-[0.22em] text-slate-400'>Planner</div>
                  {trace.workspace_scope.length > 0 ? (
                    <div className='mt-4'>
                      <div className='mb-2 text-[0.64rem] font-bold uppercase tracking-[0.18em] text-slate-400'>Workspace scope</div>
                      <div className='flex flex-wrap gap-2'>
                        {trace.workspace_scope.map((corpusKey) => (
                          <span key={`workspace-${corpusKey}`} className='rounded-full border border-slate-200 bg-white px-3 py-1 text-[0.72rem] font-semibold text-slate-600'>
                            {formatScopeChipLabel(corpusKey)}
                          </span>
                        ))}
                      </div>
                    </div>
                  ) : null}
                  {trace.effective_scope.length > 0 ? (
                    <div className='mt-4'>
                      <div className='mb-2 text-[0.64rem] font-bold uppercase tracking-[0.18em] text-slate-400'>Effective scope</div>
                      <div className='flex flex-wrap gap-2'>
                        {trace.effective_scope.map((corpusKey) => (
                          <span key={`effective-${corpusKey}`} className='rounded-full border border-indigo-100 bg-indigo-50 px-3 py-1 text-[0.72rem] font-semibold text-indigo-600'>
                            {formatScopeChipLabel(corpusKey)}
                          </span>
                        ))}
                      </div>
                    </div>
                  ) : null}
                  <div className='mt-3 flex flex-wrap gap-2'>
                    {trace.query_plan.scope_constraints.venues.map((venue) => (
                      <span key={`venue-${venue}`} className='rounded-full border border-indigo-100 bg-indigo-50 px-3 py-1 text-[0.72rem] font-semibold text-indigo-600'>
                        {venue.toUpperCase()}
                      </span>
                    ))}
                    {trace.query_plan.scope_constraints.years.map((year) => (
                      <span key={`year-${year}`} className='rounded-full border border-slate-200 bg-white px-3 py-1 text-[0.72rem] font-semibold text-slate-600'>
                        {year}
                      </span>
                    ))}
                    {(trace.query_plan.scope_constraints.tracks ?? []).map((track) => (
                      <span key={`track-${track}`} className='rounded-full border border-slate-200 bg-white px-3 py-1 text-[0.72rem] font-semibold text-slate-600'>
                        {track}
                      </span>
                    ))}
                    {trace.query_plan.entity_terms.slice(0, 6).map((term) => (
                      <span key={`entity-${term}`} className='rounded-full border border-slate-200 bg-white px-3 py-1 text-[0.72rem] font-semibold text-slate-600'>
                        {term}
                      </span>
                    ))}
                  </div>
                  {trace.query_plan.aspect_queries.length > 0 ? (
                    <div className='mt-4 space-y-2'>
                      {trace.query_plan.aspect_queries.slice(0, 4).map((aspect) => (
                        <div key={aspect.aspect_id} className='rounded-[1rem] border border-white/90 bg-white px-3 py-2 text-[0.84rem] leading-6 text-slate-600'>
                          {aspect.query}
                        </div>
                      ))}
                    </div>
                  ) : null}
                </div>

                <div className='rounded-[1.3rem] border border-slate-200 bg-slate-50/80 p-4'>
                  <div className='text-[0.66rem] font-bold uppercase tracking-[0.22em] text-slate-400'>Recall</div>
                  <div className='mt-4 grid grid-cols-2 gap-3'>
                    {Object.entries(sourceSizes).length > 0 ? Object.entries(sourceSizes).map(([source, count]) => (
                      <div key={source} className='rounded-[1rem] border border-white/90 bg-white px-3 py-3'>
                        <div className='text-[0.64rem] font-bold uppercase tracking-[0.18em] text-slate-400'>{titleCaseToken(source)}</div>
                        <div className='mt-2 text-[1.05rem] font-black tracking-tight text-slate-900'>{compactInteger(count)}</div>
                      </div>
                    )) : (
                      <div className='col-span-2 rounded-[1rem] border border-white/90 bg-white px-3 py-3 text-[0.9rem] leading-6 text-slate-500'>
                        Recall source statistics are not available for this run.
                      </div>
                    )}
                  </div>
                </div>

                <div className='rounded-[1.3rem] border border-slate-200 bg-slate-50/80 p-4'>
                  <div className='text-[0.66rem] font-bold uppercase tracking-[0.22em] text-slate-400'>Verifier</div>
                  <div className='mt-4 grid grid-cols-2 gap-3'>
                    {[
                      { label: 'Candidate pool', value: verifierSummary.candidate_pool_count },
                      { label: 'Satisfied', value: verifierSummary.satisfied_count },
                      { label: 'Partial', value: verifierSummary.partial_count },
                      { label: 'Rejected', value: verifierSummary.rejected_count },
                    ].map(({ label, value }) => (
                      <div key={label} className='rounded-[1rem] border border-white/90 bg-white px-3 py-3'>
                        <div className='text-[0.64rem] font-bold uppercase tracking-[0.18em] text-slate-400'>{label}</div>
                        <div className='mt-2 text-[1.05rem] font-black tracking-tight text-slate-900'>
                          {typeof value === 'number' ? compactInteger(value) : '—'}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className='rounded-[1.3rem] border border-slate-200 bg-slate-50/80 p-4'>
                  <div className='text-[0.66rem] font-bold uppercase tracking-[0.22em] text-slate-400'>Timing</div>
                  <div className='mt-4 space-y-2'>
                    {timingEntries.length > 0 ? timingEntries.map(([key, value]) => (
                      <div key={key} className='flex items-center justify-between rounded-[1rem] border border-white/90 bg-white px-3 py-2.5 text-[0.84rem] text-slate-600'>
                        <span className='font-semibold'>{titleCaseToken(key)}</span>
                        <span className='font-black tracking-tight text-slate-900'>{formatTimingMs(value)}</span>
                      </div>
                    )) : (
                      <div className='rounded-[1rem] border border-white/90 bg-white px-3 py-3 text-[0.9rem] leading-6 text-slate-500'>
                        Timing statistics are not available for this run.
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ) : (
              <div className='mt-4 text-[0.9rem] leading-7 text-slate-500'>Open this panel to load the structured search trace.</div>
            )}
          </div>
        </div>
      ) : null}
    </div>
  );
}

function WorkspaceConfirmDialog({
  open,
  title,
  description,
  confirmLabel,
  tone,
  busy,
  onCancel,
  onConfirm,
}: {
  open: boolean;
  title: string;
  description: string;
  confirmLabel: string;
  tone: 'warning' | 'danger';
  busy: boolean;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  const { mounted, phase } = usePresence(open, 180);
  const accentClassName =
    tone === 'danger'
      ? 'border-rose-200 bg-rose-50 text-rose-600'
      : 'border-amber-200 bg-amber-50 text-amber-600';
  const confirmClassName =
    tone === 'danger'
      ? 'bg-rose-600 hover:bg-rose-700 focus-visible:ring-rose-300'
      : 'bg-slate-950 hover:bg-indigo-600 focus-visible:ring-indigo-300';

  if (!mounted) {
    return null;
  }

  return (
    <div className="fixed inset-0 z-[120] flex items-center justify-center p-4 sm:p-6">
      <div
        data-state={phase}
        onClick={busy ? undefined : onCancel}
        className="psa-overlay-backdrop absolute inset-0 bg-slate-950/34"
      />
      <section
        data-state={phase}
        className="psa-modal-surface relative z-10 w-full max-w-[480px] overflow-hidden rounded-[2rem] border border-white/80 bg-[linear-gradient(180deg,rgba(255,255,255,0.98),rgba(248,250,252,0.98))] shadow-[0_30px_90px_rgba(15,23,42,0.24)]"
      >
            <div className="border-b border-slate-200/70 px-7 py-6">
              <div className="flex items-start gap-4">
                <div className={`inline-flex h-12 w-12 shrink-0 items-center justify-center rounded-[1rem] border ${accentClassName}`}>
                  <AlertTriangle className="h-5 w-5" />
                </div>
                <div className="min-w-0">
                  <div className="text-[0.66rem] font-black uppercase tracking-[0.22em] text-slate-400">Workspace action</div>
                  <h3 className="mt-2 text-[1.15rem] font-black tracking-tight text-slate-950">{title}</h3>
                  <p className="mt-2 text-[0.92rem] leading-7 text-slate-500">{description}</p>
                </div>
              </div>
            </div>
            <div className="flex items-center justify-end gap-3 bg-white/72 px-7 py-5">
              <button
                type="button"
                disabled={busy}
                onClick={onCancel}
                className="inline-flex items-center justify-center rounded-full border border-slate-200 bg-white px-4 py-2.5 text-[0.74rem] font-bold uppercase tracking-[0.18em] text-slate-600 transition hover:border-slate-300 hover:text-slate-900 disabled:cursor-not-allowed disabled:opacity-60"
              >
                Cancel
              </button>
              <button
                type="button"
                disabled={busy}
                onClick={onConfirm}
                className={`inline-flex min-w-[10rem] items-center justify-center rounded-full px-4 py-2.5 text-[0.74rem] font-bold uppercase tracking-[0.18em] text-white transition focus-visible:outline-none focus-visible:ring-4 disabled:cursor-not-allowed disabled:opacity-60 ${confirmClassName}`}
              >
                {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : confirmLabel}
              </button>
            </div>
      </section>
    </div>
  );
}

export default function ChatPage() {
  const [messages, setMessages] = useState<SearchMessage[]>([]);
  const [activeSearchRun, setActiveSearchRun] = useState<ActiveSearchRun | null>(null);
  const [input, setInput] = useState('');
  const [isSearching, setIsSearching] = useState(false);
  const [selectedPaper, setSelectedPaper] = useState<PaperResult | null>(null);
  const [previewPaper, setPreviewPaper] = useState<{ paper: PaperResult; threadId: string | null } | null>(null);
  const [isComposerFocused, setIsComposerFocused] = useState(false);
  const [isPaletteOpen, setIsPaletteOpen] = useState(false);
  const [paletteInput, setPaletteInput] = useState('');
  const [chatSessions, setChatSessions] = useState<Record<string, ChatMessage[]>>({});
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [currentProjectId, setCurrentProjectId] = useState<string | null>(null);
  const [projectDetail, setProjectDetail] = useState<ProjectDetailResponse | null>(null);
  const [projectState, setProjectState] = useState<'loading' | 'ready' | 'error'>('ready');
  const [projectError, setProjectError] = useState<string | null>(null);
  const [corpusCatalog, setCorpusCatalog] = useState<CorpusCatalogEntry[]>([]);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [isProjectPanelOpen, setIsProjectPanelOpen] = useState(false);
  const [isProjectMutating, setIsProjectMutating] = useState(false);
  const [isScopeSaving, setIsScopeSaving] = useState(false);
  const [activeThreadId, setActiveThreadId] = useState<string | null>(null);
  const [workspaceActionIntent, setWorkspaceActionIntent] = useState<WorkspaceActionIntent | null>(null);

  const searchScrollRef = useRef<HTMLElement | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const previousMessageCountRef = useRef(0);
  const pendingSearchScrollTopRef = useRef<number | null>(null);
  const composerTextareaRef = useRef<HTMLTextAreaElement | null>(null);
  const composerRegionRef = useRef<HTMLDivElement | null>(null);
  const searchInFlightRef = useRef(false);
  const projectLoadRequestRef = useRef(0);
  const currentProjectIdRef = useRef<string | null>(null);
  const paperSessionPersistTimersRef = useRef<Record<string, number>>({});
  const lastPaperSessionPersistSignatureRef = useRef<Record<string, string>>({});
  const activeSearchSnapshotRef = useRef<string>('none');
  const activeSearchUpdatedAtRef = useRef(0);
  const hasSearchHistory = messages.length > 0;
  const hasThreadView = hasSearchHistory || activeSearchRun != null;
  const isRunningView = activeSearchRun != null;
  const currentProjectTitle = projectDetail?.project.title ?? projects.find((project) => project.project_id === currentProjectId)?.title ?? null;
  const currentSelectedCorpora = projectDetail?.project.selected_corpora ?? [];
  const currentCatalogKeySet = useMemo(() => new Set(corpusCatalog.map((entry) => entry.corpus_key)), [corpusCatalog]);
  const unavailableSelectedCorpora = useMemo(
    () => currentSelectedCorpora.filter((corpus) => !currentCatalogKeySet.has(corpus)),
    [currentCatalogKeySet, currentSelectedCorpora],
  );
  const scopeSummary = useMemo(
    () => summarizeWorkspaceScope(currentSelectedCorpora, corpusCatalog),
    [corpusCatalog, currentSelectedCorpora],
  );
  const searchBlockedReason = useMemo(() => {
    if (!currentProjectId) {
      return null;
    }
    if (currentSelectedCorpora.length === 0) {
      return 'Select at least one corpus in this workspace before searching.';
    }
    if (unavailableSelectedCorpora.length > 0) {
      return 'Remove unavailable corpora from this workspace before searching.';
    }
    return null;
  }, [currentProjectId, currentSelectedCorpora.length, unavailableSelectedCorpora.length]);
  const hasWorkspaces = projects.length > 0;
  const isProjectBusy = projectState === 'loading' || isProjectMutating;
  const headerActivityMode = useTransientActivityMode(isSearching);
  const nextDefaultWorkspaceTitle = useMemo(() => {
    const normalizedTitles = new Set(projects.map((project) => project.title.trim().toLowerCase()));
    const baseTitle = 'Untitled workspace';
    if (!normalizedTitles.has(baseTitle.toLowerCase())) {
      return baseTitle;
    }
    let index = 2;
    while (normalizedTitles.has(`${baseTitle} ${index}`.toLowerCase())) {
      index += 1;
    }
    return `${baseTitle} ${index}`;
  }, [projects]);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, []);

  useEffect(() => {
    currentProjectIdRef.current = currentProjectId;
  }, [currentProjectId]);

  useEffect(() => {
    return () => {
      for (const timerId of Object.values(paperSessionPersistTimersRef.current)) {
        window.clearTimeout(timerId);
      }
      paperSessionPersistTimersRef.current = {};
    };
  }, []);

  useEffect(() => {
    if (selectedPaper) {
      return;
    }

    const pendingScrollTop = pendingSearchScrollTopRef.current;
    if (pendingScrollTop !== null) {
      const frameId = window.requestAnimationFrame(() => {
        searchScrollRef.current?.scrollTo({ top: pendingScrollTop, behavior: 'auto' });
        pendingSearchScrollTopRef.current = null;
        previousMessageCountRef.current = messages.length;
      });
      return () => window.cancelAnimationFrame(frameId);
    }

    if (messages.length > previousMessageCountRef.current && hasSearchHistory) {
      scrollToBottom();
    }
    previousMessageCountRef.current = messages.length;
  }, [hasSearchHistory, messages.length, scrollToBottom, selectedPaper]);

  const openSearchPalette = useCallback((prefill?: string) => {
    const nextValue = prefill ?? input.trim() ?? '';
    setPaletteInput(nextValue);
    runViewTransition(() => {
      setPreviewPaper(null);
      setIsComposerFocused(false);
      setIsPaletteOpen(true);
    });
  }, [input]);

  const closeSearchPalette = useCallback(() => {
    runViewTransition(() => {
      setIsPaletteOpen(false);
    });
  }, []);

  const syncProjectDetailState = useCallback((detail: ProjectDetailResponse) => {
    setProjectDetail(detail);
    setChatSessions(buildChatSessionsMap(detail.paper_sessions));
  }, []);

  const refreshProjectContext = useCallback(async (projectId: string) => {
    const [projectList, detail] = await Promise.all([listProjects(), fetchProject(projectId)]);
    setProjects(projectList.projects);
    syncProjectDetailState(detail);
    return detail;
  }, [syncProjectDetailState]);

  const loadCorpusCatalog = useCallback(async () => {
    try {
      const response = await fetchCorpusCatalog();
      setCorpusCatalog(response.corpora);
      setCatalogError(null);
    } catch (error) {
      setCatalogError(error instanceof Error ? error.message : 'Corpus catalog could not be loaded.');
    }
  }, []);

  const restoreProjectThread = useCallback(
    async (thread: ProjectSearchThread, options: { openPaperId?: string | null; closePanel?: boolean } = {}) => {
      let trace: SearchTrace | null = null;
      if (thread.trace_id) {
        try {
          trace = await fetchTrace(thread.trace_id);
        } catch {
          trace = null;
        }
      }

      const restoredMessages = buildRestoredSearchMessages(thread, trace);
      const displayResults = restoredMessages[1]?.results ?? [];

      pendingSearchScrollTopRef.current = 0;
      runViewTransition(() => {
        setMessages(restoredMessages);
        setPreviewPaper(null);
        setSelectedPaper(null);
        setActiveThreadId(thread.thread_id);
      });

      if (options.openPaperId) {
        const targetPaper = displayResults.find((paper) => paper.paper_id === options.openPaperId) ?? null;
        if (targetPaper) {
          runViewTransition(() => {
            setSelectedPaper(targetPaper);
          });
        }
      }

      if (options.closePanel !== false) {
        setIsProjectPanelOpen(false);
      }
    },
    [],
  );

  const loadProjectWorkspace = useCallback(
    async (requestedProjectId: string | null, options: { restoreLatestThread: boolean } = { restoreLatestThread: true }) => {
      const requestId = projectLoadRequestRef.current + 1;
      projectLoadRequestRef.current = requestId;
      setProjectState('loading');
      setProjectError(null);

      try {
        const projectList = await listProjects();
        if (projectLoadRequestRef.current !== requestId) {
          return;
        }

        const availableProjects = projectList.projects;
        if (availableProjects.length === 0) {
          setProjects([]);
          setCurrentProjectId(null);
          setProjectDetail(null);
          clearCurrentProjectId();
          setMessages([]);
          setPreviewPaper(null);
          setSelectedPaper(null);
          setActiveThreadId(null);
          setProjectState('ready');
          setProjectError(null);
          return;
        }

        const fallbackProjectId = availableProjects[0]!.project_id;
        const initialProjectId = requestedProjectId ?? loadCurrentProjectId() ?? fallbackProjectId;
        const resolvedProjectId = availableProjects.some((project) => project.project_id === initialProjectId)
          ? initialProjectId
          : fallbackProjectId;

        const detail = await fetchProject(resolvedProjectId);
        if (projectLoadRequestRef.current !== requestId) {
          return;
        }

        setProjects(availableProjects);
        setCurrentProjectId(resolvedProjectId);
        saveCurrentProjectId(resolvedProjectId);
        syncProjectDetailState(detail);
        setProjectState('ready');
        setProjectError(null);

        if (options.restoreLatestThread && detail.threads.length > 0) {
          await restoreProjectThread(detail.threads[0]!, { closePanel: false });
        } else {
          setMessages([]);
          setPreviewPaper(null);
          setSelectedPaper(null);
          setActiveThreadId(null);
        }
      } catch (error) {
        if (projectLoadRequestRef.current !== requestId) {
          return;
        }
        setProjectState('error');
        setProjectError(error instanceof Error ? error.message : 'Workspace could not be loaded.');
      }
    },
    [restoreProjectThread, syncProjectDetailState],
  );

  const persistProjectThread = useCallback(
    async (projectId: string, thread: ProjectSearchThread) => {
      try {
        await upsertProjectThread(projectId, thread.thread_id, {
          query: thread.query,
          trace_id: thread.trace_id,
          result_counts: thread.result_counts,
          paper_ids: thread.paper_ids,
          workspace_scope: thread.workspace_scope,
          query_scope: thread.query_scope,
          effective_scope: thread.effective_scope,
        });
        if (currentProjectIdRef.current === projectId) {
          await refreshProjectContext(projectId);
        }
      } catch {}
    },
    [refreshProjectContext],
  );

  const persistProjectPaperSession = useCallback(
    async (projectId: string, session: { paperId: string; paperTitle: string | null; sourceThreadId: string | null; chatHistory: ChatMessage[] }) => {
      try {
        const persisted = await upsertProjectPaperSession(projectId, session.paperId, {
          paper_title: session.paperTitle,
          source_thread_id: session.sourceThreadId,
          chat_history: serializeProjectChatHistory(session.chatHistory),
          last_active_evidence_id: null,
        });
        if (currentProjectIdRef.current === projectId) {
          setProjectDetail((previous) => {
            if (!previous || previous.project.project_id !== projectId) {
              return previous;
            }

            const nextSessions = [
              persisted,
              ...previous.paper_sessions.filter((item) => item.paper_id !== persisted.paper_id),
            ].sort((left, right) => right.updated_at.localeCompare(left.updated_at));

            return {
              ...previous,
              project: {
                ...previous.project,
                updated_at: persisted.updated_at,
                paper_session_count: nextSessions.length,
              },
              paper_sessions: nextSessions,
            };
          });
          setProjects((previous) =>
            previous.map((project) =>
              project.project_id === projectId
                ? {
                    ...project,
                    updated_at: persisted.updated_at,
                    paper_session_count: Math.max(project.paper_session_count, 1),
                  }
                : project,
            ),
          );
        }
      } catch {}
    },
    [],
  );

  const handleSelectProject = useCallback(async (projectId: string) => {
    if (isSearching) {
      return;
    }
    await loadProjectWorkspace(projectId, { restoreLatestThread: true });
    setIsProjectPanelOpen(false);
  }, [isSearching, loadProjectWorkspace]);

  const handleCreateProject = useCallback(async () => {
    if (isProjectBusy) {
      return;
    }
    setIsProjectMutating(true);
    try {
      const created = await createProject({ title: nextDefaultWorkspaceTitle });
      await loadProjectWorkspace(created.project_id, { restoreLatestThread: false });
      setIsProjectPanelOpen(false);
    } catch (error) {
      setProjectError(error instanceof Error ? error.message : 'Workspace could not be created.');
      setProjectState('error');
    } finally {
      setIsProjectMutating(false);
    }
  }, [isProjectBusy, loadProjectWorkspace, nextDefaultWorkspaceTitle]);

  const ensureWorkspaceForSearch = useCallback(async (): Promise<string> => {
    if (currentProjectId) {
      return currentProjectId;
    }

    const created = await createProject({ title: nextDefaultWorkspaceTitle });
    const detail = await refreshProjectContext(created.project_id);
    setCurrentProjectId(created.project_id);
    saveCurrentProjectId(created.project_id);
    syncProjectDetailState(detail);
    return created.project_id;
  }, [currentProjectId, nextDefaultWorkspaceTitle, refreshProjectContext, syncProjectDetailState]);

  const handleRenameProject = useCallback(async (projectId: string, title: string) => {
    if (!title.trim() || isProjectBusy) {
      return;
    }
    setIsProjectMutating(true);
    try {
      const renamed = await updateProject(projectId, { title: title.trim() });
      const projectList = await listProjects();
      setProjects(projectList.projects);
      setProjectDetail((previous) => {
        if (!previous || previous.project.project_id !== projectId) {
          return previous;
        }
        return {
          ...previous,
          project: renamed,
        };
      });
    } catch (error) {
      setProjectError(error instanceof Error ? error.message : 'Workspace could not be renamed.');
      setProjectState('error');
    } finally {
      setIsProjectMutating(false);
    }
  }, [isProjectBusy]);

  const handleUpdateProjectScope = useCallback(async (projectId: string, selectedCorpora: string[]) => {
    if (isProjectBusy || isScopeSaving) {
      return;
    }
    setIsScopeSaving(true);
    try {
      const updated = await updateProject(projectId, { selected_corpora: selectedCorpora });
      const projectList = await listProjects();
      setProjects(projectList.projects);
      setProjectDetail((previous) => {
        if (!previous || previous.project.project_id !== projectId) {
          return previous;
        }
        return {
          ...previous,
          project: updated,
        };
      });
      setProjectError(null);
      setProjectState('ready');
    } catch (error) {
      setProjectError(error instanceof Error ? error.message : 'Workspace scope could not be updated.');
      setProjectState('error');
    } finally {
      setIsScopeSaving(false);
    }
  }, [isProjectBusy, isScopeSaving]);

  const executeClearProject = useCallback(async () => {
    if (!currentProjectId || isProjectBusy) {
      return;
    }
    setIsProjectMutating(true);
    try {
      await clearProject(currentProjectId);
      await loadProjectWorkspace(currentProjectId, { restoreLatestThread: false });
      setIsProjectPanelOpen(false);
    } catch (error) {
      setProjectError(error instanceof Error ? error.message : 'Workspace could not be cleared.');
      setProjectState('error');
    } finally {
      setIsProjectMutating(false);
    }
  }, [currentProjectId, isProjectBusy, loadProjectWorkspace]);

  const executeDeleteProject = useCallback(async () => {
    if (!currentProjectId || isProjectBusy) {
      return;
    }
    setIsProjectMutating(true);
    try {
      await deleteProject(currentProjectId);
      clearCurrentProjectId();
      setCurrentProjectId(null);
      setProjectDetail(null);
      await loadProjectWorkspace(null, { restoreLatestThread: true });
      setIsProjectPanelOpen(false);
    } catch (error) {
      setProjectError(error instanceof Error ? error.message : 'Workspace could not be deleted.');
      setProjectState('error');
    } finally {
      setIsProjectMutating(false);
    }
  }, [currentProjectId, isProjectBusy, loadProjectWorkspace]);

  const handleClearProject = useCallback(() => {
    if (!currentProjectId || isProjectBusy) {
      return;
    }
    setWorkspaceActionIntent('clear');
  }, [currentProjectId, isProjectBusy]);

  const handleDeleteProject = useCallback(() => {
    if (!currentProjectId || isProjectBusy) {
      return;
    }
    setWorkspaceActionIntent('delete');
  }, [currentProjectId, isProjectBusy]);

  const handleCancelWorkspaceAction = useCallback(() => {
    if (isProjectMutating) {
      return;
    }
    setWorkspaceActionIntent(null);
  }, [isProjectMutating]);

  const handleConfirmWorkspaceAction = useCallback(async () => {
    if (!workspaceActionIntent) {
      return;
    }
    if (workspaceActionIntent === 'clear') {
      await executeClearProject();
    } else {
      await executeDeleteProject();
    }
    setWorkspaceActionIntent(null);
  }, [executeClearProject, executeDeleteProject, workspaceActionIntent]);

  const handleRestoreSavedThread = useCallback(async (thread: ProjectSearchThread) => {
    await restoreProjectThread(thread);
  }, [restoreProjectThread]);

  const handleOpenSavedPaperSession = useCallback(
    async (session: ProjectPaperSession) => {
      if (!projectDetail || !session.source_thread_id) {
        return;
      }
      const sourceThread = projectDetail.threads.find((thread) => thread.thread_id === session.source_thread_id);
      if (!sourceThread) {
        return;
      }
      await restoreProjectThread(sourceThread, { openPaperId: session.paper_id });
    },
    [projectDetail, restoreProjectThread],
  );

  useEffect(() => {
    void loadProjectWorkspace(loadCurrentProjectId(), { restoreLatestThread: true });
  }, [loadProjectWorkspace]);

  useEffect(() => {
    void loadCorpusCatalog();
  }, [loadCorpusCatalog]);

  const handleSearch = useCallback(async (query: string) => {
    const normalizedQuery = query.trim();
    if (!normalizedQuery || searchInFlightRef.current || projectState === 'loading' || searchBlockedReason) {
      return;
    }
    searchInFlightRef.current = true;

    const assistantId = `${Date.now()}-assistant`;
    const searchStartedAt = Date.now();
    setActiveSearchRun({
      assistantId,
      query: normalizedQuery,
      status: 'running',
      currentStage: 'queued',
      stageMessage: 'Submitting the search job.',
      progress: null,
      startedAt: searchStartedAt,
      stageStartedAt: searchStartedAt,
    });
    activeSearchSnapshotRef.current = 'running|queued|Submitting the search job.|none';
    activeSearchUpdatedAtRef.current = Date.now();
    setInput('');
    setIsSearching(true);
    setPreviewPaper(null);

    try {
      const targetProjectId = await ensureWorkspaceForSearch();
      const job = await createSearchJob({ project_id: targetProjectId, query: normalizedQuery, top_k: 15, display_k: 10 });
      let lastStatus: SearchJobStatus = job;
      const applyActiveSearchStatus = (status: SearchJobStatus, force = false) => {
        const snapshot = buildSearchStatusSnapshot(status);
        const nextSignature = searchStatusSnapshotSignature(snapshot);
        const now = Date.now();
        const shouldCommit =
          force ||
          nextSignature !== activeSearchSnapshotRef.current ||
          now - activeSearchUpdatedAtRef.current >= 900;

        if (!shouldCommit) {
          return;
        }

        activeSearchSnapshotRef.current = nextSignature;
        activeSearchUpdatedAtRef.current = now;
        setActiveSearchRun((previous) => {
          if (!previous || previous.assistantId !== assistantId) {
            return previous;
          }
          const stageChanged = previous.currentStage !== snapshot.currentStage;
          return {
            ...previous,
            ...snapshot,
            stageStartedAt: force || stageChanged ? now : previous.stageStartedAt,
          };
        });
      };

      applyActiveSearchStatus(job, true);

      while (lastStatus.status !== 'completed' && lastStatus.status !== 'failed') {
        await new Promise((resolve) => setTimeout(resolve, 1400));
        lastStatus = await fetchSearchJob(job.job_id);
        applyActiveSearchStatus(lastStatus);
      }

      if (lastStatus.status === 'completed') {
        const result = await fetchSearchJobResult(job.job_id);
        const displayPapers = result.display_results;
        const threadRecord: ProjectSearchThread = {
          project_id: targetProjectId,
          thread_id: result.trace_id,
          query: normalizedQuery,
          trace_id: result.trace_id,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
          result_counts: result.counts,
          paper_ids: displayPapers.map((paper) => paper.paper_id),
          workspace_scope: result.workspace_scope,
          query_scope: result.query_scope,
          effective_scope: result.effective_scope,
        };

        const userMessage: SearchMessage = {
          id: `${Date.now()}-user`,
          role: 'user',
          content: normalizedQuery,
        };
        const resultMessage: SearchMessage = {
          id: assistantId,
          role: 'assistant',
          content: `Search complete. ${displayPapers.length} papers are ready for review.`,
          status: 'completed',
          type: 'search_results',
          results: displayPapers,
          progress: lastStatus.progress ?? null,
          traceId: result.trace_id,
          trace: null,
          traceState: 'idle',
          traceError: null,
          traceOpen: false,
        };

        setActiveSearchRun((previous) =>
          previous && previous.assistantId === assistantId
            ? {
                ...previous,
                status: 'completed',
                currentStage: 'completed',
                stageMessage: 'Preparing the result view.',
                progress: {
                  stage_index: lastStatus.progress?.stage_total ?? lastStatus.progress?.stage_index ?? SEARCH_STAGE_SEQUENCE.length,
                  stage_total: lastStatus.progress?.stage_total ?? SEARCH_STAGE_SEQUENCE.length,
                  stage_progress: 1,
                  overall_progress: 1,
                  completed_items: lastStatus.progress?.completed_items ?? null,
                  total_items: lastStatus.progress?.total_items ?? null,
                },
                stageStartedAt: Date.now(),
              }
            : previous,
        );
        await new Promise((resolve) => window.setTimeout(resolve, 420));
        setActiveSearchRun(null);
        setMessages((previous) => [...previous, userMessage, resultMessage]);
        setActiveThreadId(result.trace_id);
        void persistProjectThread(targetProjectId, threadRecord);
        return;
      }

      throw new Error('Search failed.');
    } catch {
      const userMessage: SearchMessage = {
        id: `${Date.now()}-user`,
        role: 'user',
        content: normalizedQuery,
      };
      const errorMessage: SearchMessage = {
        id: assistantId,
        role: 'assistant',
        content: 'The search did not complete successfully.',
        status: 'error',
        progress: null,
      };
      setActiveSearchRun(null);
      setMessages((previous) => [...previous, userMessage, errorMessage]);
    } finally {
      searchInFlightRef.current = false;
      setIsSearching(false);
      activeSearchSnapshotRef.current = 'none';
    }
  }, [ensureWorkspaceForSearch, persistProjectThread, projectState, searchBlockedReason]);

  const handleToggleTrace = useCallback(async (messageId: string) => {
    const message = messages.find((item) => item.id === messageId);
    if (!message?.traceId) {
      return;
    }

    const nextOpenState = !(message.traceOpen ?? false);
    setMessages((previous) =>
      previous.map((item) =>
        item.id === messageId
          ? {
              ...item,
              traceOpen: nextOpenState,
            }
          : item,
      ),
    );

    if (!nextOpenState || message.traceState === 'loaded' || message.traceState === 'loading') {
      return;
    }

    setMessages((previous) =>
      previous.map((item) =>
        item.id === messageId
          ? {
              ...item,
              traceState: 'loading',
              traceError: null,
            }
          : item,
      ),
    );

    try {
      const trace = await fetchTrace(message.traceId);
      setMessages((previous) =>
        previous.map((item) =>
          item.id === messageId
            ? {
                ...item,
                trace,
                traceState: 'loaded',
                traceError: null,
              }
            : item,
        ),
      );
    } catch (error) {
      setMessages((previous) =>
        previous.map((item) =>
          item.id === messageId
            ? {
                ...item,
                traceState: 'error',
                traceError: error instanceof Error ? error.message : 'Trace details could not be loaded.',
              }
            : item,
        ),
      );
    }
  }, [messages]);

  const handleGlobalSearchSubmit = useCallback((query: string) => {
    const normalizedQuery = query.trim();
    if (!normalizedQuery || isSearching) {
      return;
    }

    runViewTransition(() => {
      setIsPaletteOpen(false);
      setPaletteInput('');
      setPreviewPaper(null);
      if (selectedPaper) {
        setSelectedPaper(null);
      }
    });
    void handleSearch(normalizedQuery);
  }, [handleSearch, isSearching, selectedPaper]);

  const updateChatSession = useCallback((paperId: string, nextMessages: ChatMessage[]) => {
    setChatSessions((previous) => ({ ...previous, [paperId]: nextMessages }));
    if (currentProjectId && selectedPaper) {
      const sessionPayload = {
        paperId,
        paperTitle: selectedPaper.title,
        sourceThreadId: activeThreadId,
        chatHistory: nextMessages,
      };
      const signature = buildProjectSessionSignature(sessionPayload);
      if (lastPaperSessionPersistSignatureRef.current[paperId] === signature) {
        return;
      }

      const existingTimerId = paperSessionPersistTimersRef.current[paperId];
      if (existingTimerId != null) {
        window.clearTimeout(existingTimerId);
      }

      paperSessionPersistTimersRef.current[paperId] = window.setTimeout(() => {
        lastPaperSessionPersistSignatureRef.current[paperId] = signature;
        void persistProjectPaperSession(currentProjectId, sessionPayload);
        delete paperSessionPersistTimersRef.current[paperId];
      }, 450);
    }
  }, [activeThreadId, currentProjectId, persistProjectPaperSession, selectedPaper]);

  const handleQuickPeek = useCallback((paper: PaperResult, threadId?: string | null) => {
    runViewTransition(() => {
      setPreviewPaper({ paper, threadId: threadId ?? activeThreadId ?? null });
    });
  }, [activeThreadId]);

  const handleOpenPaper = useCallback((paper: PaperResult, threadId?: string | null) => {
    pendingSearchScrollTopRef.current = searchScrollRef.current?.scrollTop ?? 0;
    runViewTransition(() => {
      setPreviewPaper(null);
      if (threadId) {
        setActiveThreadId(threadId);
      }
      setSelectedPaper(paper);
    });
  }, []);

  const handleBackFromPaper = useCallback(() => {
    runViewTransition(() => {
      setSelectedPaper(null);
    });
  }, []);

  useEffect(() => {
    const handleWindowKeydown = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
        event.preventDefault();
        openSearchPalette();
        return;
      }

      if (event.key === 'Escape' && isPaletteOpen) {
        event.preventDefault();
        closeSearchPalette();
      }
    };

    window.addEventListener('keydown', handleWindowKeydown);
    return () => {
      window.removeEventListener('keydown', handleWindowKeydown);
    };
  }, [closeSearchPalette, isPaletteOpen, openSearchPalette]);

  const renderQueryForm = useCallback(
    (variant: 'hero' | 'dock') => {
      const isHero = variant === 'hero';
      const showSuggestions = isComposerFocused && !isSearching;

      return (
        <div
          ref={composerRegionRef}
          onFocusCapture={() => setIsComposerFocused(true)}
          onBlurCapture={(event) => {
            const nextTarget = event.relatedTarget;
            if (nextTarget instanceof Node && composerRegionRef.current?.contains(nextTarget)) {
              return;
            }
            setIsComposerFocused(false);
          }}
          className='w-full space-y-4'
        >
          <form
            onSubmit={(event) => {
              event.preventDefault();
              void handleSearch(input);
            }}
            className={`w-full transition-all duration-300 ${isComposerFocused ? 'relative z-30' : ''}`}
          >
            <div
              className={`relative transition-all duration-300 ${
                isHero
                  ? 'rounded-[2.1rem] border border-white/80 bg-white/84 shadow-scholar-lg backdrop-blur-xl'
                  : 'rounded-[2rem] border border-white/84 bg-white/92 shadow-scholar-lg backdrop-blur-xl'
              } ${
                isComposerFocused
                  ? 'border-indigo-200/90 bg-white shadow-[0_30px_70px_rgba(15,23,42,0.12),0_10px_24px_rgba(79,70,229,0.08)]'
                  : ''
              }`}
            >
              <textarea
                ref={composerTextareaRef}
                value={input}
                onChange={(event) => setInput(event.target.value)}
                disabled={isSearching || searchBlockedReason != null}
                rows={isHero ? 4 : 2}
                placeholder='Ask for papers, datasets, methods, or evidence.'
                onKeyDown={(event) => {
                  if (event.nativeEvent.isComposing) {
                    return;
                  }
                  if (event.key === 'Enter' && !event.shiftKey) {
                    event.preventDefault();
                    void handleSearch(input);
                  }
                }}
                className={`w-full resize-none rounded-[2rem] border-0 bg-transparent pr-20 text-slate-900 outline-none placeholder:text-slate-400 ${
                  isHero
                    ? 'min-h-[10.5rem] px-7 py-7 text-[1.04rem] leading-8 sm:text-[1.1rem]'
                    : 'min-h-[6.9rem] px-6 py-5 text-[0.98rem] leading-7 sm:text-[1.02rem]'
                }`}
              />
              <button
                type='submit'
                disabled={isSearching || input.trim().length === 0 || searchBlockedReason != null}
                className={`absolute right-3 top-1/2 inline-flex -translate-y-1/2 items-center justify-center rounded-full bg-slate-950 text-white transition-all hover:bg-indigo-600 active:scale-95 disabled:cursor-not-allowed disabled:bg-slate-300 ${
                  isHero ? 'h-12 w-12' : 'h-11 w-11'
                }`}
              >
                {isSearching ? <Loader2 className='h-5 w-5 animate-spin' /> : <Search className='h-5 w-5' />}
              </button>

              {isHero ? (
                <div className='pointer-events-none absolute inset-x-6 bottom-4 flex items-center justify-between text-[0.72rem] font-medium text-slate-400'>
                  <span>Shift + Enter for a new line</span>
                  <span>Evidence-aware search</span>
                </div>
              ) : null}
            </div>
          </form>

          {searchBlockedReason ? (
            <div className="rounded-[1.25rem] border border-amber-200 bg-amber-50 px-4 py-3 text-[0.84rem] leading-6 text-amber-700">
              {searchBlockedReason}
            </div>
          ) : null}

          {!searchBlockedReason && catalogError ? (
            <div className="rounded-[1.25rem] border border-rose-200 bg-rose-50 px-4 py-3 text-[0.84rem] leading-6 text-rose-700">
              {catalogError}
            </div>
          ) : null}

          {showSuggestions ? (
            <div className={isHero ? 'relative z-30 psa-fade-up-enter' : 'relative z-30 psa-fade-up-enter'}>
              <SearchSuggestionPanel
                variant={variant}
                onSelect={(query) => {
                  setInput(query);
                  setIsComposerFocused(false);
                  void handleSearch(query);
                }}
              />
            </div>
          ) : null}
        </div>
      );
    },
    [catalogError, handleSearch, input, isComposerFocused, isSearching, searchBlockedReason],
  );

  if (projectState === 'loading' && currentProjectId == null && !projectDetail) {
    return (
      <div className="flex min-h-dvh items-center justify-center bg-[#f8fafc] px-6">
        <div className="rounded-[2rem] border border-slate-200/80 bg-white/92 px-8 py-7 text-center shadow-scholar-lg">
          <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-full bg-slate-950 text-white">
            <Loader2 className="h-5 w-5 animate-spin" />
          </div>
          <div className="mt-4 text-[0.72rem] font-bold uppercase tracking-[0.22em] text-slate-400">Workspace</div>
          <div className="mt-2 text-[1rem] font-semibold tracking-tight text-slate-900">Restoring your saved workspace</div>
        </div>
      </div>
    );
  }

  if (selectedPaper) {
    return (
      <div className="h-dvh bg-[#f8fafc]">
        <SplitPaneWorkspace
          key={selectedPaper.paper_id}
          paper={selectedPaper}
          initialChatHistory={chatSessions[selectedPaper.paper_id] || []}
          onChatHistoryUpdate={(nextMessages) => updateChatSession(selectedPaper.paper_id, nextMessages)}
          onBack={handleBackFromPaper}
          onOpenGlobalSearch={() => openSearchPalette()}
        />
        <GlobalSearchPalette
          open={isPaletteOpen}
          value={paletteInput}
          isSearching={isSearching}
          suggestionGroups={QUERY_SUGGESTION_GROUPS}
          onChange={setPaletteInput}
          onClose={closeSearchPalette}
          onSubmit={handleGlobalSearchSubmit}
        />
      </div>
    );
  }

  return (
    <div className={`relative flex min-h-dvh flex-col overflow-x-hidden pt-[5.6rem] text-slate-900 ${isRunningView ? 'bg-[#f6f8fb]' : 'bg-transparent'}`}>
      <ProjectWorkspacePanel
        open={isProjectPanelOpen}
        loading={projectState === 'loading'}
        disabled={isSearching || isProjectBusy}
        scopeSaving={isScopeSaving}
        projects={projects}
        currentProjectId={currentProjectId}
        detail={projectDetail}
        corpusCatalog={corpusCatalog}
        error={projectError}
        onClose={() => runViewTransition(() => setIsProjectPanelOpen(false))}
        onSelectProject={(projectId) => void handleSelectProject(projectId)}
        onCreateProject={() => void handleCreateProject()}
        onRenameProject={(projectId, title) => void handleRenameProject(projectId, title)}
        onUpdateProjectScope={(projectId, selectedCorpora) => void handleUpdateProjectScope(projectId, selectedCorpora)}
        onClearProject={() => void handleClearProject()}
        onDeleteProject={() => void handleDeleteProject()}
        onRestoreThread={(thread) => void handleRestoreSavedThread(thread)}
        onOpenPaperSession={(session) => void handleOpenSavedPaperSession(session)}
      />
      <WorkspaceConfirmDialog
        open={workspaceActionIntent != null}
        title={
          workspaceActionIntent === 'delete'
            ? `Delete ${currentProjectTitle ?? 'this workspace'}?`
            : `Clear ${currentProjectTitle ?? 'this workspace'}?`
        }
        description={
          workspaceActionIntent === 'delete'
            ? 'This removes the saved workspace state itself, including its stored searches and paper chat sessions.'
            : 'This keeps the workspace, but removes all saved searches and paper chat sessions inside it.'
        }
        confirmLabel={workspaceActionIntent === 'delete' ? 'Delete workspace' : 'Clear workspace'}
        tone={workspaceActionIntent === 'delete' ? 'danger' : 'warning'}
        busy={isProjectMutating}
        onCancel={handleCancelWorkspaceAction}
        onConfirm={() => void handleConfirmWorkspaceAction()}
      />
      {!isRunningView ? (
        <div className="pointer-events-none absolute inset-0 overflow-hidden">
          <div className="absolute left-[-8rem] top-[-9rem] h-[26rem] w-[26rem] rounded-full bg-[radial-gradient(circle,rgba(165,180,252,0.28),rgba(165,180,252,0)_68%)]" />
          <div className="absolute right-[-10rem] top-[8rem] h-[30rem] w-[30rem] rounded-full bg-[radial-gradient(circle,rgba(186,230,253,0.28),rgba(186,230,253,0)_68%)]" />
          <div className="absolute bottom-[-12rem] left-1/2 h-[26rem] w-[40rem] -translate-x-1/2 rounded-full bg-[radial-gradient(circle,rgba(226,232,240,0.42),rgba(226,232,240,0)_72%)]" />
        </div>
      ) : null}
      {isComposerFocused ? (
        <div className='pointer-events-none absolute inset-0 z-20 bg-[linear-gradient(180deg,rgba(15,23,42,0.06),rgba(15,23,42,0.14))] backdrop-blur-[2px]' />
      ) : null}

      <header className={`${isRunningView ? 'border-b border-slate-200/70 bg-white/92' : 'glass-header'} fixed inset-x-0 top-0 z-50`}>
        <div className="mx-auto flex w-full max-w-[1440px] items-center justify-between px-5 py-4 sm:px-8">
          <div className="flex items-center gap-4">
            <ActivityMark
              mode={headerActivityMode}
              label={headerActivityMode === 'active' ? 'Searching' : headerActivityMode === 'done' ? 'Ready' : null}
              layout="stacked"
              size="lg"
              minimal={isRunningView}
            />
            <div>
              <div className="text-[0.95rem] font-black tracking-tight text-slate-950">{APP_NAME}</div>
              <div className="text-[0.66rem] font-semibold uppercase tracking-[0.22em] text-slate-400">{APP_TAGLINE}</div>
            </div>
          </div>
          <div className="flex items-center gap-3">
            {currentProjectId ? (
              <div className={`hidden max-w-[18rem] truncate rounded-full border px-4 py-2 text-[0.66rem] font-bold uppercase tracking-[0.16em] text-slate-500 xl:inline-flex ${isRunningView ? 'border-slate-200 bg-white shadow-none' : 'border-white/75 bg-white/80 shadow-scholar-sm backdrop-blur-xl'}`}>
                {scopeSummary}
              </div>
            ) : null}
            <button
              type="button"
              onClick={() => {
                if (!hasWorkspaces) {
                  void handleCreateProject();
                  return;
                }
                runViewTransition(() => setIsProjectPanelOpen(true));
              }}
              className={`inline-flex items-center gap-3 rounded-full border px-4 py-2 text-[0.72rem] font-bold uppercase tracking-[0.18em] text-slate-600 transition hover:border-indigo-200 hover:text-indigo-600 ${isRunningView ? 'border-slate-200 bg-white shadow-none' : 'border-white/75 bg-white/80 shadow-scholar-sm backdrop-blur-xl'}`}
            >
              {hasWorkspaces ? <FolderOpen className="h-3.5 w-3.5" /> : <Plus className="h-3.5 w-3.5" />}
              <span className="max-w-[10rem] truncate">{hasWorkspaces ? currentProjectTitle : 'Create workspace'}</span>
            </button>
            <button
              type="button"
              onClick={() => openSearchPalette()}
              className={`inline-flex items-center gap-3 rounded-full border px-4 py-2 text-[0.72rem] font-bold uppercase tracking-[0.18em] text-slate-600 transition hover:border-indigo-200 hover:text-indigo-600 ${isRunningView ? 'border-slate-200 bg-white shadow-none' : 'border-white/75 bg-white/80 shadow-scholar-sm backdrop-blur-xl'}`}
            >
              <Search className="h-3.5 w-3.5" />
              Search
              <span className="rounded-full border border-slate-200 bg-slate-50 px-2 py-0.5 text-[0.64rem] text-slate-400">
                ⌘K
              </span>
            </button>
          </div>
        </div>
        {!isRunningView ? (
          <div
            className={`header-status-rail ${
              headerActivityMode === 'active'
                ? 'header-status-rail--active'
                : headerActivityMode === 'done'
                  ? 'header-status-rail--done'
                  : ''
            }`}
          />
        ) : null}
      </header>

      {!hasThreadView ? (
          <main
            key="hero"
            className="psa-main-switch relative z-30 flex flex-1 items-center"
          >
            <div className="mx-auto flex w-full max-w-[1440px] flex-1 items-center px-5 py-10 sm:px-8">
              <section className="mx-auto w-full max-w-[1040px] text-center">
                <div className="inline-flex items-center gap-2 rounded-full border border-indigo-100 bg-white/86 px-4 py-2 text-[0.68rem] font-bold uppercase tracking-[0.24em] text-indigo-600 shadow-scholar-sm backdrop-blur-xl">
                  <span className="h-2 w-2 rounded-full bg-indigo-500" />
                  Scholar-grade retrieval
                </div>

                <h1 className="mt-8 text-[clamp(3.2rem,8vw,6.4rem)] font-black tracking-[-0.075em] text-slate-950">
                  Search papers with
                  <span className="block text-indigo-600">scholarly precision.</span>
                </h1>

                <p className="font-scholar mx-auto mt-6 max-w-[760px] text-[1.16rem] italic leading-8 text-slate-500 sm:text-[1.32rem]">
                  A cleaner interface for finding relevant papers, opening the right manuscript, and following the evidence all the way into the paper itself.
                </p>

                <div className="mx-auto mt-12 max-w-[900px]">{renderQueryForm('hero')}</div>

                {!isComposerFocused ? (
                  <div className="mt-7 flex flex-wrap items-center justify-center gap-3">
                    {SAMPLE_QUERIES.map((query) => (
                      <button
                        key={query}
                        type="button"
                        onClick={() => setInput(query)}
                        className="rounded-full border border-white/70 bg-white/72 px-4 py-2 text-[0.82rem] font-medium text-slate-600 shadow-scholar-sm backdrop-blur-xl transition hover:border-indigo-200 hover:bg-white hover:text-indigo-600"
                      >
                        {query}
                      </button>
                    ))}
                  </div>
                ) : null}
              </section>
            </div>
          </main>
        ) : isRunningView && activeSearchRun ? (
          <SearchRunningPreview run={activeSearchRun} />
        ) : (
          <div
            key="thread"
            className="psa-main-switch relative z-30 flex flex-1 flex-col"
          >
            <main ref={searchScrollRef} className="custom-scrollbar flex-1 overflow-y-auto">
              <div className="mx-auto w-full max-w-[1440px] px-5 pb-12 pt-10 sm:px-8">
                <div className="mx-auto max-w-[1220px] space-y-8">
                  {messages.map((message) => {
                    const isSearchResultMessage = message.type === 'search_results' && !!message.results;

                    if (isSearchResultMessage) {
                      return (
                        <section key={message.id} className="flex justify-center">
                          <div className="w-full max-w-[1180px]">
                            <div className="space-y-5">
                              <div className="assistant-answer-ready overflow-hidden rounded-[1.9rem] border border-white/80 bg-[linear-gradient(180deg,rgba(255,255,255,0.98),rgba(247,249,252,0.96))] px-6 py-5 shadow-[0_22px_60px_rgba(15,23,42,0.08)]">
                                <div className="flex items-start justify-between gap-4">
                                  <div className="min-w-0 flex items-center gap-3">
                                    <ActivityMark mode="done" label={null} layout="inline" size="md" />
                                    <div className="min-w-0">
                                      <div className="text-[0.68rem] font-bold uppercase tracking-[0.22em] text-indigo-500">
                                        {getSearchResultHeadline(message)}
                                      </div>
                                      <div className="mt-1 text-[1.08rem] font-semibold tracking-tight text-slate-950">
                                        {message.content}
                                      </div>
                                    </div>
                                  </div>
                                  <div className="hidden rounded-full border border-slate-200 bg-white px-3 py-1.5 text-[0.66rem] font-bold uppercase tracking-[0.18em] text-slate-500 sm:inline-flex">
                                    Ready for review
                                  </div>
                                </div>
                              </div>

                              <SearchTracePanel
                                message={message}
                                onToggle={() => void handleToggleTrace(message.id)}
                              />

                              <SearchResultRevealGrid
                                papers={message.results ?? []}
                                traceId={message.traceId ?? null}
                                onOpenPaper={handleOpenPaper}
                                onQuickPeek={handleQuickPeek}
                              />
                            </div>
                          </div>
                        </section>
                      );
                    }

                    return (
                      <section
                        key={message.id}
                        className="flex justify-center"
                      >
                        <div className="w-full max-w-[1180px]">
                          <div className="min-w-0 flex-1 space-y-5">
                            <div
                              className={`rounded-[2rem] border px-6 py-5 ${
                                message.role === 'user'
                                  ? 'ml-auto w-fit max-w-[min(100%,52rem)] border-slate-200/80 bg-white/86 text-slate-900 shadow-scholar-lg backdrop-blur-xl'
                                  : 'mr-auto w-fit max-w-[min(100%,54rem)] border-slate-200/90 bg-white/90 text-slate-700 shadow-scholar-lg backdrop-blur-xl'
                              }`}
                            >
                              <div className="mb-3 text-[0.68rem] font-bold uppercase tracking-[0.24em] text-slate-400">
                                {message.role === 'user' ? 'Query' : 'Search system'}
                              </div>

                              <div
                                className={`${
                                  message.role === 'user'
                                    ? 'font-scholar text-[1.18rem] italic leading-9 text-slate-700'
                                    : 'text-[1rem] leading-8'
                                }`}
                              >
                                {message.content}
                              </div>
                            </div>
                          </div>
                        </div>
                      </section>
                    );
                  })}
                  <div ref={messagesEndRef} />
                </div>
              </div>
            </main>

            <footer className="glass-header sticky bottom-0 z-30 border-t border-white/70">
              <div className="mx-auto w-full max-w-[1440px] px-5 py-4 sm:px-8">
                <div className="mx-auto max-w-[1080px]">{renderQueryForm('dock')}</div>
              </div>
            </footer>
          </div>
        )}

      <QuickPeekPanel
        paper={previewPaper?.paper ?? null}
        onClose={() => runViewTransition(() => setPreviewPaper(null))}
        onOpenPaper={(paper) => handleOpenPaper(paper, previewPaper?.threadId ?? null)}
      />
      <GlobalSearchPalette
        open={isPaletteOpen}
        value={paletteInput}
        isSearching={isSearching}
        suggestionGroups={QUERY_SUGGESTION_GROUPS}
        onChange={setPaletteInput}
        onClose={closeSearchPalette}
        onSubmit={handleGlobalSearchSubmit}
      />
    </div>
  );
}
