'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { AlertTriangle, ChevronDown, ChevronUp, FolderOpen, Loader2, Plus, Search } from 'lucide-react';

import { ActivityMark, useTransientActivityMode } from '@/components/activity-mark';
import { GlobalSearchPalette } from '@/components/global-search-palette';
import { PaperResultCard, PaperResultCardSkeleton } from '@/components/paper-result-card';
import { ProjectWorkspacePanel } from '@/components/project-workspace-panel';
import { QuickPeekPanel } from '@/components/quick-peek-panel';
import { SplitPaneWorkspace } from '@/components/split-pane-workspace';
import {
  clearProject,
  createProject,
  createSearchJob,
  deleteProject,
  fetchProject,
  fetchSearchJob,
  fetchSearchJobResult,
  fetchTrace,
  listProjects,
  renameProject,
  upsertProjectPaperSession,
  upsertProjectThread,
} from '@/lib/client-api';
import { clearCurrentProjectId, loadCurrentProjectId, saveCurrentProjectId } from '@/lib/project-ui-state';
import type {
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
  'Find ACL 2024 long papers that run experiments on the MATH dataset.',
  'Find ACL 2024 long papers that introduce a new benchmark or evaluation suite.',
  'Find ACL 2024 long papers that compare multi-agent systems against single-agent baselines.',
];

const QUERY_SUGGESTION_GROUPS = [
  {
    label: 'Dataset / Benchmark',
    items: [
      'Find ACL 2024 long papers that evaluate on the GAIA benchmark.',
      'Find ACL 2024 long papers that introduce a new benchmark or evaluation suite.',
      'Find ACL 2024 long papers that report experiments on the MATH dataset.',
    ],
  },
  {
    label: 'Method / Evaluation',
    items: [
      'Find ACL 2024 long papers that compare multi-agent systems against single-agent baselines.',
      'Find ACL 2024 long papers that study reasoning or deliberation strategies.',
      'Find ACL 2024 long papers that report ablations for routing or planning modules.',
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

  if (progress.stage_index > 0 && progress.stage_total > 0) {
    return `Stage ${progress.stage_index} of ${progress.stage_total}`;
  }
  return null;
}

function SearchProgressPanel({ message }: { message: SearchMessage }) {
  const progress = message.progress ?? null;
  const stageMeta = getStageMeta(message.currentStage);
  const overallProgressPercent = clampPercentage(Math.round((progress?.overall_progress ?? 0) * 100));
  const detail = formatProgressDetail(message.currentStage, progress);
  const activeStageIndex = progress?.stage_index ?? 0;

  return (
    <div className="mt-6 w-full max-w-[560px] rounded-[1.6rem] border border-indigo-100/90 bg-white/96 p-5 shadow-scholar-sm">
      <div className="flex items-center justify-between gap-4">
        <div>
          <div className="text-[0.68rem] font-bold uppercase tracking-[0.22em] text-slate-400">Live pipeline progress</div>
          <div className="mt-2 text-[0.96rem] font-semibold tracking-tight text-slate-900">
            {stageMeta?.label ?? message.stageMessage ?? 'Preparing the search pipeline'}
          </div>
        </div>
        <div className="text-right">
          <div className="text-[1.35rem] font-black tracking-tight text-slate-950">{overallProgressPercent}%</div>
          <div className="text-[0.68rem] font-semibold uppercase tracking-[0.18em] text-slate-400">Overall</div>
        </div>
      </div>

      <div className="mt-4 h-2.5 overflow-hidden rounded-full bg-slate-100">
        <div
          className="h-full rounded-full bg-[linear-gradient(90deg,#4f46e5_0%,#6366f1_55%,#60a5fa_100%)] transition-[width] duration-500"
          style={{ width: `${overallProgressPercent}%` }}
        />
      </div>

      <div className="mt-4 flex flex-wrap gap-2">
        {SEARCH_STAGE_SEQUENCE.map((stage, index) => {
          const stageNumber = index + 1;
          const isCompleted = activeStageIndex > stageNumber || message.status === 'completed';
          const isActive = message.currentStage === stage.id;
          return (
            <div
              key={stage.id}
              className={`inline-flex items-center gap-2 rounded-full border px-3 py-2 text-[0.68rem] font-bold uppercase tracking-[0.16em] transition-colors ${
                isCompleted
                  ? 'border-indigo-200 bg-indigo-50 text-indigo-700'
                  : isActive
                    ? 'border-slate-300 bg-slate-950 text-white'
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

      <div className="mt-4 flex items-center gap-2 text-[0.74rem] font-semibold uppercase tracking-[0.16em] text-slate-500">
        <Loader2 className="h-3.5 w-3.5 animate-spin" />
        {message.stageMessage || 'Searching'}
      </div>

      {detail ? <div className="mt-2 text-[0.82rem] leading-6 text-slate-500">{detail}</div> : null}
    </div>
  );
}

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
        <div className='text-[0.68rem] font-medium text-slate-400'>Click to draft, then edit freely</div>
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

function SearchResultsSkeletonGrid() {
  return (
    <div className='grid grid-cols-1 gap-6 lg:grid-cols-2 2xl:grid-cols-3'>
      {Array.from({ length: 6 }, (_, index) => (
        <PaperResultCardSkeleton key={`skeleton-${index}`} />
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

      <AnimatePresence initial={false}>
        {isOpen ? (
          <motion.div
            initial={{ opacity: 0, height: 0, y: 8 }}
            animate={{ opacity: 1, height: 'auto', y: 0 }}
            exit={{ opacity: 0, height: 0, y: 6 }}
            transition={{ duration: 0.18, ease: 'easeOut' }}
            className='overflow-hidden'
          >
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
          </motion.div>
        ) : null}
      </AnimatePresence>
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
  const accentClassName =
    tone === 'danger'
      ? 'border-rose-200 bg-rose-50 text-rose-600'
      : 'border-amber-200 bg-amber-50 text-amber-600';
  const confirmClassName =
    tone === 'danger'
      ? 'bg-rose-600 hover:bg-rose-700 focus-visible:ring-rose-300'
      : 'bg-slate-950 hover:bg-indigo-600 focus-visible:ring-indigo-300';

  return (
    <AnimatePresence>
      {open ? (
        <div className="fixed inset-0 z-[120] flex items-center justify-center p-4 sm:p-6">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={busy ? undefined : onCancel}
            className="absolute inset-0 bg-slate-950/34 backdrop-blur-[6px]"
          />
          <motion.section
            initial={{ opacity: 0, y: 14, scale: 0.97 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 10, scale: 0.97 }}
            transition={{ duration: 0.18, ease: 'easeOut' }}
            className="relative z-10 w-full max-w-[480px] overflow-hidden rounded-[2rem] border border-white/80 bg-[linear-gradient(180deg,rgba(255,255,255,0.98),rgba(248,250,252,0.98))] shadow-[0_30px_90px_rgba(15,23,42,0.24)]"
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
          </motion.section>
        </div>
      ) : null}
    </AnimatePresence>
  );
}

export default function ChatPage() {
  const [messages, setMessages] = useState<SearchMessage[]>([]);
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
  const [projectState, setProjectState] = useState<'loading' | 'ready' | 'error'>('loading');
  const [projectError, setProjectError] = useState<string | null>(null);
  const [isProjectPanelOpen, setIsProjectPanelOpen] = useState(false);
  const [isProjectMutating, setIsProjectMutating] = useState(false);
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
  const hasSearchHistory = messages.length > 0;
  const currentProjectTitle = projectDetail?.project.title ?? projects.find((project) => project.project_id === currentProjectId)?.title ?? null;
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

  const activeLoadingMessage = useMemo(() => {
    for (let index = messages.length - 1; index >= 0; index -= 1) {
      if (messages[index]?.status === 'loading') {
        return messages[index];
      }
    }
    return null;
  }, [messages]);

  const openSearchPalette = useCallback((prefill?: string) => {
    const nextValue = prefill ?? input.trim() ?? '';
    setPaletteInput(nextValue);
    setPreviewPaper(null);
    setIsComposerFocused(false);
    setIsPaletteOpen(true);
  }, [input]);

  const closeSearchPalette = useCallback(() => {
    setIsPaletteOpen(false);
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
      setMessages(restoredMessages);
      setPreviewPaper(null);
      setSelectedPaper(null);
      setActiveThreadId(thread.thread_id);

      if (options.openPaperId) {
        const targetPaper = displayResults.find((paper) => paper.paper_id === options.openPaperId) ?? null;
        if (targetPaper) {
          setSelectedPaper(targetPaper);
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
      const renamed = await renameProject(projectId, { title: title.trim() });
      const projectList = await listProjects();
      setProjects(projectList.projects);
      setProjectDetail((previous) => {
        if (!previous || previous.project.project_id !== projectId) {
          return previous;
        }
        return {
          ...previous,
          project: {
            ...previous.project,
            title: renamed.title,
            updated_at: renamed.updated_at,
          },
        };
      });
    } catch (error) {
      setProjectError(error instanceof Error ? error.message : 'Workspace could not be renamed.');
      setProjectState('error');
    } finally {
      setIsProjectMutating(false);
    }
  }, [isProjectBusy]);

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

  const handleSearch = useCallback(async (query: string) => {
    const normalizedQuery = query.trim();
    if (!normalizedQuery || searchInFlightRef.current || projectState === 'loading') {
      return;
    }
    searchInFlightRef.current = true;

    const userMessage: SearchMessage = {
      id: `${Date.now()}-user`,
      role: 'user',
      content: normalizedQuery,
    };

    const assistantId = `${Date.now()}-assistant`;
    const loadingMessage: SearchMessage = {
      id: assistantId,
      role: 'assistant',
      content: 'Running evidence-aware scholarly retrieval.',
      status: 'loading',
      currentStage: 'queued',
      stageMessage: 'Submitting the search job.',
      progress: null,
    };

    setMessages((previous) => [...previous, userMessage, loadingMessage]);
    setInput('');
    setIsSearching(true);
    setPreviewPaper(null);

    try {
      const targetProjectId = await ensureWorkspaceForSearch();
      const job = await createSearchJob({ query: normalizedQuery, top_k: 15, display_k: 10 });
      let lastStatus: SearchJobStatus = job;

      setMessages((previous) => {
        let changed = false;
        const next = previous.map((message) => {
          if (message.id !== assistantId) {
            return message;
          }

          const nextProgress = job.progress ?? null;
          if (
            message.currentStage === job.stage &&
            message.stageMessage === job.message &&
            progressSignature(message.progress) === progressSignature(nextProgress)
          ) {
            return message;
          }

          changed = true;
          return {
            ...message,
            currentStage: job.stage,
            stageMessage: job.message,
            progress: nextProgress,
          };
        });

        return changed ? next : previous;
      });

      while (lastStatus.status !== 'completed' && lastStatus.status !== 'failed') {
        await new Promise((resolve) => setTimeout(resolve, 1400));
        lastStatus = await fetchSearchJob(job.job_id);

        setMessages((previous) => {
          let changed = false;
          const next = previous.map((message) => {
            if (message.id !== assistantId) {
              return message;
            }

            const nextProgress = lastStatus.progress ?? null;
            if (
              message.currentStage === lastStatus.stage &&
              message.stageMessage === lastStatus.message &&
              progressSignature(message.progress) === progressSignature(nextProgress)
            ) {
              return message;
            }

            changed = true;
            return {
              ...message,
              currentStage: lastStatus.stage,
              stageMessage: lastStatus.message,
              progress: nextProgress,
            };
          });

          return changed ? next : previous;
        });
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
        };

        setMessages((previous) =>
          previous.map((message) =>
            message.id === assistantId
              ? {
                  ...message,
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
                }
              : message,
          ),
        );
        setActiveThreadId(result.trace_id);
        void persistProjectThread(targetProjectId, threadRecord);
        return;
      }

      throw new Error('Search failed.');
    } catch {
      setMessages((previous) =>
        previous.map((message) =>
          message.id === assistantId
            ? {
                ...message,
                content: 'The search did not complete successfully.',
                status: 'error',
                progress: null,
              }
            : message,
        ),
      );
    } finally {
      searchInFlightRef.current = false;
      setIsSearching(false);
    }
  }, [ensureWorkspaceForSearch, persistProjectThread, projectState]);

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

    setIsPaletteOpen(false);
    setPaletteInput('');
    setPreviewPaper(null);
    if (selectedPaper) {
      setSelectedPaper(null);
    }
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
    setPreviewPaper({ paper, threadId: threadId ?? activeThreadId ?? null });
  }, [activeThreadId]);

  const handleOpenPaper = useCallback((paper: PaperResult, threadId?: string | null) => {
    pendingSearchScrollTopRef.current = searchScrollRef.current?.scrollTop ?? 0;
    setPreviewPaper(null);
    if (threadId) {
      setActiveThreadId(threadId);
    }
    setSelectedPaper(paper);
  }, []);

  const handleBackFromPaper = useCallback(() => {
    setSelectedPaper(null);
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
                disabled={isSearching}
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
                disabled={isSearching || input.trim().length === 0}
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

          <AnimatePresence initial={false}>
            {showSuggestions ? (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 8 }}
                transition={{ duration: 0.18, ease: 'easeOut' }}
                className={isHero ? 'relative z-30' : 'relative z-30'}
              >
                <SearchSuggestionPanel
                  variant={variant}
                  onSelect={(query) => {
                    setInput(query);
                    composerTextareaRef.current?.focus();
                  }}
                />
              </motion.div>
            ) : null}
          </AnimatePresence>
        </div>
      );
    },
    [handleSearch, input, isComposerFocused, isSearching],
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
    <div className="relative flex min-h-dvh flex-col overflow-x-hidden bg-transparent pt-[5.6rem] text-slate-900">
      <ProjectWorkspacePanel
        open={isProjectPanelOpen}
        loading={projectState === 'loading'}
        disabled={isSearching || isProjectBusy}
        projects={projects}
        currentProjectId={currentProjectId}
        detail={projectDetail}
        error={projectError}
        onClose={() => setIsProjectPanelOpen(false)}
        onSelectProject={(projectId) => void handleSelectProject(projectId)}
        onCreateProject={() => void handleCreateProject()}
        onRenameProject={(projectId, title) => void handleRenameProject(projectId, title)}
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
      <div className="pointer-events-none absolute inset-0 overflow-hidden">
        <div className="absolute left-[-8rem] top-[-9rem] h-[26rem] w-[26rem] rounded-full bg-indigo-200/25 blur-3xl" />
        <div className="absolute right-[-10rem] top-[8rem] h-[30rem] w-[30rem] rounded-full bg-cyan-100/30 blur-3xl" />
        <div className="absolute bottom-[-12rem] left-1/2 h-[26rem] w-[40rem] -translate-x-1/2 rounded-full bg-slate-200/30 blur-3xl" />
      </div>
      <AnimatePresence>
        {isComposerFocused ? (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className='pointer-events-none absolute inset-0 z-20 bg-[linear-gradient(180deg,rgba(15,23,42,0.06),rgba(15,23,42,0.14))] backdrop-blur-[2px]'
          />
        ) : null}
      </AnimatePresence>

      <header className="glass-header fixed inset-x-0 top-0 z-50">
        <div className="mx-auto flex w-full max-w-[1440px] items-center justify-between px-5 py-4 sm:px-8">
            <div className="flex items-center gap-4">
            <ActivityMark
              mode={headerActivityMode}
              label={headerActivityMode === 'active' ? 'Searching' : headerActivityMode === 'done' ? 'Ready' : null}
              layout="stacked"
              size="lg"
            />
            <div>
              <div className="text-[0.95rem] font-black tracking-tight text-slate-950">Scholar Agent</div>
              <div className="text-[0.66rem] font-semibold uppercase tracking-[0.22em] text-slate-400">Neural literature interface</div>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={() => {
                if (!hasWorkspaces) {
                  void handleCreateProject();
                  return;
                }
                setIsProjectPanelOpen(true);
              }}
              className="inline-flex items-center gap-3 rounded-full border border-white/75 bg-white/80 px-4 py-2 text-[0.72rem] font-bold uppercase tracking-[0.18em] text-slate-600 shadow-scholar-sm backdrop-blur-xl transition hover:border-indigo-200 hover:text-indigo-600"
            >
              {hasWorkspaces ? <FolderOpen className="h-3.5 w-3.5" /> : <Plus className="h-3.5 w-3.5" />}
              <span className="max-w-[10rem] truncate">{hasWorkspaces ? currentProjectTitle : 'Create workspace'}</span>
            </button>
            <button
              type="button"
              onClick={() => openSearchPalette()}
              className="inline-flex items-center gap-3 rounded-full border border-white/75 bg-white/80 px-4 py-2 text-[0.72rem] font-bold uppercase tracking-[0.18em] text-slate-600 shadow-scholar-sm backdrop-blur-xl transition hover:border-indigo-200 hover:text-indigo-600"
            >
              <Search className="h-3.5 w-3.5" />
              Search
              <span className="rounded-full border border-slate-200 bg-slate-50 px-2 py-0.5 text-[0.64rem] text-slate-400">
                ⌘K
              </span>
            </button>
          </div>
        </div>
      </header>

      <AnimatePresence mode="wait">
        {!hasSearchHistory ? (
          <motion.main
            key="hero"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -12 }}
            className="relative z-30 flex flex-1 items-center"
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
          </motion.main>
        ) : (
          <motion.div
            key="thread"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -12 }}
            className="relative z-30 flex flex-1 flex-col"
          >
            <main ref={searchScrollRef} className="custom-scrollbar flex-1 overflow-y-auto">
              <div className="mx-auto w-full max-w-[1440px] px-5 pb-12 pt-10 sm:px-8">
                <div className="mx-auto max-w-[1220px] space-y-8">
                  {messages.map((message) => (
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
                                : message.type === 'search_results'
                                  ? 'border-transparent bg-transparent px-0 py-0 shadow-none'
                                  : 'mr-auto w-fit max-w-[min(100%,54rem)] border-slate-200/90 bg-white/90 text-slate-700 shadow-scholar-lg backdrop-blur-xl'
                            }`}
                          >
                            {message.type !== 'search_results' ? (
                              <div className="mb-3 text-[0.68rem] font-bold uppercase tracking-[0.24em] text-slate-400">
                                {message.role === 'user' ? 'Query' : 'Search system'}
                              </div>
                            ) : null}

                            <div
                              className={`${
                                message.type === 'search_results'
                                  ? 'text-[1.75rem] font-black tracking-tight text-slate-950'
                                  : message.role === 'user'
                                    ? 'font-scholar text-[1.18rem] italic leading-9 text-slate-700'
                                    : 'text-[1rem] leading-8'
                              }`}
                            >
                              {message.content}
                            </div>

                            {message.status === 'loading' ? <SearchProgressPanel message={message} /> : null}
                          </div>

                          {message.status === 'loading' ? <SearchResultsSkeletonGrid /> : null}

                          {message.type === 'search_results' ? (
                            <SearchTracePanel
                              message={message}
                              onToggle={() => void handleToggleTrace(message.id)}
                            />
                          ) : null}

                          {message.type === 'search_results' && message.results ? (
                            <div className="grid grid-cols-1 gap-6 lg:grid-cols-2 2xl:grid-cols-3">
                              {message.results.map((paper) => (
                                <PaperResultCard
                                  key={paper.paper_id}
                                  paper={paper}
                                  onOpenPaper={(nextPaper) => handleOpenPaper(nextPaper, message.traceId ?? null)}
                                  onQuickPeek={(nextPaper) => handleQuickPeek(nextPaper, message.traceId ?? null)}
                                />
                              ))}
                            </div>
                          ) : null}
                        </div>
                      </div>
                    </section>
                  ))}
                  <div ref={messagesEndRef} />
                </div>
              </div>
            </main>

            <footer className="glass-header sticky bottom-0 z-30 border-t border-white/70">
              <div className="mx-auto w-full max-w-[1440px] px-5 py-4 sm:px-8">
                <div className="mx-auto max-w-[1080px] space-y-3">
                  {activeLoadingMessage?.stageMessage ? (
                    <div className="text-center text-[0.72rem] font-semibold uppercase tracking-[0.18em] text-slate-500">
                      {activeLoadingMessage.stageMessage}
                      {activeLoadingMessage.progress ? ` • ${clampPercentage(Math.round(activeLoadingMessage.progress.overall_progress * 100))}%` : ''}
                    </div>
                  ) : null}
                  {renderQueryForm('dock')}
                </div>
              </div>
            </footer>
          </motion.div>
        )}
      </AnimatePresence>

      <QuickPeekPanel
        paper={previewPaper?.paper ?? null}
        onClose={() => setPreviewPaper(null)}
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
