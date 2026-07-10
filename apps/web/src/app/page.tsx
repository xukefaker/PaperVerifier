'use client';

import Image from 'next/image';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import * as Progress from '@radix-ui/react-progress';
import * as Select from '@radix-ui/react-select';
import * as Switch from '@radix-ui/react-switch';
import * as Tabs from '@radix-ui/react-tabs';
import {
  AlertCircle,
  BookOpen,
  Brain,
  CheckCircle2,
  ChevronDown,
  Database,
  Filter,
  Layers3,
  Loader2,
  PanelLeftClose,
  PanelLeftOpen,
  PanelRightOpen,
  Play,
  Quote,
  RefreshCw,
  Search,
  Settings,
  ThumbsDown,
  ThumbsUp,
  UploadCloud,
} from 'lucide-react';
import {
  createColumnHelper,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  useReactTable,
  type SortingState,
} from '@tanstack/react-table';

import { APP_NAME } from '@/lib/branding';
import type { IndexingDevice, RetrievalMethod } from '@/lib/workbench-store';
import { PaperPdfViewer } from '@/components/paper-pdf-viewer';

type LibraryPaper = {
  paper_id: string;
  title: string;
  authors: string[];
  year: number;
  venue: string;
  pages: number;
  figures: number;
  status: 'ready' | 'indexing' | 'failed' | 'queued';
  tags: string[];
  updated_at: string;
  abstract: string;
  preview_label: string;
};

type LibraryJob = {
  job_id: string;
  kind: 'upload' | 'parse' | 'index';
  file_name: string;
  status: 'queued' | 'running' | 'ready' | 'failed';
  progress: number;
  message: string;
  paper_id?: string;
};

type SearchResult = LibraryPaper & {
  rank: number;
  score: number;
  retrieval_method: RetrievalMethod;
  matched_terms: string[];
  reason: string;
  preview_image_url?: string | null;
};

type EvidenceUnit = {
  evidence_id: string;
  evidence_type?: string;
  heading: string;
  page_start: number;
  page_end: number;
  text: string;
  caption?: string;
  footnote?: string;
  image_url?: string | null;
  table?: {
    rows: {
      cells: {
        text: string;
        colspan?: number;
        rowspan?: number;
        is_header?: boolean;
      }[];
    }[];
  } | null;
  alias_evidence_ids?: string[];
};

type VisibleEvidenceUnit = EvidenceUnit & {
  aliasEvidenceIds: string[];
};

type PaperViewer = {
  paper_id: string;
  title: string;
  abstract: string;
  evidence_units: EvidenceUnit[];
};

type WorkbenchSettings = {
  library_path: string;
  retrieval_method: RetrievalMethod;
  qa_model: string;
  qa_base_url: string;
  qa_api_key: string;
  qa_api_key_set?: boolean;
  max_context_tokens: number;
  qa_timeout_seconds: number;
  enable_citations: boolean;
  indexing_device: IndexingDevice;
  cuda_visible_devices: string;
};

type SystemGpu = {
  index: number;
  name: string;
  memory_total_mb: number;
  memory_used_mb: number;
  utilization_gpu: number;
  processes: { pid: string; name: string; used_memory_mb: number }[];
};

type SystemGpuPayload = {
  available: boolean;
  cuda_visible_devices: string;
  gpus: SystemGpu[];
};

type SearchStatus = {
  job_id: string;
  status: 'running' | 'completed' | 'failed';
  stage: string;
  message: string;
  progress: number;
};

export type ChatMessage = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  pending?: boolean;
  citations?: {
    evidence_id: string;
    page_start: number;
    page_end: number;
    section_path: string[];
    snippet: string;
  }[];
  feedback?: {
    vote: 'up' | 'down';
    reason?: FeedbackReason;
    note?: string;
    submitted_at: string;
  };
};

type FeedbackReason = 'incorrect' | 'missing_evidence' | 'not_clear' | 'other';

type FeedbackDraft = {
  open: boolean;
  reason: FeedbackReason;
  note: string;
  saving: boolean;
};

const retrievalOptions: { value: RetrievalMethod; label: string; shortLabel: string }[] = [
  { value: 'bm25_full_text', label: 'BM25 full text', shortLabel: 'BM25' },
  { value: 'colbertv2', label: 'ColBERTv2', shortLabel: 'ColBERTv2' },
  { value: 'spladepp', label: 'SPLADE++', shortLabel: 'SPLADE++' },
  { value: 'hybrid_bm25_colbertv2', label: 'Hybrid BM25 + ColBERTv2', shortLabel: 'BM25 + ColBERTv2' },
  { value: 'hybrid_bm25_spladepp', label: 'Hybrid BM25 + SPLADE++', shortLabel: 'BM25 + SPLADE++' },
];

const indexingDeviceOptions: { value: IndexingDevice; label: string }[] = [
  { value: 'auto', label: 'Auto' },
  { value: 'cpu', label: 'CPU' },
  { value: 'cuda', label: 'CUDA' },
];

const feedbackReasons: { value: FeedbackReason; label: string }[] = [
  { value: 'incorrect', label: 'Incorrect' },
  { value: 'missing_evidence', label: 'Missing evidence' },
  { value: 'not_clear', label: 'Not clear' },
  { value: 'other', label: 'Other' },
];

const sampleQuery = 'Find papers on donor-acceptor covalent organic frameworks for coupled CO2 reduction and water oxidation.';
const columnHelper = createColumnHelper<LibraryPaper>();

function makeMessageId(role: ChatMessage['role']) {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return `${role}-${crypto.randomUUID()}`;
  }
  return `${role}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function questionForAssistant(messages: ChatMessage[], assistantIndex: number) {
  for (let index = assistantIndex - 1; index >= 0; index -= 1) {
    if (messages[index]?.role === 'user') return messages[index].content;
  }
  return '';
}

function normalizeEvidenceText(text: string) {
  return text.replace(/\s+/g, ' ').trim();
}

function isGenericEvidenceHeading(heading?: string | null) {
  const compact = (heading ?? '').trim().toLowerCase().replace(/[^a-z0-9]+/g, '');
  return compact === 'papertext' || compact === 'articleinfo' || compact === 'checkforupdates';
}

function getEvidenceDedupeKey(unit: EvidenceUnit) {
  return `${unit.evidence_type ?? ''}|${unit.heading.trim().toLowerCase()}|${unit.page_start}|${unit.page_end}|${normalizeEvidenceText(unit.text)}`;
}

function isStructuredEvidenceUnit(unit: EvidenceUnit) {
  return Boolean(unit.image_url || unit.table?.rows?.length);
}

function evidenceAliases(unit: EvidenceUnit) {
  return unit.alias_evidence_ids ?? [];
}

function pagesOverlap(left: EvidenceUnit, right: EvidenceUnit) {
  return left.page_start <= right.page_end && right.page_start <= left.page_end;
}

function sameEvidenceRegion(left: EvidenceUnit, right: EvidenceUnit) {
  return left.heading.trim().toLowerCase() === right.heading.trim().toLowerCase() && pagesOverlap(left, right);
}

function compactEvidenceUnits(units: EvidenceUnit[]): VisibleEvidenceUnit[] {
  const compacted: VisibleEvidenceUnit[] = [];

  for (const unit of units) {
    if (isStructuredEvidenceUnit(unit)) {
      compacted.push({ ...unit, aliasEvidenceIds: evidenceAliases(unit) });
      continue;
    }

    const key = getEvidenceDedupeKey(unit);
    const normalizedText = normalizeEvidenceText(unit.text);
    const exactDuplicate = compacted.find((entry) => getEvidenceDedupeKey(entry) === key);
    if (exactDuplicate) {
      exactDuplicate.aliasEvidenceIds.push(unit.evidence_id, ...evidenceAliases(unit));
      continue;
    }

    const container = compacted.find((entry) => {
      if (isStructuredEvidenceUnit(entry)) return false;
      if (!sameEvidenceRegion(entry, unit)) return false;
      const entryText = normalizeEvidenceText(entry.text);
      return entryText.length > normalizedText.length && entryText.includes(normalizedText);
    });
    if (container) {
      container.aliasEvidenceIds.push(unit.evidence_id, ...evidenceAliases(unit));
      continue;
    }

    const containedIndex = compacted.findIndex((entry) => {
      if (isStructuredEvidenceUnit(entry)) return false;
      if (!sameEvidenceRegion(entry, unit)) return false;
      const entryText = normalizeEvidenceText(entry.text);
      return normalizedText.length > entryText.length && normalizedText.includes(entryText);
    });
    if (containedIndex >= 0) {
      const contained = compacted[containedIndex];
      compacted[containedIndex] = {
        ...unit,
        aliasEvidenceIds: [contained.evidence_id, ...contained.aliasEvidenceIds, ...evidenceAliases(unit)],
      };
      continue;
    }

    compacted.push({ ...unit, aliasEvidenceIds: evidenceAliases(unit) });
  }

  return compacted;
}

function StructuredEvidenceTable({ unit }: { unit: VisibleEvidenceUnit }) {
  if (!unit.table?.rows?.length) return null;
  return (
    <div className="overflow-x-auto rounded-xl border border-slate-200 bg-white">
      <table className="min-w-full border-collapse text-left text-[13px] leading-6 text-slate-700">
        <tbody>
          {unit.table.rows.map((row, rowIndex) => (
            <tr key={`${unit.evidence_id}-row-${rowIndex}`} className={rowIndex === 0 ? 'bg-slate-50' : 'odd:bg-white even:bg-slate-50/50'}>
              {row.cells.map((cell, cellIndex) => {
                const cellClassName = 'border border-slate-200 px-3 py-2 align-top';
                const key = `${unit.evidence_id}-cell-${rowIndex}-${cellIndex}`;
                return cell.is_header || rowIndex === 0 ? (
                  <th key={key} colSpan={cell.colspan} rowSpan={cell.rowspan} className={`${cellClassName} font-semibold text-slate-950`}>
                    {cell.text}
                  </th>
                ) : (
                  <td key={key} colSpan={cell.colspan} rowSpan={cell.rowspan} className={cellClassName}>
                    {cell.text}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function EvidenceUnitBody({ unit }: { unit: VisibleEvidenceUnit }) {
  if (unit.image_url) {
    return (
      <figure className="space-y-3">
        <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
          <Image
            src={unit.image_url}
            alt={unit.caption || unit.text || unit.heading}
            width={960}
            height={720}
            unoptimized
            className="mx-auto max-h-[520px] w-full object-contain"
          />
        </div>
        {unit.caption || unit.text ? <figcaption className="text-sm leading-7 text-slate-600">{unit.caption || unit.text}</figcaption> : null}
        {unit.footnote ? <p className="text-xs leading-5 text-slate-500">{unit.footnote}</p> : null}
      </figure>
    );
  }

  if (unit.table?.rows?.length) {
    return (
      <div className="space-y-3">
        <StructuredEvidenceTable unit={unit} />
        {unit.caption ? <p className="text-sm leading-7 text-slate-600">{unit.caption}</p> : null}
        {unit.footnote ? <p className="text-xs leading-5 text-slate-500">{unit.footnote}</p> : null}
      </div>
    );
  }

  return unit.text ? <p className="text-sm leading-7 text-slate-700">{unit.text}</p> : null;
}

async function getJson<T>(path: string, init?: RequestInit): Promise<T> {
  const body = init?.body;
  const isFormData = typeof FormData !== 'undefined' && body instanceof FormData;
  const response = await fetch(path, {
    ...init,
    headers: body && !isFormData ? { 'Content-Type': 'application/json', ...(init.headers ?? {}) } : init?.headers,
    cache: 'no-store',
  });
  if (!response.ok) {
    const text = await response.text();
    let message = text;
    try {
      const payload = JSON.parse(text) as { detail?: string; error?: { message?: string; fix?: string } };
      message = payload.error?.message ?? payload.detail ?? text;
      if (payload.error?.fix) message = `${message} ${payload.error.fix}`;
    } catch {}
    throw new Error(message);
  }
  return (await response.json()) as T;
}

function StatusBadge({ status }: { status: LibraryPaper['status'] | LibraryJob['status'] }) {
  const palette = {
    ready: 'border-emerald-200 bg-emerald-50 text-emerald-700',
    running: 'border-blue-200 bg-blue-50 text-blue-700',
    indexing: 'border-blue-200 bg-blue-50 text-blue-700',
    queued: 'border-amber-200 bg-amber-50 text-amber-700',
    failed: 'border-rose-200 bg-rose-50 text-rose-700',
  }[status];
  return <span className={`inline-flex rounded-full border px-2.5 py-1 text-xs font-medium ${palette}`}>{status}</span>;
}

function SelectControl<T extends string>({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: T;
  options: { value: T; label: string }[];
  onChange: (value: T) => void;
}) {
  return (
    <label className="block">
      <span className="mb-2 block text-xs font-semibold uppercase tracking-[0.14em] text-slate-500">{label}</span>
      <Select.Root value={value} onValueChange={(next) => onChange(next as T)}>
        <Select.Trigger className="flex h-11 w-full items-center justify-between rounded-xl border border-slate-200 bg-white px-3 text-left text-sm font-medium text-slate-900 shadow-sm outline-none transition hover:border-slate-300 focus:border-blue-500">
          <Select.Value />
          <Select.Icon>
            <ChevronDown className="h-4 w-4 text-slate-500" />
          </Select.Icon>
        </Select.Trigger>
        <Select.Portal>
          <Select.Content className="z-50 min-w-[260px] overflow-hidden rounded-xl border border-slate-200 bg-white p-1 shadow-xl">
            <Select.Viewport>
              {options.map((option) => (
                <Select.Item
                  key={option.value}
                  value={option.value}
                  className="cursor-pointer rounded-lg px-3 py-2 text-sm outline-none hover:bg-slate-100 data-[highlighted]:bg-slate-100"
                >
                  <Select.ItemText>
                    <span className="block font-medium text-slate-900">{option.label}</span>
                  </Select.ItemText>
                </Select.Item>
              ))}
            </Select.Viewport>
          </Select.Content>
        </Select.Portal>
      </Select.Root>
    </label>
  );
}

function MetricCard({ label, value, icon: Icon }: { label: string; value: string | number; icon: typeof Database }) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="mb-3 flex h-9 w-9 items-center justify-center rounded-xl bg-slate-100 text-slate-700">
        <Icon className="h-4 w-4" />
      </div>
      <div className="text-2xl font-semibold text-slate-950">{value}</div>
      <div className="text-sm text-slate-500">{label}</div>
    </div>
  );
}

function isGpuBusy(gpu: SystemGpu) {
  return gpu.processes.length > 0 || gpu.memory_used_mb > 1024;
}

function GpuState({ gpu, selected }: { gpu: SystemGpu; selected: boolean }) {
  const busy = isGpuBusy(gpu);
  const state = selected ? 'Selected for indexing' : busy ? 'In use' : 'Available';
  const palette = selected
    ? 'bg-blue-50 text-blue-700'
    : busy
      ? 'bg-amber-50 text-amber-700'
      : 'bg-emerald-50 text-emerald-700';
  return <span className={`rounded-full px-2.5 py-1 text-xs font-semibold ${palette}`}>{state}</span>;
}

function PaperPreview({ imageUrl }: { imageUrl?: string | null }) {
  return (
    <div className="relative h-28 min-w-40 overflow-hidden rounded-xl border border-slate-200 bg-slate-50">
      {imageUrl ? (
        <Image src={imageUrl} alt="" fill sizes="160px" className="object-contain p-2" unoptimized />
      ) : (
        <div className="flex h-full w-full items-center justify-center px-3 text-center text-xs font-medium text-slate-400">
          No figure preview
        </div>
      )}
    </div>
  );
}

function RetrievalPicker({ value, onChange }: { value: RetrievalMethod; onChange: (value: RetrievalMethod) => void }) {
  return (
    <>
      <div className="lg:hidden">
        <SelectControl label="Retrieval" value={value} options={retrievalOptions} onChange={onChange} />
      </div>
      <div className="hidden flex-1 rounded-xl border border-slate-200 bg-white p-1 shadow-sm lg:grid lg:grid-cols-[0.7fr_0.9fr_0.9fr_1.35fr_1.35fr]">
        {retrievalOptions.map((option) => {
          const selected = option.value === value;
          return (
            <button
              key={option.value}
              type="button"
              onClick={() => onChange(option.value)}
              className={`h-10 rounded-lg px-3 text-sm font-semibold transition ${
                selected ? 'bg-white text-blue-700 shadow-sm ring-1 ring-blue-500' : 'text-slate-600 hover:bg-slate-50 hover:text-slate-950'
              }`}
              aria-pressed={selected}
            >
              {option.shortLabel}
            </button>
          );
        })}
      </div>
    </>
  );
}

export default function WorkbenchPage() {
  const [activeView, setActiveView] = useState('search');
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [papers, setPapers] = useState<LibraryPaper[]>([]);
  const [jobs, setJobs] = useState<LibraryJob[]>([]);
  const [settings, setSettings] = useState<WorkbenchSettings | null>(null);
  const [gpuPayload, setGpuPayload] = useState<SystemGpuPayload | null>(null);
  const [sorting, setSorting] = useState<SortingState>([]);
  const [libraryFilter, setLibraryFilter] = useState('');
  const [searchQuery, setSearchQuery] = useState(sampleQuery);
  const [retrievalMethod, setRetrievalMethod] = useState<RetrievalMethod>('hybrid_bm25_colbertv2');
  const [qaDraft, setQaDraft] = useState({ qa_base_url: '', qa_api_key: '', qa_model: 'gpt-5.4-mini', max_context_tokens: 128000, qa_timeout_seconds: 120 });
  const [llmTest, setLlmTest] = useState<{ ok: boolean; message: string } | null>(null);
  const [llmTesting, setLlmTesting] = useState(false);
  const [searchStatus, setSearchStatus] = useState<SearchStatus | null>(null);
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [selectedPaper, setSelectedPaper] = useState<LibraryPaper | null>(null);
  const [viewer, setViewer] = useState<PaperViewer | null>(null);
  const [paperViewMode, setPaperViewMode] = useState<'text' | 'pdf'>('text');
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [feedbackDrafts, setFeedbackDrafts] = useState<Record<string, FeedbackDraft>>({});
  const [chatInput, setChatInput] = useState('What is the paper main idea?');
  const isPaperThinking = chatMessages.some((message) => message.pending);
  const [activeEvidenceId, setActiveEvidenceId] = useState<string | null>(null);
  const [bannerError, setBannerError] = useState<string | null>(null);
  const [isDraggingUpload, setIsDraggingUpload] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const evidenceRefs = useRef<Record<string, HTMLDivElement | null>>({});
  const paperScrollRef = useRef<HTMLDivElement | null>(null);
  const chatEndRef = useRef<HTMLDivElement | null>(null);
  const showError = useCallback((message: string) => {
    setBannerError(message);
  }, []);

  const loadLibrary = useCallback(async () => {
    const [paperPayload, jobPayload] = await Promise.all([
      getJson<{ papers: LibraryPaper[] }>('/api/library/papers'),
      getJson<{ jobs: LibraryJob[] }>('/api/library/jobs'),
    ]);
    setPapers(paperPayload.papers);
    setJobs(jobPayload.jobs);
    setSelectedPaper((current) => current ?? paperPayload.papers[0] ?? null);
  }, []);

  const loadSettings = useCallback(async () => {
    const nextSettings = await getJson<WorkbenchSettings>('/api/settings');
    setSettings(nextSettings);
    setRetrievalMethod(nextSettings.retrieval_method);
    setQaDraft({
      qa_base_url: nextSettings.qa_base_url,
      qa_api_key: '',
      qa_model: nextSettings.qa_model,
      max_context_tokens: nextSettings.max_context_tokens,
      qa_timeout_seconds: nextSettings.qa_timeout_seconds,
    });
  }, []);

  const loadGpus = useCallback(async () => {
    setGpuPayload(await getJson<SystemGpuPayload>('/api/system/gpus'));
  }, []);

  useEffect(() => {
    void loadLibrary();
    void loadSettings();
    void loadGpus();
  }, [loadGpus, loadLibrary, loadSettings]);

  useEffect(() => {
    if (!jobs.some((job) => job.status === 'queued' || job.status === 'running')) return;
    const timer = window.setInterval(() => {
      void loadLibrary();
    }, 2000);
    return () => window.clearInterval(timer);
  }, [jobs, loadLibrary]);

  useEffect(() => {
    if (activeView === 'settings') {
      void loadGpus();
    }
  }, [activeView, loadGpus]);

  const uploadFiles = useCallback(
    async (files: FileList | File[]) => {
      const pdfs = Array.from(files).filter((file) => file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf'));
      if (!pdfs.length) {
        showError('Only PDF files are supported.');
        return;
      }
      try {
        for (const file of pdfs) {
          const formData = new FormData();
          formData.append('file', file, file.name);
          await getJson('/api/library/upload', { method: 'POST', body: formData });
        }
        setBannerError(null);
        await loadLibrary();
      } catch (error) {
        showError(error instanceof Error ? error.message : 'Upload failed.');
      }
    },
    [loadLibrary, showError],
  );

  useEffect(() => {
    if (!selectedPaper) return;
    void getJson<PaperViewer>(`/api/papers/${encodeURIComponent(selectedPaper.paper_id)}/viewer`).then((payload) => {
      setViewer(payload);
      setActiveEvidenceId(null);
    });
  }, [selectedPaper]);

  const scrollEvidenceIntoView = useCallback((evidenceId: string) => {
    const node = evidenceRefs.current[evidenceId];
    const scroller = paperScrollRef.current;
    if (!node) return;
    if (!scroller) {
      node.scrollIntoView({ block: 'center', behavior: 'smooth' });
      return;
    }
    const nodeBox = node.getBoundingClientRect();
    const scrollerBox = scroller.getBoundingClientRect();
    const top = scroller.scrollTop + nodeBox.top - scrollerBox.top - Math.max(24, Math.round(scroller.clientHeight * 0.12));
    scroller.scrollTo({ top: Math.max(0, top), behavior: 'smooth' });
  }, []);

  const jumpToEvidence = useCallback((evidenceId: string) => {
    setPaperViewMode('text');
    setActiveEvidenceId(evidenceId);
    window.setTimeout(() => scrollEvidenceIntoView(evidenceId), 50);
  }, [scrollEvidenceIntoView]);

  useEffect(() => {
    if (!activeEvidenceId) return;
    window.setTimeout(() => scrollEvidenceIntoView(activeEvidenceId), 50);
  }, [activeEvidenceId, scrollEvidenceIntoView]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ block: 'end', behavior: 'smooth' });
  }, [chatMessages]);

  const filteredPapers = useMemo(() => {
    const needle = libraryFilter.trim().toLowerCase();
    if (!needle) return papers;
    return papers.filter((paper) => `${paper.title} ${paper.authors.join(' ')}`.toLowerCase().includes(needle));
  }, [libraryFilter, papers]);

  const visibleJobs = useMemo(() => {
    const seen = new Set<string>();
    const unique = jobs.filter((job) => {
      const key = job.paper_id ?? job.file_name;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
    const active = unique.filter((job) => job.status === 'queued' || job.status === 'running' || job.status === 'failed');
    const ready = unique.filter((job) => job.status === 'ready').slice(0, Math.max(0, 5 - active.length));
    return [...active, ...ready].slice(0, 5);
  }, [jobs]);

  const visibleEvidenceUnits = useMemo(() => compactEvidenceUnits(viewer?.evidence_units ?? []), [viewer]);

  const columns = useMemo(
    () => [
      columnHelper.accessor('title', {
        header: 'Paper',
        cell: (info) => (
          <button className="text-left" onClick={() => setSelectedPaper(info.row.original)}>
            <span className="block font-semibold text-slate-950">{info.getValue()}</span>
            <span className="block text-xs text-slate-500">{info.row.original.authors.join(', ')}</span>
          </button>
        ),
      }),
      columnHelper.accessor('status', { header: 'Status', cell: (info) => <StatusBadge status={info.getValue()} /> }),
    ],
    [],
  );

  const table = useReactTable({
    data: filteredPapers,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  });

  const startIndexing = async (retryFailed = false) => {
    try {
      await getJson('/api/library/index', {
        method: 'POST',
        body: JSON.stringify({ retry_failed: retryFailed }),
      });
      setBannerError(null);
      await loadLibrary();
    } catch (error) {
      showError(error instanceof Error ? error.message : 'Indexing service could not be started.');
    }
  };

  const runSearch = async () => {
    setActiveView('search');
    setSearchResults([]);
    setBannerError(null);
    const loadResults = async (jobId: string) => {
      const result = await getJson<{ results: SearchResult[] }>(`/api/search/jobs/${encodeURIComponent(jobId)}/result`);
      setSearchResults(result.results);
      setSelectedPaper(result.results[0] ?? null);
    };
    setSearchStatus({
      job_id: 'starting',
      status: 'running',
      stage: 'Starting search',
      message: 'Preparing the indexed paper library.',
      progress: 5,
    });
    try {
      const status = await getJson<SearchStatus>('/api/search/jobs', {
        method: 'POST',
        body: JSON.stringify({ query: searchQuery, retrieval_method: retrievalMethod, qa_model: settings?.qa_model, corpus_scope: 'ready-papers' }),
      });
      setSearchStatus(status);
      if (status.status === 'failed') {
        showError(status.message);
        return;
      }
      if (status.status === 'completed') {
        await loadResults(status.job_id);
        return;
      }
      const timer = window.setInterval(async () => {
        try {
          const nextStatus = await getJson<SearchStatus>(`/api/search/jobs/${encodeURIComponent(status.job_id)}`);
          setSearchStatus(nextStatus);
          if (nextStatus.status === 'failed') {
            window.clearInterval(timer);
            showError(nextStatus.message);
          }
          if (nextStatus.status === 'completed') {
            window.clearInterval(timer);
            await loadResults(status.job_id);
          }
        } catch (error) {
          window.clearInterval(timer);
          showError(error instanceof Error ? error.message : 'Search failed.');
        }
      }, 500);
    } catch (error) {
      setSearchStatus(null);
      showError(error instanceof Error ? error.message : 'Search failed.');
    }
  };

  const askPaper = async () => {
    if (!selectedPaper || !chatInput.trim() || isPaperThinking) return;
    const question = chatInput.trim();
    const userMessage: ChatMessage = { id: makeMessageId('user'), role: 'user', content: question };
    const assistantMessageId = makeMessageId('assistant');
    setChatInput('');
    setChatMessages((messages) => [...messages, userMessage, { id: assistantMessageId, role: 'assistant', content: '', pending: true }]);
    try {
      const response = await getJson<ChatMessage & { answer: string; citations: ChatMessage['citations'] }>('/api/chat/paper', {
        method: 'POST',
        body: JSON.stringify({ paper_id: selectedPaper.paper_id, query: question }),
      });
      setChatMessages((messages) =>
        messages.map((message) =>
          message.id === assistantMessageId
            ? { id: assistantMessageId, role: 'assistant', content: response.answer, citations: response.citations }
            : message,
        ),
      );
    } catch (error) {
      setChatMessages((messages) => messages.filter((message) => message.id !== assistantMessageId));
      showError(error instanceof Error ? error.message : 'Paper QA failed.');
    }
  };

  const openDownvotePanel = (message: ChatMessage) => {
    setFeedbackDrafts((drafts) => ({
      ...drafts,
      [message.id]: {
        open: true,
        reason: message.feedback?.reason ?? drafts[message.id]?.reason ?? 'incorrect',
        note: message.feedback?.note ?? drafts[message.id]?.note ?? '',
        saving: false,
      },
    }));
  };

  const updateFeedbackDraft = (messageId: string, patch: Partial<FeedbackDraft>) => {
    setFeedbackDrafts((drafts) => ({
      ...drafts,
      [messageId]: {
        reason: patch.reason ?? drafts[messageId]?.reason ?? 'incorrect',
        note: patch.note ?? drafts[messageId]?.note ?? '',
        saving: patch.saving ?? drafts[messageId]?.saving ?? false,
        open: true,
      },
    }));
  };

  const submitFeedback = async (message: ChatMessage, question: string, vote: 'up' | 'down', reason?: FeedbackReason, note = '') => {
    if (!selectedPaper) return;
    setFeedbackDrafts((drafts) => ({
      ...drafts,
      [message.id]: {
        open: vote === 'down',
        reason: reason ?? drafts[message.id]?.reason ?? 'incorrect',
        note,
        saving: true,
      },
    }));
    try {
      const submitted = await getJson<{ feedback: NonNullable<ChatMessage['feedback']> }>('/api/feedback', {
        method: 'POST',
        body: JSON.stringify({
          paper_id: selectedPaper.paper_id,
          answer_id: message.id,
          question,
          vote,
          reason: vote === 'down' ? reason : null,
          note: vote === 'down' ? note : '',
        }),
      });
      setChatMessages((messages) =>
        messages.map((item) => (item.id === message.id ? { ...item, feedback: submitted.feedback } : item)),
      );
      setFeedbackDrafts((drafts) => ({
        ...drafts,
        [message.id]: {
          open: false,
          reason: reason ?? drafts[message.id]?.reason ?? 'incorrect',
          note,
          saving: false,
        },
      }));
    } catch (error) {
      setFeedbackDrafts((drafts) => ({
        ...drafts,
        [message.id]: { ...(drafts[message.id] ?? { open: vote === 'down', reason: reason ?? 'incorrect', note, saving: false }), saving: false },
      }));
      showError(error instanceof Error ? error.message : 'Feedback could not be saved.');
    }
  };

  const saveSettings = async (patch: Partial<WorkbenchSettings>) => {
    const next = await getJson<WorkbenchSettings>('/api/settings', {
      method: 'PATCH',
      body: JSON.stringify(patch),
    });
    setSettings(next);
    return next;
  };

  const testLlm = async () => {
    setLlmTesting(true);
    setLlmTest(null);
    try {
      const payload = await getJson<{ ok: boolean; answer?: string; detail?: string }>('/api/settings/test-llm', {
        method: 'POST',
        body: JSON.stringify(qaDraft),
      });
      setLlmTest({ ok: true, message: payload.answer ?? 'ok' });
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Connection test failed.';
      setLlmTest({ ok: false, message });
      showError(message);
    } finally {
      setLlmTesting(false);
    }
  };

  const readyCount = papers.filter((paper) => paper.status === 'ready').length;
  const indexingCount = papers.filter((paper) => paper.status === 'indexing' || paper.status === 'queued').length;
  const failedCount = papers.filter((paper) => paper.status === 'failed').length;
  const isSearching = searchStatus?.status === 'running';
  const pageTitle =
    { upload: 'Add Papers', library: 'Library', search: 'Search Papers', workspace: 'Workspace', settings: 'Settings' }[activeView] ?? 'Search Papers';
  const selectedGpuIndexes = new Set(
    settings?.indexing_device === 'cuda'
      ? settings.cuda_visible_devices
          .split(',')
          .map((item) => Number(item.trim()))
          .filter(Number.isFinite)
      : [],
  );
  const toggleGpuForIndexing = (gpu: SystemGpu) => {
    const selected = selectedGpuIndexes.has(gpu.index);
    if (isGpuBusy(gpu) && !selected) return;

    const next = new Set(selectedGpuIndexes);
    if (selected) {
      next.delete(gpu.index);
    } else {
      next.add(gpu.index);
    }
    const cuda_visible_devices = Array.from(next)
      .sort((left, right) => left - right)
      .join(',');
    void saveSettings({ indexing_device: 'cuda', cuda_visible_devices });
  };

  return (
    <main className="min-h-screen bg-[#f6f8fb] text-slate-950">
      <div className="flex min-h-screen">
        <aside className={`hidden shrink-0 border-r border-slate-200 bg-white py-6 transition-all duration-200 lg:block ${isSidebarOpen ? 'w-72 px-5' : 'w-20 px-3'}`}>
          <div className={`mb-8 flex ${isSidebarOpen ? 'items-center justify-between' : 'flex-col items-center gap-3'}`}>
            <div className={`flex items-center gap-3 ${isSidebarOpen ? '' : 'justify-center'}`}>
              <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-slate-950 text-white">
                <Layers3 className="h-5 w-5" />
              </div>
              {isSidebarOpen ? (
              <div>
                <div className="text-lg font-semibold">{APP_NAME}</div>
              </div>
              ) : null}
            </div>
            <button
              type="button"
              onClick={() => setIsSidebarOpen((value) => !value)}
              className="flex h-9 w-9 items-center justify-center rounded-xl border border-slate-200 text-slate-500 transition hover:bg-slate-100 hover:text-slate-950"
              aria-label={isSidebarOpen ? 'Collapse navigation' : 'Expand navigation'}
              title={isSidebarOpen ? 'Collapse navigation' : 'Expand navigation'}
            >
              {isSidebarOpen ? <PanelLeftClose className="h-4 w-4" /> : <PanelLeftOpen className="h-4 w-4" />}
            </button>
          </div>

          <nav className="space-y-1">
            {[
              ['upload', 'Upload', UploadCloud],
              ['library', 'Library', Database],
              ['search', 'Search', Search],
              ['workspace', 'Workspace', BookOpen],
              ['settings', 'Settings', Settings],
            ].map(([id, label, Icon]) => (
              <button
                key={id as string}
                onClick={() => setActiveView(id as string)}
                className={`flex w-full items-center rounded-xl py-2.5 text-sm font-medium transition ${isSidebarOpen ? 'gap-3 px-3' : 'justify-center px-0'} ${
                  activeView === id ? 'bg-slate-950 text-white shadow-sm' : 'text-slate-600 hover:bg-slate-100 hover:text-slate-950'
                }`}
                title={label as string}
              >
                <Icon className="h-4 w-4" />
                {isSidebarOpen ? label as string : <span className="sr-only">{label as string}</span>}
              </button>
            ))}
          </nav>
        </aside>

        <section className="flex min-w-0 flex-1 flex-col">
          <header className="sticky top-0 z-20 border-b border-slate-200 bg-white/90 px-5 py-4 backdrop-blur lg:px-8">
            <h1 className="text-2xl font-semibold tracking-tight text-slate-950">{pageTitle}</h1>
          </header>

          <div className="flex-1 px-5 py-6 lg:px-8">
            {bannerError ? (
              <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/35 px-5 backdrop-blur-sm" role="alertdialog" aria-modal="true" aria-labelledby="workbench-error-title">
                <div className="w-full max-w-md rounded-3xl border border-rose-100 bg-white p-5 shadow-2xl">
                  <div className="mb-4 flex h-11 w-11 items-center justify-center rounded-2xl bg-rose-50 text-rose-600">
                    <AlertCircle className="h-5 w-5" />
                  </div>
                  <h2 id="workbench-error-title" className="text-lg font-semibold text-slate-950">Action failed</h2>
                  <p className="mt-2 text-sm leading-6 text-slate-600">{bannerError}</p>
                  <div className="mt-5 flex justify-end">
                    <button type="button" onClick={() => setBannerError(null)} className="rounded-xl bg-slate-950 px-4 py-2.5 text-sm font-semibold text-white hover:bg-slate-800">Dismiss</button>
                  </div>
                </div>
              </div>
            ) : null}
            <Tabs.Root value={activeView} onValueChange={setActiveView}>
              <Tabs.List className="mb-5 flex gap-2 overflow-x-auto lg:hidden">
                {['upload', 'library', 'search', 'workspace', 'settings'].map((id) => (
                  <Tabs.Trigger key={id} value={id} className="rounded-full border border-slate-200 bg-white px-4 py-2 text-sm capitalize data-[state=active]:bg-slate-950 data-[state=active]:text-white">
                    {id}
                  </Tabs.Trigger>
                ))}
              </Tabs.List>

              <Tabs.Content value="upload" className="outline-none">
                <div className="grid gap-5 xl:grid-cols-[1.25fr_0.75fr]">
                  <section
                    className={`flex min-h-[260px] flex-col items-center justify-center rounded-3xl border border-dashed bg-white p-8 text-center shadow-sm transition ${
                      isDraggingUpload ? 'border-blue-400 bg-blue-50' : 'border-slate-300'
                    }`}
                    onDragOver={(event) => {
                      event.preventDefault();
                      setIsDraggingUpload(true);
                    }}
                    onDragLeave={() => setIsDraggingUpload(false)}
                    onDrop={(event) => {
                      event.preventDefault();
                      setIsDraggingUpload(false);
                      void uploadFiles(event.dataTransfer.files);
                    }}
                  >
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept="application/pdf,.pdf"
                      multiple
                      className="hidden"
                      onChange={(event) => {
                        if (event.target.files) void uploadFiles(event.target.files);
                        event.currentTarget.value = '';
                      }}
                    />
                    <div className="mb-5 flex h-14 w-14 items-center justify-center rounded-2xl bg-slate-950 text-white">
                      <UploadCloud className="h-6 w-6" />
                    </div>
                    <h2 className="text-xl font-semibold text-slate-950">Add PDF papers</h2>
                    <p className="mt-2 text-sm text-slate-500">Drag files here, or choose PDFs from your computer.</p>
                    <button
                      type="button"
                      onClick={() => fileInputRef.current?.click()}
                      className="mt-6 rounded-xl bg-slate-950 px-5 py-3 text-sm font-semibold text-white transition hover:bg-slate-800"
                    >
                      Browse PDFs
                    </button>
                  </section>

                  <section className="space-y-4">
                    <div className="grid grid-cols-3 gap-3">
                      <MetricCard label="Ready" value={readyCount} icon={CheckCircle2} />
                      <MetricCard label="Indexing" value={indexingCount} icon={Loader2} />
                      <MetricCard label="Failed" value={failedCount} icon={AlertCircle} />
                    </div>
                    <div className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
                      <div className="mb-4 flex items-center justify-between">
                        <div>
                          <h3 className="font-semibold">Recent activity</h3>
                        </div>
                        <button onClick={() => void startIndexing(failedCount > 0)} aria-label="Start or retry indexing" className="rounded-full border border-slate-200 p-2 text-slate-500 hover:bg-slate-100">
                          <RefreshCw className="h-4 w-4" />
                        </button>
                      </div>
                      <div className="space-y-2">
                        {visibleJobs.length ? visibleJobs.map((job) => (
                          <div key={job.job_id} className="rounded-2xl border border-slate-200 p-3">
                            <div className="mb-2 flex items-start justify-between gap-3">
                              <div className="min-w-0">
                                <div className="truncate text-sm font-medium text-slate-950">{job.file_name}</div>
                                {job.status === 'failed' ? <div className="mt-1 text-xs text-rose-600">{job.message}</div> : null}
                                {job.status === 'queued' || job.status === 'running' ? <div className="mt-1 text-xs text-slate-500">{job.message}</div> : null}
                              </div>
                              <StatusBadge status={job.status} />
                            </div>
                            {job.status === 'queued' || job.status === 'running' ? (
                              <Progress.Root value={job.progress} className="h-1.5 overflow-hidden rounded-full bg-slate-100">
                                <Progress.Indicator className="h-full rounded-full bg-slate-950 transition-transform" style={{ transform: `translateX(-${100 - job.progress}%)` }} />
                              </Progress.Root>
                            ) : null}
                          </div>
                        )) : <div className="rounded-2xl border border-dashed border-slate-200 p-5 text-center text-sm text-slate-500">No recent indexing activity.</div>}
                        {jobs.length > visibleJobs.length ? (
                          <div className="pt-1 text-center text-xs text-slate-400">{jobs.length - visibleJobs.length} older items hidden</div>
                        ) : null}
                      </div>
                    </div>
                  </section>
                </div>
              </Tabs.Content>

              <Tabs.Content value="library" className="outline-none">
                <section className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
                  <div className="mb-5 flex justify-end">
                    <div className="flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-3 py-2">
                      <Filter className="h-4 w-4 text-slate-400" />
                      <input value={libraryFilter} onChange={(event) => setLibraryFilter(event.target.value)} id="library-filter" name="library-filter" placeholder="Search title or author" className="w-72 bg-transparent text-sm outline-none" />
                    </div>
                  </div>
                  <div className="overflow-hidden rounded-2xl border border-slate-200">
                    <table className="w-full border-collapse text-sm">
                      <thead className="bg-slate-50 text-left text-xs uppercase tracking-[0.12em] text-slate-500">
                        {table.getHeaderGroups().map((headerGroup) => (
                          <tr key={headerGroup.id}>
                            {headerGroup.headers.map((header) => (
                              <th key={header.id} className="border-b border-slate-200 px-4 py-3 font-semibold">
                                {flexRender(header.column.columnDef.header, header.getContext())}
                              </th>
                            ))}
                          </tr>
                        ))}
                      </thead>
                      <tbody>
                        {table.getRowModel().rows.map((row) => (
                          <tr key={row.id} className="border-b border-slate-100 last:border-0 hover:bg-slate-50">
                            {row.getVisibleCells().map((cell) => (
                              <td key={cell.id} className="px-4 py-3 align-top">
                                {flexRender(cell.column.columnDef.cell, cell.getContext())}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </section>
              </Tabs.Content>

              <Tabs.Content value="search" className="outline-none">
                <div className="mx-auto max-w-5xl space-y-5">
                  <section className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
                    <textarea
                      id="paper-search-query"
                      name="paper-search-query"
                      value={searchQuery}
                      onChange={(event) => setSearchQuery(event.target.value)}
                      className="min-h-28 w-full resize-none rounded-2xl border border-slate-200 bg-slate-50 p-4 text-sm leading-6 outline-none transition focus:border-blue-500 focus:bg-white"
                    />
                    <div className="mt-4 flex flex-col gap-3 lg:flex-row lg:items-end">
                      <RetrievalPicker value={retrievalMethod} onChange={setRetrievalMethod} />
                      <button
                        onClick={runSearch}
                        disabled={isSearching || !searchQuery.trim()}
                        className="flex h-12 w-full items-center justify-center gap-2 rounded-xl bg-slate-950 px-4 text-sm font-semibold text-white shadow-sm transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:bg-slate-400 lg:w-44"
                      >
                        {isSearching ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
                        {isSearching ? 'Searching' : 'Run search'}
                      </button>
                    </div>
                    {searchStatus && searchStatus.status !== 'completed' ? (
                      <div className="mt-4 rounded-2xl border border-blue-100 bg-blue-50/50 p-4">
                        <div className="mb-2 flex items-center justify-between text-sm">
                          <span className="font-medium text-slate-900">{searchStatus.stage}</span>
                          <span className="text-slate-500">{searchStatus.job_id === 'starting' ? 'running' : `${searchStatus.progress}%`}</span>
                        </div>
                        <Progress.Root value={searchStatus.progress} className="h-2 overflow-hidden rounded-full bg-slate-100">
                          <Progress.Indicator
                            className={`h-full rounded-full bg-blue-600 ${searchStatus.job_id === 'starting' ? 'animate-pulse' : 'transition-transform'}`}
                            style={{ transform: searchStatus.job_id === 'starting' ? 'none' : `translateX(-${100 - searchStatus.progress}%)` }}
                          />
                        </Progress.Root>
                        <p className="mt-2 text-xs text-slate-500">{searchStatus.message}</p>
                      </div>
                    ) : null}
                  </section>

                  <section className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
                    <div className="mb-4 flex items-center justify-between">
                      <div>
                        <h2 className="text-lg font-semibold">Results</h2>
                      </div>
                      <PanelRightOpen className="h-5 w-5 text-slate-400" />
                    </div>
                    <div className="space-y-3">
                      {searchResults.length ? searchResults.map((paper) => (
                        <article key={paper.paper_id} className="rounded-2xl border border-slate-200 p-4 transition hover:border-slate-300 hover:shadow-sm">
                          <div className="flex flex-col gap-4 md:flex-row">
                            <PaperPreview imageUrl={paper.preview_image_url} />
                            <div className="min-w-0 flex-1">
                              <div className="mb-2 flex flex-wrap items-center gap-2">
                                <span className="rounded-full bg-slate-950 px-2.5 py-1 text-xs font-semibold text-white">#{paper.rank}</span>
                              </div>
                              <h3 className="text-base font-semibold leading-6 text-slate-950">{paper.title}</h3>
                              <p className="mt-1 text-sm text-slate-500">{paper.authors.join(', ')}</p>
                            </div>
                            <button
                              onClick={() => {
                                setSelectedPaper(paper);
                                setActiveView('workspace');
                              }}
                              className="h-10 shrink-0 rounded-xl border border-slate-200 px-3 text-sm font-semibold text-slate-700 hover:bg-slate-100"
                            >
                              Read paper
                            </button>
                          </div>
                        </article>
                      )) : (
                        <div className="rounded-2xl border border-dashed border-slate-200 p-8 text-center text-sm text-slate-500">
Run search to show ranked papers.
                        </div>
                      )}
                    </div>
                  </section>
                </div>
              </Tabs.Content>

              <Tabs.Content value="workspace" className="outline-none">
                <div data-workspace-layout="true" className="grid min-h-[720px] gap-5 lg:grid-cols-[380px_minmax(0,1fr)]">
                  <aside className="flex min-h-[680px] flex-col overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-sm lg:sticky lg:top-24 lg:max-h-[calc(100vh-7.5rem)]">
                    <div className="flex-1 space-y-4 overflow-y-auto p-5">
                      {chatMessages.length ? (
                        <>
                          {chatMessages.map((message, index) => {
                            const question = questionForAssistant(chatMessages, index);
                            const feedbackDraft = feedbackDrafts[message.id];
                            return (
                              <div
                                key={message.id}
                                className={`rounded-2xl p-4 text-sm leading-6 ${
                                  message.role === 'user'
                                    ? 'ml-10 bg-slate-950 text-white'
                                    : 'mr-4 border border-slate-200 bg-slate-50 text-slate-700'
                                }`}
                              >
                                {message.pending ? (
                                  <div className="flex items-center gap-3">
                                    <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-white text-slate-700 shadow-sm">
                                      <Brain className="h-4 w-4" />
                                    </span>
                                    <span className="flex items-center gap-1" aria-label="Assistant is thinking">
                                      <span className="h-2 w-2 animate-bounce rounded-full bg-slate-400" style={{ animationDelay: '-0.2s' }} />
                                      <span className="h-2 w-2 animate-bounce rounded-full bg-slate-400" style={{ animationDelay: '-0.1s' }} />
                                      <span className="h-2 w-2 animate-bounce rounded-full bg-slate-400" />
                                    </span>
                                  </div>
                                ) : (
                                  message.content
                                )}
                                {message.citations?.length ? (
                                  <div className="mt-3 flex flex-wrap gap-2">
                                    {message.citations.map((citation) => (
                                      <button
                                        key={citation.evidence_id}
                                        onClick={() => jumpToEvidence(citation.evidence_id)}
                                        className="inline-flex items-center gap-1 rounded-full bg-white px-2.5 py-1 text-xs font-semibold text-blue-700 shadow-sm"
                                      >
                                        <Quote className="h-3 w-3" />{' '}
                                        {isGenericEvidenceHeading(citation.section_path[0])
                                          ? `pp. ${citation.page_start}-${citation.page_end}`
                                          : `${citation.section_path[0]} pp. ${citation.page_start}-${citation.page_end}`}
                                      </button>
                                    ))}
                                  </div>
                                ) : null}
                                {message.role === 'assistant' && !message.pending ? (
                                  <div className="mt-3 border-t border-slate-200 pt-3">
                                    <div className="flex items-center gap-2">
                                      <button
                                        type="button"
                                        aria-label="Mark this answer as helpful"
                                        onClick={() => void submitFeedback(message, question, 'up')}
                                        className={`inline-flex h-8 w-8 items-center justify-center rounded-full border transition ${
                                          message.feedback?.vote === 'up'
                                            ? 'border-emerald-200 bg-emerald-50 text-emerald-700'
                                            : 'border-slate-200 bg-white text-slate-500 hover:border-slate-300 hover:text-slate-950'
                                        }`}
                                      >
                                        <ThumbsUp className="h-4 w-4" />
                                      </button>
                                      <button
                                        type="button"
                                        aria-label="Mark this answer as not helpful"
                                        onClick={() => openDownvotePanel(message)}
                                        className={`inline-flex h-8 w-8 items-center justify-center rounded-full border transition ${
                                          message.feedback?.vote === 'down'
                                            ? 'border-blue-200 bg-blue-50 text-blue-700'
                                            : 'border-slate-200 bg-white text-slate-500 hover:border-slate-300 hover:text-slate-950'
                                        }`}
                                      >
                                        <ThumbsDown className="h-4 w-4" />
                                      </button>
                                      {message.feedback ? (
                                        <span className="inline-flex items-center gap-1 rounded-full px-2 py-1 text-xs font-medium text-slate-500">
                                          <CheckCircle2 className="h-3.5 w-3.5" /> Submitted
                                        </span>
                                      ) : null}
                                    </div>
                                    {feedbackDraft?.open ? (
                                      <div className="mt-3 rounded-2xl border border-slate-200 bg-white p-3 shadow-sm">
                                        <div className="grid grid-cols-2 gap-2">
                                          {feedbackReasons.map((reason) => (
                                            <button
                                              key={reason.value}
                                              type="button"
                                              onClick={() => updateFeedbackDraft(message.id, { reason: reason.value })}
                                              className={`rounded-xl border px-3 py-2 text-left text-xs font-medium transition ${
                                                feedbackDraft.reason === reason.value
                                                  ? 'border-blue-200 bg-blue-50 text-blue-700'
                                                  : 'border-slate-200 text-slate-600 hover:bg-slate-50'
                                              }`}
                                            >
                                              {reason.label}
                                            </button>
                                          ))}
                                        </div>
                                        <textarea
                                          id={`feedback-note-${message.id}`}
                                          name={`feedback-note-${message.id}`}
                                          aria-label="Optional feedback note"
                                          value={feedbackDraft.note}
                                          onChange={(event) => updateFeedbackDraft(message.id, { note: event.target.value })}
                                          placeholder="Add a note (optional)"
                                          className="mt-2 min-h-16 w-full resize-none rounded-xl border border-slate-200 px-3 py-2 text-xs leading-5 outline-none focus:border-blue-500"
                                        />
                                        <button
                                          type="button"
                                          disabled={feedbackDraft.saving}
                                          onClick={() => void submitFeedback(message, question, 'down', feedbackDraft.reason, feedbackDraft.note)}
                                          className="mt-2 flex h-9 w-full items-center justify-center rounded-xl bg-slate-950 px-3 text-xs font-semibold text-white transition hover:bg-slate-800 disabled:bg-slate-400"
                                        >
                                          {feedbackDraft.saving ? 'Submitting' : 'Submit'}
                                        </button>
                                      </div>
                                    ) : null}
                                  </div>
                                ) : null}
                              </div>
                            );
                          })}
                        </>
                      ) : (
                        <div className="rounded-2xl border border-dashed border-slate-200 p-5 text-sm leading-6 text-slate-500">
                          Ask a question about this paper.
                        </div>
                      )}
                      <div ref={chatEndRef} />
                    </div>

                    <div className="border-t border-slate-200 p-4">
                      <textarea
                        id="paper-chat-question"
                        name="paper-chat-question"
                        value={chatInput}
                        onChange={(event) => setChatInput(event.target.value)}
                        className="mb-3 min-h-24 w-full resize-none rounded-2xl border border-slate-200 bg-slate-50 p-3 text-sm leading-6 outline-none transition focus:border-blue-500 focus:bg-white"
                      />
                      <button
                        onClick={askPaper}
                        disabled={isPaperThinking || !chatInput.trim() || !selectedPaper}
                        className="flex w-full items-center justify-center gap-2 rounded-2xl bg-slate-950 px-4 py-3 text-sm font-semibold text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:bg-slate-400"
                      >
                        {isPaperThinking ? <Loader2 className="h-4 w-4 animate-spin" /> : <Brain className="h-4 w-4" />}
                        {isPaperThinking ? 'Thinking' : 'Ask paper'}
                      </button>
                    </div>
                  </aside>

                  <section className="overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-sm">
                    <div className="flex items-start justify-between gap-4 border-b border-slate-200 p-5">
                      <h2 className="min-w-0 text-xl font-semibold leading-7 text-slate-950">{viewer?.title ?? selectedPaper?.title ?? 'Select a paper'}</h2>
                      <div className="flex shrink-0 rounded-xl border border-slate-200 bg-slate-50 p-1">
                        {(['text', 'pdf'] as const).map((mode) => (
                          <button
                            key={mode}
                            type="button"
                            onClick={() => setPaperViewMode(mode)}
                            className={`h-8 rounded-lg px-3 text-sm font-semibold ${paperViewMode === mode ? 'bg-white text-slate-950 shadow-sm' : 'text-slate-500 hover:text-slate-950'}`}
                          >
                            {mode === 'text' ? 'Text' : 'PDF'}
                          </button>
                        ))}
                      </div>
                    </div>
                    <div ref={paperScrollRef} data-paper-scroll="true" className="max-h-[calc(100vh-10rem)] min-h-[620px] overflow-y-auto bg-[#fffdf8] px-5 py-6">
                      {paperViewMode === 'pdf' && (viewer?.paper_id || selectedPaper?.paper_id) ? (
                        <PaperPdfViewer
                          pdfUrl={`/api/papers/${encodeURIComponent(viewer?.paper_id ?? selectedPaper?.paper_id ?? '')}/pdf`}
                          scrollContainerRef={paperScrollRef}
                        />
                      ) : (
                        <div className="mx-auto max-w-3xl">
                          {visibleEvidenceUnits.length ? (
                            visibleEvidenceUnits.map((unit, index) => (
                              <div
                                key={`${unit.evidence_id}-${unit.page_start}-${unit.page_end}-${index}`}
                                data-evidence-card="true"
                                ref={(node) => {
                                  evidenceRefs.current[unit.evidence_id] = node;
                                  unit.aliasEvidenceIds.forEach((evidenceId) => {
                                    evidenceRefs.current[evidenceId] = node;
                                  });
                                }}
                                className={`mb-4 rounded-2xl border p-4 transition ${
                                  activeEvidenceId === unit.evidence_id || unit.aliasEvidenceIds.includes(activeEvidenceId ?? '')
                                    ? 'border-blue-300 bg-blue-50/70 ring-4 ring-blue-100'
                                    : 'border-slate-200 bg-white'
                                }`}
                              >
                                <div className="mb-2 flex items-center justify-between gap-3">
                                  {isGenericEvidenceHeading(unit.heading) ? (
                                    <div />
                                  ) : (
                                    <h3 className="font-semibold text-slate-950">{unit.heading}</h3>
                                  )}
                                  <span className="shrink-0 rounded-full bg-slate-100 px-2.5 py-1 text-xs text-slate-500">pp. {unit.page_start}-{unit.page_end}</span>
                                </div>
                                <EvidenceUnitBody unit={unit} />
                              </div>
                            ))
                          ) : (
                            <div className="rounded-2xl border border-dashed border-slate-200 bg-white p-8 text-center text-sm text-slate-500">
                              No parsed paper content available.
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  </section>
                </div>
              </Tabs.Content>

              <Tabs.Content value="settings" className="outline-none">
                <div className="max-w-5xl space-y-5">
                  <section className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
                    <h2 className="mb-5 text-lg font-semibold">Paper indexing</h2>
                    <div className="max-w-md">
                      <SelectControl
                        label="Indexing device"
                        value={settings?.indexing_device ?? 'auto'}
                        options={indexingDeviceOptions}
                        onChange={(indexing_device) => void saveSettings({ indexing_device })}
                      />
                    </div>
                    <div className="mt-5 flex items-center justify-between">
                      <div className="text-sm font-semibold text-slate-950">Detected GPUs</div>
                      <button aria-label="Refresh GPUs" onClick={loadGpus} className="rounded-full border border-slate-200 p-2 text-slate-500 hover:bg-slate-100">
                        <RefreshCw className="h-4 w-4" />
                      </button>
                    </div>
                    <div className="mt-3 grid gap-3 md:grid-cols-2">
                      {gpuPayload?.gpus.length ? (
                        gpuPayload.gpus.map((gpu) => (
                          <button
                            key={gpu.index}
                            type="button"
                            disabled={isGpuBusy(gpu) && !selectedGpuIndexes.has(gpu.index)}
                            onClick={() => toggleGpuForIndexing(gpu)}
                            className={`rounded-2xl border p-4 text-left transition ${
                              selectedGpuIndexes.has(gpu.index)
                                ? 'border-blue-300 bg-blue-50/40'
                                : 'border-slate-200 bg-white hover:border-slate-300'
                            } disabled:cursor-not-allowed disabled:hover:border-slate-200`}
                          >
                            <div className="flex items-start justify-between gap-3">
                              <div>
                                <div className="font-semibold text-slate-950">GPU {gpu.index}</div>
                                <div className="text-sm text-slate-500">{gpu.name}</div>
                              </div>
                              <GpuState gpu={gpu} selected={selectedGpuIndexes.has(gpu.index)} />
                            </div>
                          </button>
                        ))
                      ) : (
                        <div className="rounded-2xl border border-slate-200 p-4 text-sm text-slate-500">No NVIDIA GPU detected.</div>
                      )}
                    </div>
                  </section>

                  <section className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
                    <h2 className="mb-5 text-lg font-semibold">Paper QA provider</h2>
                    <form className="grid gap-4 md:grid-cols-2" onSubmit={(event) => event.preventDefault()}>
                      <label className="block md:col-span-2">
                        <span className="mb-2 block text-xs font-semibold uppercase tracking-[0.14em] text-slate-500">Base URL</span>
                        <input id="qa-base-url" name="qa-base-url" value={qaDraft.qa_base_url} onChange={(event) => setQaDraft({ ...qaDraft, qa_base_url: event.target.value })} placeholder="http://127.0.0.1:8017/v1" className="h-11 w-full rounded-xl border border-slate-200 px-3 text-sm outline-none" />
                      </label>
                      <label className="block">
                        <span className="mb-2 block text-xs font-semibold uppercase tracking-[0.14em] text-slate-500">Model ID</span>
                        <input id="qa-model-id" name="qa-model-id" value={qaDraft.qa_model} onChange={(event) => setQaDraft({ ...qaDraft, qa_model: event.target.value })} className="h-11 w-full rounded-xl border border-slate-200 px-3 text-sm outline-none" />
                      </label>
                      <label className="block">
                        <span className="mb-2 block text-xs font-semibold uppercase tracking-[0.14em] text-slate-500">Max context tokens</span>
                        <input id="qa-max-context-tokens" name="qa-max-context-tokens" type="number" value={qaDraft.max_context_tokens} onChange={(event) => setQaDraft({ ...qaDraft, max_context_tokens: Number(event.target.value) || 128000 })} className="h-11 w-full rounded-xl border border-slate-200 px-3 text-sm outline-none" />
                      </label>
                      <label className="block">
                        <span className="mb-2 block text-xs font-semibold uppercase tracking-[0.14em] text-slate-500">QA timeout seconds</span>
                        <input id="qa-timeout-seconds" name="qa-timeout-seconds" type="number" min={5} value={qaDraft.qa_timeout_seconds} onChange={(event) => setQaDraft({ ...qaDraft, qa_timeout_seconds: Number(event.target.value) || 120 })} className="h-11 w-full rounded-xl border border-slate-200 px-3 text-sm outline-none" />
                      </label>
                      <label className="block md:col-span-2">
                        <span className="mb-2 block text-xs font-semibold uppercase tracking-[0.14em] text-slate-500">API key</span>
                        <input id="qa-api-key" name="qa-api-key" type="password" autoComplete="off" value={qaDraft.qa_api_key} onChange={(event) => setQaDraft({ ...qaDraft, qa_api_key: event.target.value })} placeholder={settings?.qa_api_key_set ? 'Saved' : 'Optional for local vLLM'} className="h-11 w-full rounded-xl border border-slate-200 px-3 text-sm outline-none" />
                      </label>
                    </form>
                    <div className="mt-5 flex flex-wrap items-center gap-3">
                      <button
                        type="button"
                        onClick={async () => {
                          try {
                            await saveSettings(qaDraft);
                            setQaDraft({ ...qaDraft, qa_api_key: '' });
                            setLlmTest({ ok: true, message: 'Saved' });
                          } catch (error) {
                            showError(error instanceof Error ? error.message : 'Settings could not be saved.');
                          }
                        }}
                        className="rounded-xl bg-slate-950 px-4 py-2.5 text-sm font-semibold text-white hover:bg-slate-800"
                      >
                        Save provider
                      </button>
                      <button type="button" onClick={testLlm} disabled={llmTesting} className="rounded-xl border border-slate-200 px-4 py-2.5 text-sm font-semibold text-slate-700 hover:bg-slate-100 disabled:text-slate-400">
                        {llmTesting ? 'Testing...' : 'Test connection'}
                      </button>
                      {llmTest ? <span className={`text-sm font-medium ${llmTest.ok ? 'text-emerald-700' : 'text-rose-700'}`}>{llmTest.message}</span> : null}
                    </div>
                  </section>

                  <section className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
                    <h2 className="mb-5 text-lg font-semibold">Workspace</h2>
                    <div className="flex items-center justify-between gap-4 rounded-2xl border border-slate-200 p-4">
                      <div className="text-sm font-semibold text-slate-950">Citation navigation</div>
                      <Switch.Root checked={settings?.enable_citations ?? true} onCheckedChange={(enable_citations) => void saveSettings({ enable_citations })} className="relative h-7 w-12 rounded-full bg-slate-200 data-[state=checked]:bg-slate-950">
                        <Switch.Thumb className="block h-6 w-6 translate-x-0.5 rounded-full bg-white shadow transition data-[state=checked]:translate-x-[22px]" />
                      </Switch.Root>
                    </div>
                  </section>
                </div>
              </Tabs.Content>
            </Tabs.Root>
          </div>
        </section>
      </div>
    </main>
  );
}
