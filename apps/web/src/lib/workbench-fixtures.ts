export type RetrievalMethod =
  | 'bm25_full_text'
  | 'colbertv2'
  | 'spladepp'
  | 'hybrid_bm25_colbertv2'
  | 'hybrid_bm25_spladepp';

export type QaModel = string;
export type IndexingDevice = 'auto' | 'cpu' | 'cuda';

export type LibraryPaper = {
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

export type LibraryJob = {
  job_id: string;
  kind: 'upload' | 'parse' | 'index';
  file_name: string;
  status: 'queued' | 'running' | 'ready' | 'failed';
  progress: number;
  message: string;
};

export type SearchJob = {
  job_id: string;
  query: string;
  retrieval_method: RetrievalMethod;
  qa_model: QaModel;
  corpus_scope: string;
  created_at: number;
};

export type SearchResult = LibraryPaper & {
  rank: number;
  score: number;
  retrieval_method: RetrievalMethod;
  matched_terms: string[];
  reason: string;
};

export type EvidenceUnit = {
  evidence_id: string;
  heading: string;
  page_start: number;
  page_end: number;
  text: string;
};

export type PaperViewer = {
  paper_id: string;
  title: string;
  abstract: string;
  evidence_units: EvidenceUnit[];
};

export type WorkbenchSettings = {
  library_path: string;
  retrieval_method: RetrievalMethod;
  qa_model: QaModel;
  qa_base_url: string;
  qa_api_key: string;
  qa_api_key_set?: boolean;
  max_context_tokens: number;
  enable_citations: boolean;
  indexing_device: IndexingDevice;
  cuda_visible_devices: string;
};

type WorkbenchStore = {
  papers: LibraryPaper[];
  libraryJobs: LibraryJob[];
  searchJobs: Map<string, SearchJob>;
  settings: WorkbenchSettings;
};

const papers: LibraryPaper[] = [
  {
    paper_id: 'cof-co2-water-2024',
    title: 'Donor-Acceptor Covalent Organic Frameworks for Coupled CO2 Reduction and Water Oxidation',
    authors: ['Y. Liu', 'M. Chen', 'R. Zhang'],
    year: 2024,
    venue: 'JACS',
    pages: 14,
    figures: 6,
    status: 'ready',
    tags: ['COF', 'CO2 reduction', 'water oxidation'],
    updated_at: '2026-07-06T18:42:00Z',
    preview_label: 'COF photocatalysis scheme',
    abstract:
      'A donor-acceptor covalent organic framework is reported for paired photocatalytic CO2 reduction and water oxidation under visible light.',
  },
  {
    paper_id: 'porphyrin-cof-charge-2023',
    title: 'Porphyrin-Based COF Photocatalysts with Extended Charge Separation for Solar Fuel Production',
    authors: ['A. Kumar', 'S. Wang'],
    year: 2023,
    venue: 'Angew. Chem.',
    pages: 11,
    figures: 5,
    status: 'ready',
    tags: ['photocatalysis', 'porphyrin', 'charge separation'],
    updated_at: '2026-07-06T18:12:00Z',
    preview_label: 'Charge separation diagram',
    abstract:
      'The study tunes porphyrin linkers in crystalline frameworks to improve charge separation and photocatalytic fuel-forming reactions.',
  },
  {
    paper_id: 'cof-h2o2-orr-2025',
    title: 'Covalent Organic Framework Photocatalysts for Two-Electron Oxygen Reduction to Hydrogen Peroxide',
    authors: ['L. Sun', 'J. Zhao', 'K. Patel'],
    year: 2025,
    venue: 'Nature Catalysis',
    pages: 16,
    figures: 7,
    status: 'ready',
    tags: ['COF', 'oxygen reduction', 'H2O2'],
    updated_at: '2026-07-06T17:54:00Z',
    preview_label: 'Oxygen reduction pathway',
    abstract:
      'A conjugated framework is optimized for selective two-electron oxygen reduction and peroxide production in water.',
  },
  {
    paper_id: 'tio2-vacancy-benzene-2022',
    title: 'Oxygen-Vacancy-Rich TiO2 for Hydroperoxy-Mediated Selective Benzene Oxidation',
    authors: ['D. Park', 'E. Rossi'],
    year: 2022,
    venue: 'ACS Catalysis',
    pages: 12,
    figures: 4,
    status: 'ready',
    tags: ['TiO2', 'oxygen vacancy', 'benzene oxidation'],
    updated_at: '2026-07-05T22:33:00Z',
    preview_label: 'Vacancy active site',
    abstract:
      'Defect-rich TiO2 surfaces activate hydroperoxy intermediates and improve selective oxidation under mild photocatalytic conditions.',
  },
  {
    paper_id: 'mof-cof-junction-2024',
    title: 'Internal Electric Fields in MOF/COF Heterojunctions for Photocatalytic Water Splitting',
    authors: ['N. Smith', 'Q. Li', 'H. Tang'],
    year: 2024,
    venue: 'Chem',
    pages: 13,
    figures: 5,
    status: 'indexing',
    tags: ['MOF', 'COF', 'heterojunction'],
    updated_at: '2026-07-06T19:02:00Z',
    preview_label: 'Heterojunction field map',
    abstract:
      'A mixed MOF/COF interface is designed to create an internal electric field for charge separation during water splitting.',
  },
];

const initialJobs: LibraryJob[] = [
  {
    job_id: 'job-upload-001',
    kind: 'index',
    file_name: 'mof-cof-junction-2024.pdf',
    status: 'running',
    progress: 68,
    message: 'Building full-text retrieval records',
  },
  {
    job_id: 'job-upload-002',
    kind: 'parse',
    file_name: 'ligand-screening-notes.pdf',
    status: 'failed',
    progress: 31,
    message: 'PDF text layer is incomplete. Retry after OCR.',
  },
];

const defaultSettings: WorkbenchSettings = {
  library_path: '/mnt/data2/users/x6k/demo-chem-library',
  retrieval_method: 'hybrid_bm25_colbertv2',
  qa_model: 'gpt-5.4-mini',
  qa_base_url: '',
  qa_api_key: '',
  max_context_tokens: 128000,
  enable_citations: true,
  indexing_device: 'auto',
  cuda_visible_devices: '',
};

const globalStore = globalThis as typeof globalThis & { __chemverifyWorkbench?: WorkbenchStore };

export function getWorkbenchStore(): WorkbenchStore {
  if (!globalStore.__chemverifyWorkbench) {
    globalStore.__chemverifyWorkbench = {
      papers: [...papers],
      libraryJobs: [...initialJobs],
      searchJobs: new Map(),
      settings: { ...defaultSettings },
    };
  }
  return globalStore.__chemverifyWorkbench;
}

export function publicSettings(settings = getWorkbenchStore().settings): WorkbenchSettings {
  return { ...settings, qa_api_key: '', qa_api_key_set: Boolean(settings.qa_api_key) };
}

export function createUploadJob(fileName: string): LibraryJob {
  const store = getWorkbenchStore();
  const device = store.settings.indexing_device === 'cuda' ? `CUDA ${store.settings.cuda_visible_devices || 'auto'}` : store.settings.indexing_device.toUpperCase();
  const job: LibraryJob = {
    job_id: `upload-${Date.now()}`,
    kind: 'upload',
    file_name: fileName,
    status: 'ready',
    progress: 100,
    message: `Queued for indexing on ${device}.`,
  };
  store.libraryJobs = [job, ...store.libraryJobs].slice(0, 8);
  const paperId = fileName.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 48) || `paper-${Date.now()}`;
  if (!store.papers.some((paper) => paper.paper_id === paperId)) {
    store.papers = [
      {
        paper_id: paperId,
        title: fileName.replace(/\.pdf$/i, '').replace(/[-_]+/g, ' '),
        authors: ['Uploaded Author'],
        year: 2026,
        venue: 'Local Library',
        pages: 8,
        figures: 2,
        status: 'queued',
        tags: ['uploaded'],
        updated_at: new Date().toISOString(),
        preview_label: 'Uploaded PDF',
        abstract: 'Fixture metadata for an uploaded local PDF. Real parsing will populate this record later.',
      },
      ...store.papers,
    ];
  }
  return job;
}

export function createSearchJob(input: { query: string; retrieval_method?: RetrievalMethod; qa_model?: QaModel; corpus_scope?: string }): SearchJob {
  const store = getWorkbenchStore();
  const job: SearchJob = {
    job_id: `search-${Date.now()}`,
    query: input.query,
    retrieval_method: input.retrieval_method ?? store.settings.retrieval_method,
    qa_model: input.qa_model ?? store.settings.qa_model,
    corpus_scope: input.corpus_scope ?? 'ready-papers',
    created_at: Date.now(),
  };
  store.searchJobs.set(job.job_id, job);
  return job;
}

export function searchJobStatus(job: SearchJob) {
  const elapsed = Date.now() - job.created_at;
  const progress = Math.min(100, Math.floor(elapsed / 28));
  const stages = ['Reading query', 'Ranking local papers', 'Preparing result cards'];
  const stage = stages[Math.min(stages.length - 1, Math.floor(progress / 38))];
  return {
    job_id: job.job_id,
    status: progress >= 100 ? 'completed' : 'running',
    stage,
    message: progress >= 100 ? 'Search results are ready.' : `${stage} with ${retrievalLabel(job.retrieval_method)}.`,
    progress,
    created_at: new Date(job.created_at).toISOString(),
  };
}

export function retrievalLabel(method: RetrievalMethod): string {
  return {
    bm25_full_text: 'BM25 full text',
    colbertv2: 'ColBERTv2',
    spladepp: 'SPLADE++',
    hybrid_bm25_colbertv2: 'Hybrid BM25 + ColBERTv2',
    hybrid_bm25_spladepp: 'Hybrid BM25 + SPLADE++',
  }[method];
}

export function rankedResults(job: SearchJob): SearchResult[] {
  const store = getWorkbenchStore();
  const boostByMethod: Record<RetrievalMethod, number[]> = {
    bm25_full_text: [0.91, 0.86, 0.78, 0.63, 0.58],
    colbertv2: [0.89, 0.84, 0.82, 0.61, 0.55],
    spladepp: [0.87, 0.79, 0.76, 0.68, 0.57],
    hybrid_bm25_colbertv2: [0.96, 0.88, 0.83, 0.69, 0.61],
    hybrid_bm25_spladepp: [0.94, 0.87, 0.81, 0.71, 0.60],
  };
  return store.papers
    .map((paper, index) => ({
      ...paper,
      rank: index + 1,
      score: boostByMethod[job.retrieval_method][index] ?? Math.max(0.25, 0.55 - index * 0.03),
      retrieval_method: job.retrieval_method,
      matched_terms: paper.tags.slice(0, 3),
      reason:
        paper.paper_id === 'cof-co2-water-2024'
          ? 'Matches donor-acceptor COF, CO2 reduction, and water oxidation constraints in the query.'
          : 'Shares material or reaction evidence with the query and remains available for paper-level QA.',
    }))
    .sort((a, b) => b.score - a.score)
    .map((paper, index) => ({ ...paper, rank: index + 1 }));
}

export function paperViewer(paperId: string): PaperViewer {
  const store = getWorkbenchStore();
  const paper = store.papers.find((item) => item.paper_id === paperId) ?? store.papers[0];
  return {
    paper_id: paper.paper_id,
    title: paper.title,
    abstract: paper.abstract,
    evidence_units: [
      {
        evidence_id: `${paper.paper_id}-ev1`,
        heading: 'Abstract',
        page_start: 1,
        page_end: 1,
        text: paper.abstract,
      },
      {
        evidence_id: `${paper.paper_id}-ev2`,
        heading: 'Photocatalytic system',
        page_start: 3,
        page_end: 4,
        text:
          'The paper describes a structured chemistry system, reports the material design, and connects the reaction setting to measured photocatalytic performance.',
      },
      {
        evidence_id: `${paper.paper_id}-ev3`,
        heading: 'Main finding',
        page_start: 8,
        page_end: 9,
        text:
          'The main result is that careful control over donor and acceptor units improves charge separation and supports the target chemical transformation.',
      },
    ],
  };
}
