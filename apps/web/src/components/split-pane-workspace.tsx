'use client';

import { useCallback, useEffect, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent, type RefObject, type UIEvent } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import {
  BookMarked,
  BookOpen,
  ChevronDown,
  ChevronLeft,
  ChevronUp,
  Clock,
  ExternalLink,
  FileText,
  GripVertical,
  Quote,
  Search,
  Send,
  Sparkles,
  X,
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import rehypeKatex from 'rehype-katex';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';

import { ChatMessage } from '@/app/page';
import { ActivityMark, useTransientActivityMode } from '@/components/activity-mark';
import { PaperPdfViewer, paperPdfPageDomId } from '@/components/paper-pdf-viewer';
import { chatWithPaper, fetchPaperViewer, paperZoteroPageUrl } from '@/lib/client-api';
import {
  ManuscriptOutlineRail,
  ManuscriptOutlineSheet,
  type ManuscriptOutlineItem,
} from '@/components/manuscript-outline';
import {
  loadWorkspaceUiState,
  saveDesktopSplitPct,
  saveMobilePane,
  saveRationaleOpen,
  type WorkspaceMobilePane,
} from '@/lib/workspace-ui-state';
import type {
  EvidenceNavigationTarget,
  PaperChatCitation,
  PaperResult,
  PaperViewerBlock,
  PaperViewerReference,
  PaperViewerResponse,
  ViewerMode,
} from '@/lib/types';

type SplitPaneWorkspaceProps = {
  paper: PaperResult;
  initialChatHistory: ChatMessage[];
  onChatHistoryUpdate: (messages: ChatMessage[]) => void;
  onBack: () => void;
  onOpenGlobalSearch: () => void;
};

type ManuscriptPaneProps = {
  viewer: PaperViewerResponse | null;
  isLoading: boolean;
  viewerMode: ViewerMode;
  scrollProgress: number;
  activeBlock: PaperViewerBlock | null;
  previewBlockId: string | null;
  activeSectionCueBlockId: string | null;
  previewSectionCueBlockId: string | null;
  linkedSectionCueBlockId: string | null;
  linkedCueBlockId: string | null;
  flashBlockId: string | null;
  activePdfTarget: EvidenceNavigationTarget['pdf_target'] | null;
  previewPdfTarget: EvidenceNavigationTarget['pdf_target'] | null;
  linkedPdfTarget: EvidenceNavigationTarget['pdf_target'] | null;
  outlineItems: ManuscriptOutlineItem[];
  activeSectionBlockId: string | null;
  showOutlineRail: boolean;
  outlineSheetOpen: boolean;
  manuscriptScrollContainerRef: RefObject<HTMLDivElement | null>;
  pdfScrollContainerRef: RefObject<HTMLDivElement | null>;
  onClearActiveEvidence: () => void;
  onViewerModeChange: (nextMode: ViewerMode) => void;
  onPdfDocumentStateChange: (state: { isReady: boolean; pageCount: number }) => void;
  onOpenOutlineSheet: () => void;
  onOutlineSheetClose: () => void;
  onSelectOutlineItem: (blockId: string) => void;
  onManuscriptScroll: (event: UIEvent<HTMLDivElement>) => void;
  onPdfScroll: (event: UIEvent<HTMLDivElement>) => void;
};

type WorkspaceViewportMode = 'mobile' | 'tablet' | 'desktop';
type EvidenceHighlightMode = 'none' | 'active' | 'preview' | 'linked';

const MOBILE_WORKSPACE_BREAKPOINT = 900;
const DESKTOP_RESIZABLE_BREAKPOINT = 1280;
const TABLET_CHAT_WIDTH_PCT = 42;
const MIN_DESKTOP_CHAT_WIDTH_PCT = 28;
const MAX_DESKTOP_CHAT_WIDTH_PCT = 72;

function getWorkspaceViewportMode(width: number, height: number, isCoarsePointer: boolean): WorkspaceViewportMode {
  const shortestSide = Math.min(width, height);
  if (width < MOBILE_WORKSPACE_BREAKPOINT || (isCoarsePointer && shortestSide < 820)) {
    return 'mobile';
  }
  if (width < DESKTOP_RESIZABLE_BREAKPOINT) {
    return 'tablet';
  }
  return 'desktop';
}

function clampDesktopChatWidth(value: number): number {
  if (!Number.isFinite(value)) {
    return 40;
  }
  return Math.max(MIN_DESKTOP_CHAT_WIDTH_PCT, Math.min(MAX_DESKTOP_CHAT_WIDTH_PCT, value));
}

function viewerBlockDomId(blockId: string): string {
  return `viewer-block-${encodeURIComponent(blockId)}`;
}

function formatSectionLabel(sectionPath: string[]): string | null {
  const cleaned = sectionPath.map((item) => item.trim()).filter(Boolean);
  if (cleaned.length === 0) {
    return null;
  }
  return cleaned.slice(-2).join(' / ');
}

function formatCitationPageLabel(citation: PaperChatCitation): string {
  if (citation.page_end > citation.page_start) {
    return `P.${citation.page_start}-${citation.page_end}`;
  }
  return `P.${citation.page_start}`;
}

function resolveReferenceHref(reference: PaperViewerReference): string | null {
  if (reference.doi) {
    return `https://doi.org/${reference.doi}`;
  }
  if (reference.arxiv_id) {
    return `https://arxiv.org/abs/${reference.arxiv_id}`;
  }
  if (!reference.url) {
    return null;
  }
  if (/^https?:\/\//i.test(reference.url)) {
    return reference.url;
  }
  return `https://${reference.url}`;
}

function renderStructuredTable(block: PaperViewerBlock) {
  const rows = block.table?.rows ?? [];
  if (rows.length === 0) {
    return (
      <div className="overflow-x-auto whitespace-pre-wrap rounded-[1.25rem] border border-slate-200/90 bg-white px-4 py-3 text-[13px] leading-6 text-slate-700">
        {block.text}
      </div>
    );
  }

  const hasExplicitHeader = rows.some((row) => row.cells.some((cell) => Boolean(cell.is_header)));

  return (
    <div className="overflow-x-auto rounded-[1.25rem] border border-slate-200/90 bg-white">
      <table className="min-w-full border-collapse text-left text-[13px] leading-6 text-slate-700">
        <tbody>
          {rows.map((row, rowIndex) => (
            <tr
              key={`${block.block_id}-row-${rowIndex}`}
              className={rowIndex === 0 ? 'bg-slate-50/80' : 'bg-white'}
            >
              {row.cells.map((cell, cellIndex) => {
                const Tag = cell.is_header || (!hasExplicitHeader && rowIndex === 0) ? 'th' : 'td';
                return (
                  <Tag
                    key={`${block.block_id}-cell-${rowIndex}-${cellIndex}`}
                    colSpan={cell.colspan || undefined}
                    rowSpan={cell.rowspan || undefined}
                    className={`border border-slate-200 px-3 py-2 align-top whitespace-pre-wrap ${
                      Tag === 'th' ? 'font-semibold text-slate-800' : 'font-normal text-slate-700'
                    }`}
                  >
                    {cell.text || ' '}
                  </Tag>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ViewerBlock({
  block,
  highlightMode,
  shouldFlash,
}: {
  block: PaperViewerBlock;
  highlightMode: EvidenceHighlightMode;
  shouldFlash: boolean;
}) {
  const highlightClassName =
    highlightMode === 'active'
      ? 'evidence-active-block'
      : highlightMode === 'preview'
        ? 'evidence-preview-block'
        : highlightMode === 'linked'
          ? 'evidence-linked-block'
          : '';
  const baseClassName = `relative isolate scroll-mt-28 overflow-hidden rounded-[1.75rem] transition-[border-color,background-color,box-shadow,transform] duration-500 ${highlightClassName} ${
    shouldFlash ? 'evidence-flash-block' : ''
  }`;

  if (block.block_type === 'section_heading') {
    return (
      <section
        id={viewerBlockDomId(block.block_id)}
        data-block-id={block.block_id}
        className={`${baseClassName} border-b border-slate-200/85 px-2 pb-6 pt-10`}
      >
        <h2 className="text-[2.28rem] font-black tracking-[-0.04em] text-slate-950">{block.text}</h2>
      </section>
    );
  }

  if (block.block_type === 'figure_block') {
    return (
      <article
        id={viewerBlockDomId(block.block_id)}
        data-block-id={block.block_id}
        className={`${baseClassName} border border-slate-200/80 bg-[linear-gradient(180deg,rgba(248,250,252,0.96),rgba(255,255,255,0.98))] px-6 py-6 shadow-scholar-sm`}
      >
        {block.image_url ? (
          <a
            href={block.image_url}
            target="_blank"
            rel="noreferrer"
            className="block overflow-hidden rounded-[1.35rem] border border-slate-200 bg-white"
          >
            <img
              src={block.image_url}
              alt={block.caption || 'Figure'}
              loading="lazy"
              decoding="async"
              className="w-full transition-transform duration-700 hover:scale-[1.01]"
            />
          </a>
        ) : null}
        {block.caption ? (
          <div className="mt-4 text-[0.72rem] font-bold uppercase tracking-[0.24em] text-slate-500">{block.caption}</div>
        ) : null}
        {block.text ? (
          <p className="font-scholar mt-3 whitespace-pre-wrap text-[1rem] italic leading-8 text-slate-600">
            &ldquo;{block.text}&rdquo;
          </p>
        ) : null}
        {block.footnote ? <div className="mt-3 text-[0.8rem] leading-6 text-slate-400">{block.footnote}</div> : null}
      </article>
    );
  }

  if (block.block_type === 'table_block') {
    return (
      <article
        id={viewerBlockDomId(block.block_id)}
        data-block-id={block.block_id}
        className={`${baseClassName} border border-slate-200/80 bg-white/98 px-5 py-5 shadow-scholar-sm`}
      >
        {block.caption ? (
          <div className="mb-3 text-[0.74rem] font-bold uppercase tracking-[0.24em] text-slate-500">{block.caption}</div>
        ) : null}
        {renderStructuredTable(block)}
      </article>
    );
  }

  if (block.block_type === 'equation_block') {
    return (
      <article
        id={viewerBlockDomId(block.block_id)}
        data-block-id={block.block_id}
        className={`${baseClassName} border border-slate-200/70 bg-[#FCFBF9] px-8 py-6 shadow-[inset_0_1px_0_rgba(255,255,255,0.8)]`}
      >
        <pre className="overflow-x-auto whitespace-pre-wrap text-[15px] leading-loose text-indigo-950">{block.text}</pre>
      </article>
    );
  }

  return (
    <article
      id={viewerBlockDomId(block.block_id)}
      data-block-id={block.block_id}
      className={`${baseClassName} border border-transparent px-3 py-2`}
    >
      <div className="whitespace-pre-wrap font-scholar text-[1.1rem] leading-[2.05] text-slate-700/95 selection:bg-indigo-100/50">
        {block.text}
      </div>
    </article>
  );
}

function ReferencesAppendix({ references }: { references: PaperViewerReference[] }) {
  const [expanded, setExpanded] = useState(false);
  const visibleReferences = expanded ? references : references.slice(0, 12);

  return (
    <section className="mt-14 border-t border-slate-200/85 pt-10">
      <div className="mb-6 flex items-start justify-between gap-4">
        <div>
          <div className="text-[0.72rem] font-bold uppercase tracking-[0.24em] text-slate-400">Appendix</div>
          <h3 className="mt-2 text-[1.6rem] font-black tracking-tight text-slate-950">References</h3>
        </div>
        {references.length > 12 ? (
          <button
            type="button"
            onClick={() => setExpanded((current) => !current)}
            className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-4 py-2 text-[0.72rem] font-bold uppercase tracking-[0.2em] text-slate-500 transition hover:border-indigo-200 hover:text-indigo-600"
          >
            {expanded ? 'Show less' : `Show all (${references.length})`}
            {expanded ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
          </button>
        ) : null}
      </div>

      <div className="space-y-4">
        {visibleReferences.map((reference) => {
          const href = resolveReferenceHref(reference);
          return (
            <article
              key={`reference-${reference.ordinal}`}
              className="rounded-[1.35rem] border border-slate-200/85 bg-white/84 px-5 py-4 shadow-scholar-sm"
            >
              <div className="flex items-start gap-4">
                <div className="flex h-8 min-w-8 items-center justify-center rounded-full bg-slate-100 text-[0.72rem] font-bold text-slate-500">
                  {reference.ordinal}
                </div>
                <div className="min-w-0 flex-1">
                  <p className="font-scholar text-[1rem] leading-8 text-slate-700">{reference.raw_text}</p>
                  <div className="mt-3 flex flex-wrap items-center gap-2">
                    {reference.year ? (
                      <span className="rounded-full border border-slate-200 bg-slate-50 px-2.5 py-1 text-[0.68rem] font-semibold uppercase tracking-[0.18em] text-slate-500">
                        {reference.year}
                      </span>
                    ) : null}
                    {reference.arxiv_id ? (
                      <a
                        href={`https://arxiv.org/abs/${reference.arxiv_id}`}
                        target="_blank"
                        rel="noreferrer"
                        className="inline-flex items-center gap-1 rounded-full border border-indigo-100 bg-indigo-50 px-2.5 py-1 text-[0.68rem] font-semibold uppercase tracking-[0.16em] text-indigo-600 transition hover:bg-indigo-100"
                      >
                        arXiv
                        <ExternalLink className="h-3 w-3" />
                      </a>
                    ) : null}
                    {reference.doi ? (
                      <a
                        href={`https://doi.org/${reference.doi}`}
                        target="_blank"
                        rel="noreferrer"
                        className="inline-flex items-center gap-1 rounded-full border border-slate-200 bg-white px-2.5 py-1 text-[0.68rem] font-semibold uppercase tracking-[0.16em] text-slate-600 transition hover:border-indigo-200 hover:text-indigo-600"
                      >
                        DOI
                        <ExternalLink className="h-3 w-3" />
                      </a>
                    ) : null}
                    {!reference.doi && !reference.arxiv_id && href ? (
                      <a
                        href={href}
                        target="_blank"
                        rel="noreferrer"
                        className="inline-flex items-center gap-1 rounded-full border border-slate-200 bg-white px-2.5 py-1 text-[0.68rem] font-semibold uppercase tracking-[0.16em] text-slate-600 transition hover:border-indigo-200 hover:text-indigo-600"
                      >
                        Link
                        <ExternalLink className="h-3 w-3" />
                      </a>
                    ) : null}
                  </div>
                </div>
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );
}

function ManuscriptLoadingState() {
  return (
    <div className="mx-auto max-w-[980px] pb-28">
      <div className="manuscript-canvas px-8 py-9 sm:px-10 sm:py-10">
        <div className="space-y-4">
          <div className="skeleton-pulse h-3 w-28 rounded-full" />
          <div className="skeleton-pulse h-10 w-3/4 rounded-2xl" />
          <div className="skeleton-pulse h-6 w-1/2 rounded-2xl" />
        </div>
        <div className="mt-10 grid gap-4">
          <div className="skeleton-pulse h-7 w-40 rounded-2xl" />
          <div className="skeleton-pulse h-5 w-full rounded-2xl" />
          <div className="skeleton-pulse h-5 w-full rounded-2xl" />
          <div className="skeleton-pulse h-5 w-[92%] rounded-2xl" />
          <div className="skeleton-pulse h-5 w-[88%] rounded-2xl" />
        </div>
        <div className="mt-10 grid gap-4">
          <div className="skeleton-pulse h-7 w-52 rounded-2xl" />
          <div className="skeleton-pulse h-32 w-full rounded-[1.6rem]" />
        </div>
      </div>
    </div>
  );
}

function ManuscriptPane({
  viewer,
  isLoading,
  viewerMode,
  scrollProgress,
  activeBlock,
  previewBlockId,
  activeSectionCueBlockId,
  previewSectionCueBlockId,
  linkedSectionCueBlockId,
  linkedCueBlockId,
  flashBlockId,
  activePdfTarget,
  previewPdfTarget,
  linkedPdfTarget,
  outlineItems,
  activeSectionBlockId,
  showOutlineRail,
  outlineSheetOpen,
  manuscriptScrollContainerRef,
  pdfScrollContainerRef,
  onClearActiveEvidence,
  onViewerModeChange,
  onPdfDocumentStateChange,
  onOpenOutlineSheet,
  onOutlineSheetClose,
  onSelectOutlineItem,
  onManuscriptScroll,
  onPdfScroll,
}: ManuscriptPaneProps) {
  const byline = useMemo(() => {
    const authors = viewer?.display_header?.authors_structured ?? [];
    return authors.map((author) => author.name).filter(Boolean).join(' · ');
  }, [viewer]);

  const affiliationLine = useMemo(() => {
    const affiliations = viewer?.display_header?.affiliations ?? [];
    return affiliations.join(' · ');
  }, [viewer]);

  const activeEvidenceSummary = useMemo(() => {
    if (!activeBlock) {
      return null;
    }
    const sectionLabel = formatSectionLabel(activeBlock.section_path);
    const pageLabel = activeBlock.page_start > 0 ? `p.${activeBlock.page_start}` : null;
    return [sectionLabel, pageLabel].filter(Boolean).join(' • ');
  }, [activeBlock]);

  return (
    <div className="relative flex h-full flex-col overflow-hidden bg-[linear-gradient(180deg,#fdfefe_0%,#f6f9fd_100%)]">
      <div className="pointer-events-none absolute inset-x-0 top-0 h-36 bg-[radial-gradient(circle_at_top,rgba(99,102,241,0.11),transparent_62%)]" />
      <div
        className="absolute left-0 top-0 z-[60] h-1 bg-indigo-600 transition-all duration-300"
        style={{ width: `${scrollProgress}%` }}
      />

      <div className="glass-header z-10 border-b border-slate-200/70 px-7 py-4 shadow-sm">
        <div className="flex items-start justify-between gap-5">
          <div className="min-w-0">
            <h2 className="line-clamp-2 text-[1.05rem] font-black leading-tight text-slate-950" title={viewer?.title || ''}>
              {viewer?.title || 'Manuscript view'}
            </h2>
          </div>
          <div className="flex items-start gap-2">
            <div className="inline-flex rounded-full border border-slate-200 bg-white p-1 shadow-scholar-sm">
              <button
                type="button"
                onClick={() => onViewerModeChange('manuscript')}
                className={`inline-flex items-center gap-2 rounded-full px-3 py-1.5 text-[0.66rem] font-bold uppercase tracking-[0.18em] transition ${
                  viewerMode === 'manuscript' ? 'bg-slate-950 text-white' : 'text-slate-500 hover:text-indigo-600'
                }`}
              >
                <BookOpen className="h-3.5 w-3.5" />
                Manuscript
              </button>
              <button
                type="button"
                onClick={() => onViewerModeChange('pdf')}
                className={`inline-flex items-center gap-2 rounded-full px-3 py-1.5 text-[0.66rem] font-bold uppercase tracking-[0.18em] transition ${
                  viewerMode === 'pdf' ? 'bg-slate-950 text-white' : 'text-slate-500 hover:text-indigo-600'
                }`}
              >
                <FileText className="h-3.5 w-3.5" />
                PDF
              </button>
            </div>
            {viewerMode === 'manuscript' && !showOutlineRail && outlineItems.length > 0 ? (
              <button
                type="button"
                onClick={onOpenOutlineSheet}
                className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-4 py-2 text-[0.68rem] font-bold uppercase tracking-[0.18em] text-slate-600 transition hover:border-indigo-200 hover:text-indigo-600"
              >
                <BookOpen className="h-3.5 w-3.5" />
                Outline
              </button>
            ) : null}
            {activeEvidenceSummary ? (
              <button
                type="button"
                onClick={onClearActiveEvidence}
                className="inline-flex items-center gap-2 rounded-full border border-indigo-100 bg-indigo-50 px-4 py-2 text-[0.7rem] font-bold uppercase tracking-[0.18em] text-indigo-700 transition hover:bg-indigo-100"
              >
                {activeEvidenceSummary}
                <X className="h-3.5 w-3.5" />
              </button>
            ) : null}
          </div>
        </div>
      </div>

      <div className="relative flex-1 overflow-hidden">
        <div
          ref={manuscriptScrollContainerRef}
          onScroll={onManuscriptScroll}
          aria-hidden={viewerMode !== 'manuscript'}
          className={`custom-scrollbar absolute inset-0 overflow-y-auto scroll-smooth px-6 py-8 selection:bg-indigo-100 transition-opacity duration-200 sm:px-8 ${
            viewerMode === 'manuscript' ? 'pointer-events-auto opacity-100' : 'pointer-events-none opacity-0'
          }`}
        >
          {isLoading ? (
            <ManuscriptLoadingState />
          ) : viewer && viewer.blocks.length > 0 ? (
            <div className={`mx-auto pb-28 ${showOutlineRail ? 'max-w-[1240px]' : 'max-w-[980px]'}`}>
              <div className={`flex items-start gap-6 ${showOutlineRail ? 'xl:gap-8' : ''}`}>
                <div className="min-w-0 flex-1">
                  <div className="manuscript-canvas px-7 py-8 sm:px-10 sm:py-10">
                    {byline || affiliationLine ? (
                      <section className="mb-10 border-b border-slate-200/80 pb-8">
                        {byline ? <div className="text-[1.04rem] font-semibold leading-7 text-slate-800">{byline}</div> : null}
                        {affiliationLine ? (
                          <div className="font-scholar mt-2 text-[1rem] italic leading-7 text-slate-500">{affiliationLine}</div>
                        ) : null}
                      </section>
                    ) : null}

                    <div className="space-y-8">
                      {viewer.blocks.map((block) => {
                        const directHighlightMode: EvidenceHighlightMode =
                          previewBlockId === block.block_id
                            ? 'preview'
                            : activeBlock?.block_id === block.block_id
                              ? 'active'
                              : linkedCueBlockId === block.block_id
                                ? 'linked'
                                : 'none';
                        const sectionHighlightMode: EvidenceHighlightMode =
                          block.block_type === 'section_heading'
                            ? previewSectionCueBlockId === block.block_id
                              ? 'preview'
                              : activeSectionCueBlockId === block.block_id || linkedSectionCueBlockId === block.block_id
                                ? 'linked'
                                : 'none'
                            : 'none';
                        const highlightMode = directHighlightMode !== 'none' ? directHighlightMode : sectionHighlightMode;

                        return (
                          <ViewerBlock
                            key={block.block_id}
                            block={block}
                            highlightMode={highlightMode}
                            shouldFlash={flashBlockId === block.block_id}
                          />
                        );
                      })}
                    </div>

                    {viewer.references && viewer.references.length > 0 ? (
                      <ReferencesAppendix references={viewer.references} />
                    ) : null}
                  </div>
                </div>

                {showOutlineRail ? (
                  <ManuscriptOutlineRail
                    items={outlineItems}
                    activeBlockId={activeSectionBlockId}
                    previewBlockId={previewSectionCueBlockId}
                    cueBlockId={activeSectionCueBlockId ?? linkedSectionCueBlockId}
                    onSelect={onSelectOutlineItem}
                  />
                ) : null}
              </div>
            </div>
          ) : (
            <div className="mx-auto flex h-full max-w-[780px] flex-col items-center justify-center">
              <div className="rounded-[2rem] border border-slate-200/85 bg-white/92 px-10 py-12 text-center shadow-scholar-sm">
                <div className="text-[0.72rem] font-bold uppercase tracking-[0.22em] text-slate-400">Manuscript</div>
                <div className="mt-3 text-[1rem] leading-7 text-slate-500">No manuscript content available.</div>
              </div>
            </div>
          )}
        </div>

        <div
          ref={pdfScrollContainerRef}
          onScroll={onPdfScroll}
          aria-hidden={viewerMode !== 'pdf'}
          className={`custom-scrollbar absolute inset-0 overflow-y-auto scroll-smooth px-6 py-8 selection:bg-indigo-100 transition-opacity duration-200 sm:px-8 ${
            viewerMode === 'pdf' ? 'pointer-events-auto opacity-100' : 'pointer-events-none opacity-0'
          }`}
        >
          <PaperPdfViewer
            pdfUrl={viewer?.pdf_url ?? null}
            scrollContainerRef={pdfScrollContainerRef}
            activeTarget={activePdfTarget}
            previewTarget={previewPdfTarget}
            linkedTarget={linkedPdfTarget}
            onDocumentStateChange={onPdfDocumentStateChange}
          />
        </div>

        {viewerMode === 'manuscript' ? (
          <ManuscriptOutlineSheet
            items={outlineItems}
            activeBlockId={activeSectionBlockId}
            previewBlockId={previewSectionCueBlockId}
            cueBlockId={activeSectionCueBlockId ?? linkedSectionCueBlockId}
            onSelect={onSelectOutlineItem}
            open={outlineSheetOpen}
            onClose={onOutlineSheetClose}
          />
        ) : null}
      </div>
    </div>
  );
}

export function SplitPaneWorkspace({
  paper,
  initialChatHistory,
  onChatHistoryUpdate,
  onBack,
  onOpenGlobalSearch,
}: SplitPaneWorkspaceProps) {
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>(initialChatHistory);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [typingPhaseIndex, setTypingPhaseIndex] = useState(0);
  const [hoveredCitationKey, setHoveredCitationKey] = useState<string | null>(null);
  const [activeEvidenceId, setActiveEvidenceId] = useState<string | null>(null);
  const [previewEvidenceId, setPreviewEvidenceId] = useState<string | null>(null);
  const [linkedCueEvidenceId, setLinkedCueEvidenceId] = useState<string | null>(null);
  const [activeBlockId, setActiveBlockId] = useState<string | null>(null);
  const [flashBlockId, setFlashBlockId] = useState<string | null>(null);
  const [viewer, setViewer] = useState<PaperViewerResponse | null>(null);
  const [isViewerLoading, setIsViewerLoading] = useState(true);
  const [viewerMode, setViewerMode] = useState<ViewerMode>('manuscript');
  const [leftWidth, setLeftWidth] = useState(() => loadWorkspaceUiState().desktopSplitPct);
  const [isDragging, setIsDragging] = useState(false);
  const [isRationaleOpen, setIsRationaleOpen] = useState(() => loadWorkspaceUiState().rationaleOpen);
  const [scrollProgress, setScrollProgress] = useState(0);
  const [activeSectionBlockId, setActiveSectionBlockId] = useState<string | null>(null);
  const [isOutlineOpen, setIsOutlineOpen] = useState(false);
  const [pdfReadyPageCount, setPdfReadyPageCount] = useState(0);
  const [viewportWidth, setViewportWidth] = useState(() =>
    typeof window !== 'undefined' ? window.innerWidth : DESKTOP_RESIZABLE_BREAKPOINT,
  );
  const [viewportHeight, setViewportHeight] = useState(() =>
    typeof window !== 'undefined' ? window.innerHeight : DESKTOP_RESIZABLE_BREAKPOINT,
  );
  const [isCoarsePointer, setIsCoarsePointer] = useState(() =>
    typeof window !== 'undefined' ? window.matchMedia('(pointer: coarse)').matches : false,
  );
  const [mobilePane, setMobilePane] = useState<WorkspaceMobilePane>(() => loadWorkspaceUiState().mobilePane);

  const workspaceRootRef = useRef<HTMLDivElement>(null);
  const chatScrollRef = useRef<HTMLDivElement>(null);
  const pendingEvidenceIdRef = useRef<string | null>(null);
  const manuscriptScrollRef = useRef<HTMLDivElement>(null);
  const pdfScrollRef = useRef<HTMLDivElement>(null);
  const flashTimeoutRef = useRef<number | null>(null);
  const linkedCueTimeoutRef = useRef<number | null>(null);
  const draggingPointerIdRef = useRef<number | null>(null);
  const dragHandleRef = useRef<HTMLDivElement | null>(null);
  const didMountChatHistoryRef = useRef(false);

  const viewportMode = useMemo(
    () => getWorkspaceViewportMode(viewportWidth, viewportHeight, isCoarsePointer),
    [isCoarsePointer, viewportHeight, viewportWidth],
  );
  const chatPaneWidthPct = viewportMode === 'desktop' ? clampDesktopChatWidth(leftWidth) : TABLET_CHAT_WIDTH_PCT;
  const manuscriptPaneWidthPct = 100 - chatPaneWidthPct;
  const isMobile = viewportMode === 'mobile';
  const typingPhases = useMemo(
    () => ['Reading manuscript evidence', 'Aligning supporting passages', 'Drafting the answer'],
    [],
  );
  const chatActivityMode = useTransientActivityMode(isTyping);
  const chatActivityLabel = chatActivityMode === 'active'
    ? typingPhaseIndex < 2
      ? 'Reading'
      : 'Thinking'
    : chatActivityMode === 'done'
      ? 'Ready'
      : null;

  const startDragging = useCallback((event: ReactPointerEvent<HTMLDivElement>) => {
    if (viewportMode !== 'desktop') {
      return;
    }
    draggingPointerIdRef.current = event.pointerId;
    dragHandleRef.current = event.currentTarget;
    event.currentTarget.setPointerCapture(event.pointerId);
    setIsDragging(true);
    event.preventDefault();
  }, [viewportMode]);

  const stopDragging = useCallback(() => {
    if (draggingPointerIdRef.current !== null && dragHandleRef.current?.hasPointerCapture(draggingPointerIdRef.current)) {
      dragHandleRef.current.releasePointerCapture(draggingPointerIdRef.current);
    }
    draggingPointerIdRef.current = null;
    setIsDragging(false);
  }, []);

  const onDrag = useCallback(
    (event: PointerEvent) => {
      if (!isDragging) {
        return;
      }
      if (draggingPointerIdRef.current !== null && event.pointerId !== draggingPointerIdRef.current) {
        return;
      }
      const rootBounds = workspaceRootRef.current?.getBoundingClientRect();
      if (!rootBounds || rootBounds.width <= 0) {
        return;
      }
      const relativeWidth = ((event.clientX - rootBounds.left) / rootBounds.width) * 100;
      setLeftWidth(clampDesktopChatWidth(relativeWidth));
    },
    [isDragging],
  );

  const clearActiveHighlight = useCallback(() => {
    if (flashTimeoutRef.current !== null) {
      window.clearTimeout(flashTimeoutRef.current);
      flashTimeoutRef.current = null;
    }
    setActiveBlockId(null);
    setFlashBlockId(null);
  }, []);

  const sectionOutlineItems = useMemo<ManuscriptOutlineItem[]>(() => {
    if (!viewer) {
      return [];
    }
    return viewer.blocks
      .filter((block) => block.block_type === 'section_heading' && block.text.trim().length > 0)
      .map((block) => ({
        blockId: block.block_id,
        title: block.text.trim(),
        depth: Math.max(0, block.section_path.length - 1),
        pageStart: block.page_start,
      }));
  }, [viewer]);

  const sectionHeadingByBlockId = useMemo<Record<string, string>>(() => {
    if (!viewer) {
      return {};
    }
    const mapping: Record<string, string> = {};
    let currentSectionHeadingId: string | null = null;
    for (const block of viewer.blocks) {
      if (block.block_type === 'section_heading') {
        currentSectionHeadingId = block.block_id;
        mapping[block.block_id] = block.block_id;
        continue;
      }
      if (currentSectionHeadingId) {
        mapping[block.block_id] = currentSectionHeadingId;
      }
    }
    return mapping;
  }, [viewer]);

  const resolveNavigationTargetForEvidence = useCallback((evidenceId: string | null | undefined): EvidenceNavigationTarget | null => {
    if (!evidenceId) {
      return null;
    }
    return viewer?.evidence_navigation_map[evidenceId] ?? null;
  }, [viewer]);

  const resolveMarkdownBlockIdForEvidence = useCallback((evidenceId: string | null | undefined): string | null => {
    return resolveNavigationTargetForEvidence(evidenceId)?.markdown_target?.block_id ?? null;
  }, [resolveNavigationTargetForEvidence]);

  const resolvePdfTargetForEvidence = useCallback((evidenceId: string | null | undefined) => {
    return resolveNavigationTargetForEvidence(evidenceId)?.pdf_target ?? null;
  }, [resolveNavigationTargetForEvidence]);

  const resolveSectionHeadingIdForBlock = useCallback((blockId: string | null | undefined): string | null => {
    if (!blockId) {
      return null;
    }
    return sectionHeadingByBlockId[blockId] ?? null;
  }, [sectionHeadingByBlockId]);

  const resolveSectionHeadingIdForEvidence = useCallback((evidenceId: string | null | undefined): string | null => {
    const navigationTarget = resolveNavigationTargetForEvidence(evidenceId);
    const explicitSectionBlockId = navigationTarget?.markdown_target?.section_block_id ?? null;
    if (explicitSectionBlockId) {
      return explicitSectionBlockId;
    }
    const blockId = navigationTarget?.markdown_target?.block_id ?? null;
    if (!blockId) {
      return null;
    }
    return sectionHeadingByBlockId[blockId] ?? null;
  }, [resolveNavigationTargetForEvidence, sectionHeadingByBlockId]);

  const scrollToBlockId = useCallback((blockId: string, behavior: ScrollBehavior = 'smooth'): boolean => {
    const target = document.getElementById(viewerBlockDomId(blockId));
    if (!(target instanceof HTMLElement)) {
      return false;
    }
    const container = manuscriptScrollRef.current;
    if (!(container instanceof HTMLElement)) {
      target.scrollIntoView({ behavior, block: 'start' });
      return true;
    }

    const containerRect = container.getBoundingClientRect();
    const targetRect = target.getBoundingClientRect();
    const readingOffset = Math.max(92, Math.min(144, Math.round(container.clientHeight * 0.14)));
    const top = container.scrollTop + (targetRect.top - containerRect.top) - readingOffset;
    container.scrollTo({ top: Math.max(0, top), behavior });
    return true;
  }, []);

  const scrollToPdfPage = useCallback((pageNumber: number, behavior: ScrollBehavior = 'smooth'): boolean => {
    const target = document.getElementById(paperPdfPageDomId(pageNumber));
    if (!(target instanceof HTMLElement)) {
      return false;
    }
    const container = pdfScrollRef.current;
    if (!(container instanceof HTMLElement)) {
      target.scrollIntoView({ behavior, block: 'start' });
      return true;
    }

    const containerRect = container.getBoundingClientRect();
    const targetRect = target.getBoundingClientRect();
    const readingOffset = Math.max(76, Math.min(132, Math.round(container.clientHeight * 0.1)));
    const top = container.scrollTop + (targetRect.top - containerRect.top) - readingOffset;
    container.scrollTo({ top: Math.max(0, top), behavior });
    return true;
  }, []);

  const clearLinkedCue = useCallback(() => {
    if (linkedCueTimeoutRef.current !== null) {
      window.clearTimeout(linkedCueTimeoutRef.current);
      linkedCueTimeoutRef.current = null;
    }
    setLinkedCueEvidenceId(null);
  }, []);

  const clearActiveEvidence = useCallback(() => {
    pendingEvidenceIdRef.current = null;
    setActiveEvidenceId(null);
    setPreviewEvidenceId(null);
    clearLinkedCue();
    clearActiveHighlight();
  }, [clearActiveHighlight, clearLinkedCue]);

  const focusEvidence = useCallback(
    (evidenceId: string, shouldScroll: boolean): boolean => {
      const navigationTarget = resolveNavigationTargetForEvidence(evidenceId);
      const blockId = navigationTarget?.markdown_target?.block_id ?? null;
      const sectionBlockId = navigationTarget?.markdown_target?.section_block_id ?? null;
      const pdfTarget = navigationTarget?.pdf_target ?? null;

      if (viewerMode === 'pdf') {
        const primaryPage = pdfTarget?.primary_page ?? null;
        if (!primaryPage) {
          pendingEvidenceIdRef.current = evidenceId;
          return false;
        }
        const pageTarget = document.getElementById(paperPdfPageDomId(primaryPage));
        if (!(pageTarget instanceof HTMLElement)) {
          pendingEvidenceIdRef.current = evidenceId;
          return false;
        }

        pendingEvidenceIdRef.current = null;
        setPreviewEvidenceId(null);
        clearLinkedCue();
        clearActiveHighlight();
        setActiveBlockId(blockId);
        setActiveSectionBlockId(sectionBlockId ?? resolveSectionHeadingIdForBlock(blockId));

        if (shouldScroll) {
          scrollToPdfPage(primaryPage, 'smooth');
        }
        return true;
      }

      if (!blockId) {
        pendingEvidenceIdRef.current = evidenceId;
        return false;
      }

      const target = document.getElementById(viewerBlockDomId(blockId));
      if (!(target instanceof HTMLElement)) {
        pendingEvidenceIdRef.current = evidenceId;
        return false;
      }

      pendingEvidenceIdRef.current = null;
      setPreviewEvidenceId(null);
      clearLinkedCue();
      clearActiveHighlight();
      setActiveBlockId(blockId);
      setActiveSectionBlockId(sectionBlockId ?? resolveSectionHeadingIdForBlock(blockId));
      setFlashBlockId(blockId);

      if (flashTimeoutRef.current !== null) {
        window.clearTimeout(flashTimeoutRef.current);
      }
      flashTimeoutRef.current = window.setTimeout(() => {
        setFlashBlockId((current) => (current === blockId ? null : current));
        flashTimeoutRef.current = null;
      }, 2800);

      if (shouldScroll) {
        scrollToBlockId(blockId, 'smooth');
      }
      return true;
    },
    [
      clearActiveHighlight,
      clearLinkedCue,
      resolveNavigationTargetForEvidence,
      resolveSectionHeadingIdForBlock,
      scrollToBlockId,
      scrollToPdfPage,
      viewerMode,
    ],
  );

  const activateCitation = useCallback(
    (citation: PaperChatCitation) => {
      setActiveEvidenceId(citation.evidence_id);
      if (isMobile && mobilePane !== 'manuscript') {
        pendingEvidenceIdRef.current = citation.evidence_id;
        setMobilePane('manuscript');
        return;
      }
      focusEvidence(citation.evidence_id, true);
    },
    [focusEvidence, isMobile, mobilePane],
  );

  const previewCitation = useCallback((citation: PaperChatCitation) => {
    if (isCoarsePointer) {
      return;
    }
    setPreviewEvidenceId(citation.evidence_id);
  }, [isCoarsePointer]);

  const clearPreviewCitation = useCallback((citation?: PaperChatCitation) => {
    setPreviewEvidenceId((current) => {
      if (!citation || current === citation.evidence_id) {
        return null;
      }
      return current;
    });
  }, []);

  useEffect(() => {
    const handleResize = () => {
      setViewportWidth(window.innerWidth);
      setViewportHeight(window.innerHeight);
      setIsCoarsePointer(window.matchMedia('(pointer: coarse)').matches);
    };
    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, []);

  useEffect(() => {
    saveDesktopSplitPct(leftWidth);
  }, [leftWidth]);

  useEffect(() => {
    saveMobilePane(mobilePane);
  }, [mobilePane]);

  useEffect(() => {
    saveRationaleOpen(isRationaleOpen);
  }, [isRationaleOpen]);

  useEffect(() => {
    const element = viewerMode === 'pdf' ? pdfScrollRef.current : manuscriptScrollRef.current;
    if (!(element instanceof HTMLDivElement)) {
      return;
    }
    const scrollRange = element.scrollHeight - element.clientHeight;
    if (scrollRange <= 0) {
      setScrollProgress(0);
      return;
    }
    setScrollProgress((element.scrollTop / scrollRange) * 100);
  }, [viewerMode]);

  useEffect(() => {
    if (viewportMode !== 'desktop' && isDragging) {
      stopDragging();
    }
  }, [isDragging, stopDragging, viewportMode]);

  useEffect(() => {
    if (isDragging) {
      window.addEventListener('pointermove', onDrag);
      window.addEventListener('pointerup', stopDragging);
      window.addEventListener('pointercancel', stopDragging);
    } else {
      window.removeEventListener('pointermove', onDrag);
      window.removeEventListener('pointerup', stopDragging);
      window.removeEventListener('pointercancel', stopDragging);
    }
    return () => {
      window.removeEventListener('pointermove', onDrag);
      window.removeEventListener('pointerup', stopDragging);
      window.removeEventListener('pointercancel', stopDragging);
    };
  }, [isDragging, onDrag, stopDragging]);

  useEffect(() => {
    return () => {
      if (flashTimeoutRef.current !== null) {
        window.clearTimeout(flashTimeoutRef.current);
      }
      if (linkedCueTimeoutRef.current !== null) {
        window.clearTimeout(linkedCueTimeoutRef.current);
      }
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    const loadViewer = async () => {
      setViewer(null);
      setIsViewerLoading(true);
      setScrollProgress(0);
      setActiveSectionBlockId(null);
      setIsOutlineOpen(false);
      setPdfReadyPageCount(0);
      clearActiveEvidence();
      try {
        const payload = await fetchPaperViewer(paper.paper_id);
        if (!cancelled) {
          setViewer(payload);
        }
      } catch {
        if (!cancelled) {
          setViewer(null);
        }
      } finally {
        if (!cancelled) {
          setIsViewerLoading(false);
        }
      }
    };
    void loadViewer();
    return () => {
      cancelled = true;
    };
  }, [clearActiveEvidence, paper.paper_id]);

  useEffect(() => {
    if (sectionOutlineItems.length === 0) {
      setActiveSectionBlockId(null);
      return;
    }
    setActiveSectionBlockId((current) => current ?? sectionOutlineItems[0]?.blockId ?? null);
  }, [sectionOutlineItems]);

  useEffect(() => {
    onChatHistoryUpdate(chatHistory);
  }, [chatHistory, onChatHistoryUpdate]);

  useEffect(() => {
    if (!isTyping) {
      setTypingPhaseIndex(0);
      return;
    }

    const intervalId = window.setInterval(() => {
      setTypingPhaseIndex((current) => (current + 1) % typingPhases.length);
    }, 1500);

    return () => {
      window.clearInterval(intervalId);
    };
  }, [isTyping, typingPhases]);

  useEffect(() => {
    if (!didMountChatHistoryRef.current) {
      didMountChatHistoryRef.current = true;
      return;
    }

    const lastMessage = chatHistory[chatHistory.length - 1];
    if (!lastMessage || lastMessage.role !== 'assistant' || !lastMessage.citations?.length) {
      return;
    }

    const cueCitation = lastMessage.citations[0];
    if (!cueCitation) {
      return;
    }

    clearLinkedCue();
    setLinkedCueEvidenceId(cueCitation.evidence_id);
    linkedCueTimeoutRef.current = window.setTimeout(() => {
      setLinkedCueEvidenceId((current) => (current === cueCitation.evidence_id ? null : current));
      linkedCueTimeoutRef.current = null;
    }, 2600);
  }, [chatHistory, clearLinkedCue]);

  useEffect(() => {
    chatScrollRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory, isTyping]);

  useEffect(() => {
    if (isViewerLoading) {
      return;
    }
    if (isMobile && mobilePane !== 'manuscript') {
      return;
    }
    const pendingEvidenceId = pendingEvidenceIdRef.current;
    if (!pendingEvidenceId) {
      return;
    }
    const frameId = window.requestAnimationFrame(() => {
      focusEvidence(pendingEvidenceId, true);
    });
    return () => window.cancelAnimationFrame(frameId);
  }, [focusEvidence, isMobile, isViewerLoading, mobilePane, pdfReadyPageCount, viewer]);

  useEffect(() => {
    if (!activeEvidenceId || isViewerLoading) {
      return;
    }
    if (isMobile && mobilePane !== 'manuscript') {
      return;
    }
    const frameId = window.requestAnimationFrame(() => {
      focusEvidence(activeEvidenceId, true);
    });
    return () => window.cancelAnimationFrame(frameId);
  }, [activeEvidenceId, focusEvidence, isMobile, isViewerLoading, mobilePane, viewerMode]);

  const handleManuscriptScroll = useCallback((event: UIEvent<HTMLDivElement>) => {
    const element = event.currentTarget as HTMLDivElement | null;
    if (!element) {
      return;
    }
    const scrollRange = element.scrollHeight - element.clientHeight;
    if (scrollRange <= 0) {
      setScrollProgress(0);
      return;
    }
    setScrollProgress((element.scrollTop / scrollRange) * 100);

    if (sectionOutlineItems.length === 0) {
      return;
    }
    const containerTop = element.getBoundingClientRect().top;
    let nextActiveSectionId = sectionOutlineItems[0]?.blockId ?? null;
    for (const section of sectionOutlineItems) {
      const headingElement = document.getElementById(viewerBlockDomId(section.blockId));
      if (!(headingElement instanceof HTMLElement)) {
        continue;
      }
      const distanceFromTop = headingElement.getBoundingClientRect().top - containerTop;
      if (distanceFromTop <= 140) {
        nextActiveSectionId = section.blockId;
      } else {
        break;
      }
    }
    setActiveSectionBlockId((current) => (current === nextActiveSectionId ? current : nextActiveSectionId));
  }, [sectionOutlineItems]);

  const handlePdfScroll = useCallback((event: UIEvent<HTMLDivElement>) => {
    const element = event.currentTarget as HTMLDivElement | null;
    if (!element) {
      return;
    }
    const scrollRange = element.scrollHeight - element.clientHeight;
    if (scrollRange <= 0) {
      setScrollProgress(0);
      return;
    }
    setScrollProgress((element.scrollTop / scrollRange) * 100);
  }, []);

  const handleSelectOutlineItem = useCallback((blockId: string) => {
    setActiveSectionBlockId(blockId);
    scrollToBlockId(blockId, 'smooth');
  }, [scrollToBlockId]);

  const handleSendMessage = useCallback(async () => {
    const trimmedInput = input.trim();
    if (!trimmedInput || isTyping) {
      return;
    }

    const userMessage: ChatMessage = { role: 'user', content: trimmedInput };
    setChatHistory((previous) => [...previous, userMessage]);
    setInput('');
    setIsTyping(true);

    try {
      const response = await chatWithPaper({
        paper_id: paper.paper_id,
        query: trimmedInput,
        history: chatHistory.slice(-4),
      });
      setChatHistory((previous) => [
        ...previous,
        { role: 'assistant', content: response.answer, citations: response.citations },
      ]);
    } catch {
      setChatHistory((previous) => [
        ...previous,
        { role: 'assistant', content: 'The paper chat request did not complete successfully.' },
      ]);
    } finally {
      setIsTyping(false);
    }
  }, [chatHistory, input, isTyping, paper.paper_id]);

  const processMessageContent = useCallback((content: string) => {
    return content.replace(/(?<!\[)\[(\d+)\](?!\()/g, '[[$1]](#cite-$1)');
  }, []);

  const previewBlockId = useMemo(() => {
    if (activeEvidenceId && previewEvidenceId === activeEvidenceId) {
      return null;
    }
    return resolveMarkdownBlockIdForEvidence(previewEvidenceId);
  }, [activeEvidenceId, previewEvidenceId, resolveMarkdownBlockIdForEvidence]);

  const linkedCueBlockId = useMemo(() => {
    if (activeEvidenceId || previewEvidenceId) {
      return null;
    }
    return resolveMarkdownBlockIdForEvidence(linkedCueEvidenceId);
  }, [activeEvidenceId, linkedCueEvidenceId, previewEvidenceId, resolveMarkdownBlockIdForEvidence]);

  const activePdfTarget = useMemo(() => {
    return activeEvidenceId ? resolvePdfTargetForEvidence(activeEvidenceId) : null;
  }, [activeEvidenceId, resolvePdfTargetForEvidence]);

  const previewPdfTarget = useMemo(() => {
    if (activeEvidenceId && previewEvidenceId === activeEvidenceId) {
      return null;
    }
    return resolvePdfTargetForEvidence(previewEvidenceId);
  }, [activeEvidenceId, previewEvidenceId, resolvePdfTargetForEvidence]);

  const linkedPdfTarget = useMemo(() => {
    if (activeEvidenceId || previewEvidenceId) {
      return null;
    }
    return resolvePdfTargetForEvidence(linkedCueEvidenceId);
  }, [activeEvidenceId, linkedCueEvidenceId, previewEvidenceId, resolvePdfTargetForEvidence]);

  const activeSectionCueBlockId = useMemo(() => {
    return activeEvidenceId
      ? resolveSectionHeadingIdForEvidence(activeEvidenceId)
      : resolveSectionHeadingIdForBlock(activeBlockId);
  }, [activeBlockId, activeEvidenceId, resolveSectionHeadingIdForBlock, resolveSectionHeadingIdForEvidence]);

  const previewSectionCueBlockId = useMemo(() => {
    return previewEvidenceId
      ? resolveSectionHeadingIdForEvidence(previewEvidenceId)
      : resolveSectionHeadingIdForBlock(previewBlockId);
  }, [previewBlockId, previewEvidenceId, resolveSectionHeadingIdForBlock, resolveSectionHeadingIdForEvidence]);

  const linkedSectionCueBlockId = useMemo(() => {
    return linkedCueEvidenceId
      ? resolveSectionHeadingIdForEvidence(linkedCueEvidenceId)
      : resolveSectionHeadingIdForBlock(linkedCueBlockId);
  }, [linkedCueBlockId, linkedCueEvidenceId, resolveSectionHeadingIdForBlock, resolveSectionHeadingIdForEvidence]);

  const activeBlock = useMemo(() => {
    if (!viewer || !activeBlockId) {
      return null;
    }
    return viewer.blocks.find((block) => block.block_id === activeBlockId) ?? null;
  }, [activeBlockId, viewer]);

  const latestAssistantMessageIndex = useMemo(() => {
    for (let index = chatHistory.length - 1; index >= 0; index -= 1) {
      if (chatHistory[index]?.role === 'assistant') {
        return index;
      }
    }
    return -1;
  }, [chatHistory]);

  const chatRemarkPlugins = useMemo(() => [remarkGfm, remarkMath], []);
  const chatRehypePlugins = useMemo(() => [rehypeKatex], []);
  const renderChatPane = (showHeader: boolean) => (
    <div className="flex h-full min-h-0 flex-col bg-[#F7F9FC]">
      {showHeader ? (
        <div className="sticky top-0 z-30 flex items-center justify-between border-b border-slate-200/60 bg-[#F7F9FC]/92 px-6 py-4 backdrop-blur-xl">
          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={onBack}
              className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-5 py-2.5 text-[0.72rem] font-bold uppercase tracking-[0.22em] text-slate-700 shadow-scholar-sm transition hover:border-indigo-200 hover:text-indigo-600"
            >
              <ChevronLeft className="h-4 w-4" />
              Workspace
            </button>
            <button
              type="button"
              onClick={onOpenGlobalSearch}
              className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-4 py-2.5 text-[0.72rem] font-bold uppercase tracking-[0.2em] text-slate-600 shadow-scholar-sm transition hover:border-indigo-200 hover:text-indigo-600"
            >
              <Search className="h-3.5 w-3.5" />
              Search
            </button>
            <a
              href={paperZoteroPageUrl(paper.paper_id)}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-4 py-2.5 text-[0.72rem] font-bold uppercase tracking-[0.2em] text-slate-600 shadow-scholar-sm transition hover:border-indigo-200 hover:text-indigo-600"
            >
              <BookMarked className="h-3.5 w-3.5" />
              Save to Zotero
            </a>
          </div>
          <div className="text-[0.66rem] font-bold uppercase tracking-[0.26em] text-indigo-400">
            {paper.venue.toUpperCase()} {paper.year}
          </div>
        </div>
      ) : null}

      <div className="custom-scrollbar flex-1 overflow-y-auto px-6 py-6 pb-10">
        <div className="mb-6 overflow-hidden rounded-[1.55rem] border border-slate-200/80 bg-white/92 shadow-scholar-sm">
          <button
            type="button"
            onClick={() => setIsRationaleOpen((current) => !current)}
            className="flex w-full items-center justify-between px-5 py-4 text-left transition hover:bg-[linear-gradient(180deg,rgba(248,250,252,0.96),rgba(255,255,255,0.92))] focus-visible:outline-none"
          >
            <div className="flex items-center gap-3">
              <div className="rounded-xl bg-indigo-50 p-2 text-indigo-500">
                <Quote className="h-3.5 w-3.5" />
              </div>
              <div>
                <div className="text-[0.68rem] font-bold uppercase tracking-[0.2em] text-slate-400">Why it matched</div>
                <div className="mt-1 text-[0.95rem] font-semibold text-slate-800 line-clamp-1">{paper.title}</div>
              </div>
            </div>
            {isRationaleOpen ? <ChevronUp className="h-4 w-4 text-slate-300" /> : <ChevronDown className="h-4 w-4 text-slate-300" />}
          </button>
          <AnimatePresence initial={false}>
            {isRationaleOpen ? (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                className="overflow-hidden"
              >
                <div className="border-t border-slate-100 px-5 pb-5 pt-4">
                  <p className="font-scholar text-[1rem] italic leading-8 text-slate-600">&ldquo;{paper.rationale}&rdquo;</p>
                </div>
              </motion.div>
            ) : null}
          </AnimatePresence>
        </div>

        {chatHistory.length === 0 ? (
          <div className="flex min-h-[38vh] flex-col items-center justify-center px-6 text-center">
            <div className="mb-6 flex h-20 w-20 items-center justify-center rounded-[2rem] border border-slate-200 bg-white shadow-scholar-sm">
              <Sparkles className="h-8 w-8 text-indigo-500" />
            </div>
            <div className="text-[0.7rem] font-bold uppercase tracking-[0.24em] text-indigo-500">Deep paper chat</div>
            <p className="mt-3 max-w-[21rem] text-[0.95rem] leading-7 text-slate-500">
              Ask about the method, experiments, results, or assumptions. Click any cited evidence to jump to the supporting passage.
            </p>
          </div>
        ) : null}

        <div className="space-y-5">
          {chatHistory.map((message, index) => (
            (() => {
              const isLatestAssistant = message.role === 'assistant' && index === latestAssistantMessageIndex;
              return (
                <motion.div
                  key={`${message.role}-${index}`}
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`w-full rounded-[1.8rem] px-6 py-5 ${
                  message.role === 'user'
                    ? 'max-w-[85%] rounded-br-md border border-indigo-100/90 bg-[linear-gradient(180deg,rgba(255,255,255,0.98),rgba(238,242,255,0.94))] text-slate-900 shadow-[0_16px_36px_rgba(79,70,229,0.08),0_4px_10px_rgba(15,23,42,0.04)]'
                    : `max-w-[92%] rounded-bl-md border border-slate-200/80 bg-white text-slate-800 shadow-scholar-sm ${
                        isLatestAssistant ? 'assistant-answer-ready' : ''
                      }`
                }`}
                  >
                    {message.role === 'assistant' ? (
                      <div className="mb-3 flex items-center justify-between gap-3">
                        <div className="flex items-center gap-3">
                          <ActivityMark
                            mode={isLatestAssistant && chatActivityMode === 'done' ? 'done' : 'idle'}
                            label={isLatestAssistant && chatActivityMode === 'done' ? 'Ready' : null}
                            layout="inline"
                            size="sm"
                          />
                          <div className="text-[0.66rem] font-bold uppercase tracking-[0.2em] text-slate-400">Paper chat</div>
                        </div>
                        {isLatestAssistant && chatActivityMode === 'done' ? (
                          <div className="inline-flex items-center gap-2 rounded-full border border-indigo-100 bg-indigo-50/80 px-2.5 py-1 text-[0.58rem] font-bold uppercase tracking-[0.18em] text-indigo-600">
                            <span className="h-1.5 w-1.5 rounded-full bg-indigo-500" />
                            Answer ready
                          </div>
                        ) : null}
                      </div>
                    ) : (
                      <div className="mb-3 text-[0.66rem] font-bold uppercase tracking-[0.2em] text-indigo-400">Your question</div>
                    )}

                    <div
                      className={`prose prose-sm max-w-none leading-relaxed ${
                    message.role === 'user'
                      ? 'prose-slate prose-p:text-slate-900 prose-strong:text-slate-950 prose-headings:text-slate-950'
                      : 'prose-slate prose-indigo'
                  } prose-p:text-[15px] prose-li:text-[15px] selection:bg-indigo-200`}
                    >
                      <ReactMarkdown
                        remarkPlugins={chatRemarkPlugins}
                        rehypePlugins={chatRehypePlugins}
                        components={{
                          a({ href, children, ...props }) {
                            const text = String(children);
                            const match = href?.match(/^#cite-(\d+)$/);
                            if (match && message.citations) {
                              const citationIndex = parseInt(match[1], 10) - 1;
                              const citation = message.citations[citationIndex];
                              if (citation) {
                                const citationHoverKey = `${index}:${citationIndex}`;
                                const sectionLabel = formatSectionLabel(citation.section_path);
                                const isHovered = hoveredCitationKey === citationHoverKey;
                                const isActive = activeEvidenceId === citation.evidence_id;
                                const isPreview = previewEvidenceId === citation.evidence_id;
                                const isLinkedCue = linkedCueEvidenceId === citation.evidence_id;
                                return (
                                  <span
                                    className="group relative mx-0.5 inline-block cursor-pointer"
                                    onClick={() => activateCitation(citation)}
                                    onMouseEnter={() => {
                                      previewCitation(citation);
                                      setHoveredCitationKey(citationHoverKey);
                                    }}
                                    onMouseLeave={() => {
                                      clearPreviewCitation(citation);
                                      setHoveredCitationKey((current) => (current === citationHoverKey ? null : current));
                                    }}
                                    onFocus={() => {
                                      previewCitation(citation);
                                      setHoveredCitationKey(citationHoverKey);
                                    }}
                                    onBlur={() => {
                                      clearPreviewCitation(citation);
                                      setHoveredCitationKey((current) => (current === citationHoverKey ? null : current));
                                    }}
                                  >
                                    <span
                                      className={`inline-flex items-center justify-center rounded-lg border px-2 py-0.5 align-super text-[10px] font-bold shadow-sm transition-all duration-300 ${
                                    isActive
                                      ? 'scale-105 border-indigo-600 bg-indigo-600 text-white'
                                      : isPreview
                                        ? 'scale-[1.04] border-indigo-400 bg-indigo-100 text-indigo-700'
                                        : isLinkedCue
                                          ? 'border-indigo-200 bg-indigo-50 text-indigo-600 ring-2 ring-indigo-100'
                                      : 'border-indigo-100 bg-indigo-50 text-indigo-600 group-hover:border-indigo-500 group-hover:bg-indigo-600 group-hover:text-white'
                                  }`}
                                    >
                                      {text}
                                    </span>
                                    <span
                                      className={`pointer-events-none absolute bottom-full left-1/2 z-[100] mb-3 w-80 -translate-x-1/2 rounded-[1.35rem] border border-slate-100 bg-white p-4 shadow-scholar-lg transition-all duration-200 ${
                                        isHovered ? 'visible translate-y-0 opacity-100' : 'invisible translate-y-1 opacity-0'
                                      }`}
                                    >
                                      <span className="mb-3 flex items-center justify-between gap-3 border-b border-slate-100 pb-2 text-[0.64rem] font-bold uppercase tracking-[0.2em] text-slate-400">
                                        <span className="truncate">Supporting evidence</span>
                                        <span className="rounded-full bg-indigo-50 px-2 py-0.5 text-indigo-600">
                                          {formatCitationPageLabel(citation)}
                                        </span>
                                      </span>
                                      {sectionLabel ? (
                                        <span className="mb-2 block text-[0.68rem] font-semibold uppercase tracking-[0.18em] text-indigo-500">
                                          {sectionLabel}
                                        </span>
                                      ) : null}
                                      <span className="font-scholar block text-left text-[0.92rem] italic leading-7 text-slate-700">
                                        &ldquo;{citation.snippet}&rdquo;
                                      </span>
                                    </span>
                                  </span>
                                );
                              }
                            }
                            return (
                              <a href={href} className="font-semibold text-indigo-600 hover:underline" {...props}>
                                {children}
                              </a>
                            );
                          }
                        }}
                      >
                        {message.role === 'assistant' ? processMessageContent(message.content) : message.content}
                      </ReactMarkdown>
                    </div>

                    {message.citations && message.citations.length > 0 ? (
                      <motion.div
                        initial={{ opacity: 0, y: 8 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.22, ease: 'easeOut', delay: 0.06 }}
                        className="mt-6 space-y-2 border-t border-slate-100 pt-4"
                      >
                        <div className="flex items-center gap-2 text-[0.66rem] font-bold uppercase tracking-[0.22em] text-slate-400">
                          <BookOpen className="h-3.5 w-3.5" />
                          Evidence
                        </div>
                        <div className="grid gap-2.5">
                          {message.citations.map((citation, citationIndex) => {
                            const isActive = activeEvidenceId === citation.evidence_id;
                            const isPreview = previewEvidenceId === citation.evidence_id;
                            const isLinkedCue = linkedCueEvidenceId === citation.evidence_id;
                            return (
                              <button
                                key={`${citation.evidence_id}-${citationIndex}`}
                                type="button"
                                onClick={() => activateCitation(citation)}
                                onMouseEnter={() => previewCitation(citation)}
                                onMouseLeave={() => clearPreviewCitation(citation)}
                                onFocus={() => previewCitation(citation)}
                                onBlur={() => clearPreviewCitation(citation)}
                                className={`flex items-start gap-3 rounded-[1.2rem] border px-4 py-3 text-left transition ${
                              isActive
                                ? 'border-indigo-300 bg-indigo-50/70 shadow-scholar-sm ring-1 ring-indigo-200'
                                : isPreview
                                  ? 'border-indigo-200 bg-indigo-50/55 shadow-scholar-sm'
                                  : isLinkedCue
                                    ? 'border-indigo-100 bg-white shadow-[0_10px_24px_rgba(99,102,241,0.08)] ring-1 ring-indigo-100'
                                : 'border-slate-100 bg-slate-50/70 hover:border-indigo-200 hover:bg-white'
                            }`}
                              >
                                <div
                                  className={`inline-flex h-7 min-w-7 items-center justify-center rounded-full text-[0.62rem] font-bold ${
                                isActive ? 'bg-indigo-600 text-white' : 'bg-white text-indigo-600'
                              }`}
                                >
                                  {citationIndex + 1}
                                </div>
                                <div className="min-w-0 flex-1">
                                  <div className="font-scholar line-clamp-3 text-[0.92rem] italic leading-7 text-slate-600">
                                    &ldquo;{citation.snippet}&rdquo;
                                  </div>
                                  <div className="mt-2 flex items-center gap-2 text-[0.64rem] font-bold uppercase tracking-[0.18em] text-slate-400">
                                    <Clock className="h-3 w-3" />
                                    {formatCitationPageLabel(citation)}
                                    {formatSectionLabel(citation.section_path) ? (
                                      <>
                                        <span className="text-slate-300">•</span>
                                        <span className="truncate">{formatSectionLabel(citation.section_path)}</span>
                                      </>
                                    ) : null}
                                  </div>
                                </div>
                              </button>
                            );
                          })}
                        </div>
                      </motion.div>
                    ) : null}
                  </div>
                </motion.div>
              );
            })()
          ))}

          {isTyping ? (
            <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="flex justify-start">
              <div className="w-full max-w-[92%] rounded-[1.8rem] rounded-bl-md border border-slate-200/70 bg-white p-6 shadow-scholar-sm">
                <div className="mb-4 flex items-center justify-between gap-3">
                  <div className="flex items-center gap-3">
                    <ActivityMark
                      mode="active"
                      label={null}
                      layout="inline"
                      size="sm"
                    />
                    <div className="flex flex-col">
                      <div className="text-[0.66rem] font-bold uppercase tracking-[0.22em] text-indigo-500">
                        {typingPhases[typingPhaseIndex]}
                      </div>
                      <div className="mt-1 text-[0.78rem] text-slate-500">
                        Gathering the most relevant passages before drafting the reply.
                      </div>
                    </div>
                  </div>
                  <div className="rounded-full border border-indigo-100 bg-indigo-50/80 px-2.5 py-1 text-[0.58rem] font-bold uppercase tracking-[0.18em] text-indigo-600">
                    Deep chat
                  </div>
                </div>
                <div className="space-y-3">
                  <div className="skeleton-pulse h-2.5 w-[72%] rounded-full" />
                  <div className="skeleton-pulse h-2.5 w-full rounded-full" />
                  <div className="skeleton-pulse h-2.5 w-[82%] rounded-full" />
                  <div className="skeleton-pulse h-2.5 w-[58%] rounded-full" />
                </div>
              </div>
            </motion.div>
          ) : null}
        </div>

        <div ref={chatScrollRef} />
      </div>

      <div className="border-t border-slate-200/60 bg-[#F7F9FC] p-6">
        <div className="group/input relative">
          <div className="pointer-events-none absolute inset-0 rounded-[2rem] bg-indigo-500/5 blur-2xl opacity-0 transition group-focus-within/input:opacity-100" />
          <input
            type="text"
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter') {
                void handleSendMessage();
              }
            }}
            placeholder="Ask about method, experiments, limitations, or results..."
            className="relative w-full rounded-[2rem] border border-slate-200 bg-white py-5 pl-7 pr-16 text-[15px] outline-none transition shadow-scholar-md focus:border-indigo-400 focus:ring-8 focus:ring-indigo-500/5"
          />
          <button
            type="button"
            disabled={isTyping || !input.trim()}
            onClick={() => void handleSendMessage()}
            className="absolute right-3 top-1/2 flex h-11 w-11 -translate-y-1/2 items-center justify-center rounded-full bg-slate-950 text-white shadow-lg transition hover:bg-indigo-600 active:scale-95 disabled:bg-slate-300"
          >
            <Send className="h-4 w-4" />
          </button>
        </div>
      </div>
    </div>
  );

  const manuscriptPane = (
    <ManuscriptPane
      viewer={viewer}
      isLoading={isViewerLoading}
      viewerMode={viewerMode}
      scrollProgress={scrollProgress}
      activeBlock={activeBlock}
      previewBlockId={previewBlockId}
      activeSectionCueBlockId={activeSectionCueBlockId}
      previewSectionCueBlockId={previewSectionCueBlockId}
      linkedSectionCueBlockId={linkedSectionCueBlockId}
      linkedCueBlockId={linkedCueBlockId}
      flashBlockId={flashBlockId}
      activePdfTarget={activePdfTarget}
      previewPdfTarget={previewPdfTarget}
      linkedPdfTarget={linkedPdfTarget}
      outlineItems={sectionOutlineItems}
      activeSectionBlockId={activeSectionBlockId}
      showOutlineRail={viewportMode === 'desktop' && viewerMode === 'manuscript'}
      outlineSheetOpen={isOutlineOpen}
      manuscriptScrollContainerRef={manuscriptScrollRef}
      pdfScrollContainerRef={pdfScrollRef}
      onClearActiveEvidence={clearActiveEvidence}
      onViewerModeChange={setViewerMode}
      onPdfDocumentStateChange={({ pageCount }) => setPdfReadyPageCount(pageCount)}
      onOpenOutlineSheet={() => setIsOutlineOpen(true)}
      onOutlineSheetClose={() => setIsOutlineOpen(false)}
      onSelectOutlineItem={handleSelectOutlineItem}
      onManuscriptScroll={handleManuscriptScroll}
      onPdfScroll={handlePdfScroll}
    />
  );

  if (isMobile) {
    return (
      <div ref={workspaceRootRef} className="flex h-full w-full flex-col bg-white">
        <div className="glass-header z-40 border-b border-slate-200/70 px-4 py-3 shadow-sm sm:px-6">
          <div className="flex items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={onBack}
                className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-4 py-2 text-[0.68rem] font-bold uppercase tracking-[0.2em] text-slate-700 shadow-scholar-sm transition hover:border-indigo-200 hover:text-indigo-600"
              >
                <ChevronLeft className="h-4 w-4" />
                Workspace
              </button>
              <button
                type="button"
                onClick={onOpenGlobalSearch}
                className="inline-flex h-10 w-10 items-center justify-center rounded-full border border-slate-200 bg-white text-slate-600 shadow-scholar-sm transition hover:border-indigo-200 hover:text-indigo-600"
                aria-label="Open global search"
              >
                <Search className="h-4 w-4" />
              </button>
              <a
                href={paperZoteroPageUrl(paper.paper_id)}
                target="_blank"
                rel="noreferrer"
                className="inline-flex h-10 w-10 items-center justify-center rounded-full border border-slate-200 bg-white text-slate-600 shadow-scholar-sm transition hover:border-indigo-200 hover:text-indigo-600"
                aria-label="Save to Zotero"
              >
                <BookMarked className="h-4 w-4" />
              </a>
            </div>
            <div className="inline-flex rounded-full border border-slate-200 bg-white p-1 shadow-scholar-sm">
              <button
                type="button"
                onClick={() => setMobilePane('chat')}
                className={`rounded-full px-3 py-1.5 text-[0.68rem] font-bold uppercase tracking-[0.18em] transition ${
                  mobilePane === 'chat' ? 'bg-slate-950 text-white' : 'text-slate-500 hover:text-indigo-600'
                }`}
              >
                Chat
              </button>
              <button
                type="button"
                onClick={() => setMobilePane('manuscript')}
                className={`rounded-full px-3 py-1.5 text-[0.68rem] font-bold uppercase tracking-[0.18em] transition ${
                  mobilePane === 'manuscript' ? 'bg-slate-950 text-white' : 'text-slate-500 hover:text-indigo-600'
                }`}
              >
                Manuscript
              </button>
            </div>
          </div>
        </div>

        <div className="min-h-0 flex-1">
          <div className={mobilePane === 'chat' ? 'flex h-full min-h-0' : 'hidden'}>{renderChatPane(false)}</div>
          <div className={mobilePane === 'manuscript' ? 'flex h-full min-h-0' : 'hidden'}>{manuscriptPane}</div>
        </div>
      </div>
    );
  }

  return (
    <div ref={workspaceRootRef} className="flex h-full w-full bg-white" style={{ userSelect: isDragging ? 'none' : 'auto' }}>
      <div className="min-h-0" style={{ width: `${chatPaneWidthPct}%` }}>
        {renderChatPane(true)}
      </div>

      {viewportMode === 'desktop' ? (
        <div
          ref={dragHandleRef}
          onPointerDown={startDragging}
          onLostPointerCapture={stopDragging}
          className="z-50 flex w-1.5 cursor-col-resize items-center justify-center bg-slate-100 transition hover:bg-indigo-200 active:bg-indigo-500"
        >
          <GripVertical className="h-4 w-4 text-slate-300" />
        </div>
      ) : null}

      <div className="relative min-h-0 min-w-0" style={{ width: `${manuscriptPaneWidthPct}%` }}>
        {manuscriptPane}
      </div>
    </div>
  );
}
