'use client';

import { useEffect, useMemo, useRef, useState, type RefObject } from 'react';
import type { PDFDocumentProxy } from 'pdfjs-dist/types/src/display/api';

import type { EvidenceNavigationPdfPageTarget, EvidenceNavigationPdfTarget } from '@/lib/types';

type PaperPdfViewerProps = {
  pdfUrl: string | null;
  scrollContainerRef: RefObject<HTMLDivElement | null>;
  activeTarget?: EvidenceNavigationPdfTarget | null;
  previewTarget?: EvidenceNavigationPdfTarget | null;
  linkedTarget?: EvidenceNavigationPdfTarget | null;
  onDocumentStateChange?: (state: { isReady: boolean; pageCount: number }) => void;
};

type PdfHighlightMode = 'active' | 'preview' | 'linked';

type PdfPageHighlight = {
  mode: PdfHighlightMode;
  target: EvidenceNavigationPdfPageTarget;
};

type PdfPageMetric = {
  pageNumber: number;
  width: number;
  height: number;
};

type PdfPageSlotProps = {
  pdfDocument: PDFDocumentProxy;
  metric: PdfPageMetric;
  scrollContainerRef: RefObject<HTMLDivElement | null>;
  shouldPrioritize: boolean;
  activeTarget?: EvidenceNavigationPdfTarget | null;
  previewTarget?: EvidenceNavigationPdfTarget | null;
  linkedTarget?: EvidenceNavigationPdfTarget | null;
};

let pdfJsModulePromise: Promise<typeof import('pdfjs-dist/legacy/build/pdf.mjs')> | null = null;

export function paperPdfPageDomId(pageNumber: number): string {
  return `viewer-pdf-page-${pageNumber}`;
}

const PDF_WORKER_URL = '/api/pdf-worker';

async function loadPdfJsModule() {
  if (!pdfJsModulePromise) {
    pdfJsModulePromise = import('pdfjs-dist/legacy/build/pdf.mjs').then((module) => {
      module.GlobalWorkerOptions.workerSrc = PDF_WORKER_URL;
      return module;
    });
  }
  return pdfJsModulePromise;
}

function resolvePageHighlight(
  pageNumber: number,
  activeTarget?: EvidenceNavigationPdfTarget | null,
  previewTarget?: EvidenceNavigationPdfTarget | null,
  linkedTarget?: EvidenceNavigationPdfTarget | null,
): PdfPageHighlight | null {
  const activePageTarget = activeTarget?.pages.find((page) => page.page === pageNumber);
  if (activePageTarget) {
    return { mode: 'active', target: activePageTarget };
  }

  const previewPageTarget = previewTarget?.pages.find((page) => page.page === pageNumber);
  if (previewPageTarget) {
    return { mode: 'preview', target: previewPageTarget };
  }

  const linkedPageTarget = linkedTarget?.pages.find((page) => page.page === pageNumber);
  if (linkedPageTarget) {
    return { mode: 'linked', target: linkedPageTarget };
  }

  return null;
}

function highlightContainerClassName(mode: PdfHighlightMode): string {
  if (mode === 'active') {
    return 'border-indigo-300 bg-indigo-100/90 shadow-[0_0_0_1px_rgba(99,102,241,0.18),0_18px_42px_rgba(79,70,229,0.14)]';
  }
  if (mode === 'preview') {
    return 'border-indigo-200 bg-indigo-100/65 shadow-[0_0_0_1px_rgba(129,140,248,0.14),0_14px_32px_rgba(99,102,241,0.10)]';
  }
  return 'border-sky-200 bg-sky-100/60 shadow-[0_0_0_1px_rgba(125,211,252,0.16),0_12px_28px_rgba(14,165,233,0.10)]';
}

function PdfPageSlot({
  pdfDocument,
  metric,
  scrollContainerRef,
  shouldPrioritize,
  activeTarget,
  previewTarget,
  linkedTarget,
}: PdfPageSlotProps) {
  const slotRef = useRef<HTMLDivElement | null>(null);
  const frameRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [containerWidth, setContainerWidth] = useState(0);
  const [isNearViewport, setIsNearViewport] = useState(false);
  const [hasRenderedOnce, setHasRenderedOnce] = useState(false);
  const [renderedSize, setRenderedSize] = useState({ width: 0, height: 0 });
  const [renderError, setRenderError] = useState<string | null>(null);

  const pageHighlight = useMemo(
    () => resolvePageHighlight(metric.pageNumber, activeTarget, previewTarget, linkedTarget),
    [activeTarget, linkedTarget, metric.pageNumber, previewTarget],
  );

  const shouldRenderPage = shouldPrioritize || isNearViewport || hasRenderedOnce;

  useEffect(() => {
    const element = slotRef.current;
    const root = scrollContainerRef.current;
    if (!(element instanceof HTMLElement) || !(root instanceof HTMLElement)) {
      return;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        const [entry] = entries;
        if (entry?.isIntersecting) {
          setIsNearViewport(true);
          return;
        }
        setIsNearViewport(false);
      },
      {
        root,
        rootMargin: '1400px 0px 1400px 0px',
        threshold: 0.01,
      },
    );

    observer.observe(element);
    return () => observer.disconnect();
  }, [scrollContainerRef]);

  useEffect(() => {
    if (shouldPrioritize) {
      setIsNearViewport(true);
    }
  }, [shouldPrioritize]);

  useEffect(() => {
    const element = frameRef.current;
    if (!(element instanceof HTMLElement)) {
      return;
    }

    const updateWidth = () => {
      setContainerWidth((current) => {
        const nextWidth = Math.max(1, Math.floor(element.clientWidth));
        return current === nextWidth ? current : nextWidth;
      });
    };

    updateWidth();
    const observer = new ResizeObserver(() => updateWidth());
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!shouldRenderPage || containerWidth <= 0) {
      return;
    }

    let cancelled = false;
    let activeRenderTask: { cancel: () => void; promise: Promise<void> } | null = null;

    const renderPage = async () => {
      try {
        setRenderError(null);
        const page = await pdfDocument.getPage(metric.pageNumber);
        if (cancelled) {
          return;
        }

        const naturalViewport = page.getViewport({ scale: 1 });
        const scale = containerWidth / naturalViewport.width;
        const viewport = page.getViewport({ scale });
        const canvas = canvasRef.current;
        if (!(canvas instanceof HTMLCanvasElement)) {
          return;
        }

        const context = canvas.getContext('2d', { alpha: false });
        if (!context) {
          throw new Error('Canvas 2D context is unavailable.');
        }

        const outputScale = window.devicePixelRatio || 1;
        canvas.width = Math.max(1, Math.floor(viewport.width * outputScale));
        canvas.height = Math.max(1, Math.floor(viewport.height * outputScale));
        canvas.style.width = `${viewport.width}px`;
        canvas.style.height = `${viewport.height}px`;

        context.setTransform(1, 0, 0, 1, 0, 0);
        context.clearRect(0, 0, canvas.width, canvas.height);
        context.setTransform(outputScale, 0, 0, outputScale, 0, 0);

        activeRenderTask = page.render({
          canvas,
          canvasContext: context,
          viewport,
        });
        await activeRenderTask.promise;
        if (cancelled) {
          return;
        }
        setRenderedSize({ width: viewport.width, height: viewport.height });
        setHasRenderedOnce(true);
      } catch (error) {
        if (cancelled) {
          return;
        }
        const name = error instanceof Error ? error.name : '';
        if (name === 'RenderingCancelledException') {
          return;
        }
        setRenderError('PDF page rendering failed.');
      }
    };

    void renderPage();
    return () => {
      cancelled = true;
      activeRenderTask?.cancel();
    };
  }, [containerWidth, metric.pageNumber, pdfDocument, shouldRenderPage]);

  return (
    <article
      id={paperPdfPageDomId(metric.pageNumber)}
      data-page-number={metric.pageNumber}
      className="rounded-[1.9rem] border border-slate-200/85 bg-white/90 p-4 shadow-scholar-sm"
    >
      <div className="mb-3 flex items-center justify-between gap-3 px-1">
        <div className="text-[0.64rem] font-bold uppercase tracking-[0.2em] text-slate-400">PDF page</div>
        <div className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-[0.66rem] font-bold uppercase tracking-[0.18em] text-slate-500">
          P.{metric.pageNumber}
        </div>
      </div>

      <div
        ref={slotRef}
        className="relative overflow-hidden rounded-[1.45rem] border border-slate-200 bg-slate-100"
      >
        <div
          ref={frameRef}
          className="relative w-full"
          style={{ aspectRatio: `${metric.width} / ${metric.height}` }}
        >
          {shouldRenderPage ? (
            <canvas ref={canvasRef} className="block w-full" />
          ) : (
            <div className="skeleton-pulse absolute inset-0" />
          )}

          {!shouldRenderPage ? (
            <div className="pointer-events-none absolute inset-x-6 bottom-5 rounded-full border border-white/80 bg-white/85 px-4 py-2 text-center text-[0.68rem] font-bold uppercase tracking-[0.18em] text-slate-500 shadow-scholar-sm backdrop-blur">
              Page queued for on-demand rendering
            </div>
          ) : null}

          {renderError ? (
            <div className="absolute inset-0 flex items-center justify-center bg-white/92 px-6 text-center text-[0.92rem] text-slate-500">
              {renderError}
            </div>
          ) : null}

          {pageHighlight && renderedSize.width > 0 && renderedSize.height > 0 ? (
            <div className="pointer-events-none absolute inset-0">
              {pageHighlight.target.bboxes.map((bbox, index) => {
                const originalWidth = pageHighlight.target.width || renderedSize.width;
                const originalHeight = pageHighlight.target.height || renderedSize.height;
                const left = (bbox.x0 / originalWidth) * renderedSize.width;
                const top = (bbox.y0 / originalHeight) * renderedSize.height;
                const width = ((bbox.x1 - bbox.x0) / originalWidth) * renderedSize.width;
                const height = ((bbox.y1 - bbox.y0) / originalHeight) * renderedSize.height;

                return (
                  <div
                    key={`pdf-highlight-${metric.pageNumber}-${index}`}
                    className={`absolute rounded-[0.9rem] border transition-all duration-300 ${highlightContainerClassName(pageHighlight.mode)} ${
                      pageHighlight.mode === 'active' ? 'animate-pulse' : ''
                    }`}
                    style={{
                      left: `${left}px`,
                      top: `${top}px`,
                      width: `${Math.max(width, 10)}px`,
                      height: `${Math.max(height, 10)}px`,
                    }}
                  />
                );
              })}
            </div>
          ) : null}
        </div>
      </div>
    </article>
  );
}

export function PaperPdfViewer({
  pdfUrl,
  scrollContainerRef,
  activeTarget,
  previewTarget,
  linkedTarget,
  onDocumentStateChange,
}: PaperPdfViewerProps) {
  const [pdfDocument, setPdfDocument] = useState<PDFDocumentProxy | null>(null);
  const [pageMetrics, setPageMetrics] = useState<PdfPageMetric[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  useEffect(() => {
    onDocumentStateChange?.({
      isReady: Boolean(pdfDocument && pageMetrics.length > 0),
      pageCount: pageMetrics.length,
    });
  }, [onDocumentStateChange, pageMetrics.length, pdfDocument]);

  useEffect(() => {
    let cancelled = false;
    let activeDocument: PDFDocumentProxy | null = null;
    let activeLoadingTask: { destroy: () => void; promise: Promise<PDFDocumentProxy> } | null = null;

    const loadDocument = async () => {
      if (!pdfUrl) {
        setPdfDocument(null);
        setPageMetrics([]);
        setErrorMessage(null);
        setIsLoading(false);
        return;
      }

      setPdfDocument(null);
      setPageMetrics([]);
      setErrorMessage(null);
      setIsLoading(true);

      try {
        const pdfjs = await loadPdfJsModule();
        if (cancelled) {
          return;
        }

        activeLoadingTask = pdfjs.getDocument({ url: pdfUrl });
        activeDocument = await activeLoadingTask.promise;
        if (cancelled) {
          await activeDocument.destroy();
          return;
        }

        const metrics: PdfPageMetric[] = [];
        for (let pageNumber = 1; pageNumber <= activeDocument.numPages; pageNumber += 1) {
          const page = await activeDocument.getPage(pageNumber);
          if (cancelled) {
            await activeDocument.destroy();
            return;
          }
          const viewport = page.getViewport({ scale: 1 });
          metrics.push({
            pageNumber,
            width: viewport.width,
            height: viewport.height,
          });
        }

        if (cancelled) {
          await activeDocument.destroy();
          return;
        }

        setPdfDocument(activeDocument);
        setPageMetrics(metrics);
      } catch (error) {
        console.error('PaperPdfViewer failed to load PDF document', error);
        if (!cancelled) {
          setErrorMessage(
            error instanceof Error && error.message
              ? `The PDF viewer could not load this paper. ${error.message}`
              : 'The PDF viewer could not load this paper.',
          );
        }
      } finally {
        if (!cancelled) {
          setIsLoading(false);
        }
      }
    };

    void loadDocument();
    return () => {
      cancelled = true;
      activeLoadingTask?.destroy();
      void activeDocument?.destroy();
    };
  }, [pdfUrl]);

  const prioritizedPages = useMemo(() => {
    const pageSet = new Set<number>();
    for (const page of activeTarget?.pages ?? []) {
      pageSet.add(page.page);
    }
    for (const page of previewTarget?.pages ?? []) {
      pageSet.add(page.page);
    }
    for (const page of linkedTarget?.pages ?? []) {
      pageSet.add(page.page);
    }
    return pageSet;
  }, [activeTarget, linkedTarget, previewTarget]);

  if (!pdfUrl) {
    return (
      <div className="mx-auto flex max-w-[780px] flex-col items-center justify-center px-6 py-20 text-center">
        <div className="rounded-[2rem] border border-slate-200/85 bg-white/92 px-10 py-12 shadow-scholar-sm">
          <div className="text-[0.72rem] font-bold uppercase tracking-[0.22em] text-slate-400">PDF</div>
          <div className="mt-3 text-[1rem] leading-7 text-slate-500">No PDF artifact is available for this paper.</div>
        </div>
      </div>
    );
  }

  if (errorMessage) {
    return (
      <div className="mx-auto flex max-w-[780px] flex-col items-center justify-center px-6 py-20 text-center">
        <div className="rounded-[2rem] border border-slate-200/85 bg-white/92 px-10 py-12 shadow-scholar-sm">
          <div className="text-[0.72rem] font-bold uppercase tracking-[0.22em] text-slate-400">PDF</div>
          <div className="mt-3 text-[1rem] leading-7 text-slate-500">{errorMessage}</div>
        </div>
      </div>
    );
  }

  if (isLoading && (!pdfDocument || pageMetrics.length === 0)) {
    return (
      <div className="mx-auto max-w-[1080px] space-y-6 pb-24">
        {Array.from({ length: 3 }, (_, index) => (
          <div
            key={`pdf-loading-${index}`}
            className="overflow-hidden rounded-[1.9rem] border border-slate-200/85 bg-white/92 p-4 shadow-scholar-sm"
          >
            <div className="mb-3 flex items-center justify-between gap-3 px-1">
              <div className="skeleton-pulse h-3 w-20 rounded-full" />
              <div className="skeleton-pulse h-7 w-16 rounded-full" />
            </div>
            <div className="skeleton-pulse h-[520px] w-full rounded-[1.45rem]" />
          </div>
        ))}
      </div>
    );
  }

  if (!pdfDocument || pageMetrics.length === 0) {
    return (
      <div className="mx-auto flex max-w-[780px] flex-col items-center justify-center px-6 py-20 text-center">
        <div className="rounded-[2rem] border border-slate-200/85 bg-white/92 px-10 py-12 shadow-scholar-sm">
          <div className="text-[0.72rem] font-bold uppercase tracking-[0.22em] text-slate-400">PDF</div>
          <div className="mt-3 text-[1rem] leading-7 text-slate-500">The PDF viewer is unavailable.</div>
        </div>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-[1080px] space-y-6 pb-24">
      {pageMetrics.map((metric) => (
        <PdfPageSlot
          key={`pdf-page-${metric.pageNumber}`}
          pdfDocument={pdfDocument}
          metric={metric}
          scrollContainerRef={scrollContainerRef}
          shouldPrioritize={prioritizedPages.has(metric.pageNumber)}
          activeTarget={activeTarget}
          previewTarget={previewTarget}
          linkedTarget={linkedTarget}
        />
      ))}
    </div>
  );
}
