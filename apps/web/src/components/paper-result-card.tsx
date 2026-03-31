import { paperZoteroPageUrl } from '@/lib/client-api';
import type { PaperResult } from '@/lib/types';
import { motion } from 'framer-motion';
import { ArrowRight, BookMarked, ImageIcon, Landmark, Users } from 'lucide-react';

type PaperResultCardProps = {
  paper: PaperResult;
  onOpenPaper: (paper: PaperResult) => void;
  onQuickPeek?: (paper: PaperResult) => void;
};

export function PaperResultCard({ paper, onOpenPaper, onQuickPeek }: PaperResultCardProps) {
  const isSatisfied = paper.verdict.toLowerCase() === 'satisfied';
  const isPartial = paper.verdict.toLowerCase() === 'partial';
  const authorsText = paper.authors?.join(', ') || '';
  const affiliationsText = paper.affiliations?.join(', ') || '';
  const verdictClassName = isSatisfied
    ? 'border-emerald-100/90 bg-emerald-50/92 text-emerald-700'
    : isPartial
      ? 'border-amber-100/90 bg-amber-50/92 text-amber-700'
      : 'border-slate-200/90 bg-slate-100/92 text-slate-600';

  return (
    <motion.article
      role='button'
      tabIndex={0}
      layout
      onClick={() => onQuickPeek?.(paper)}
      onKeyDown={(event) => {
        if ((event.key === 'Enter' || event.key === ' ') && onQuickPeek) {
          event.preventDefault();
          onQuickPeek(paper);
        }
      }}
      whileHover={{
        y: -6,
        rotate: -0.25,
        boxShadow: '0 34px 70px rgba(15, 23, 42, 0.11), 0 12px 28px rgba(15, 23, 42, 0.06)',
      }}
      className='group relative flex h-full min-h-[37rem] cursor-pointer flex-col overflow-hidden rounded-[2rem] border border-white/80 bg-white/88 shadow-scholar-lg backdrop-blur-xl transition-all duration-500 hover:border-indigo-200/60 focus:outline-none focus-visible:border-indigo-300 focus-visible:ring-4 focus-visible:ring-indigo-100'
      aria-label={`Quick peek ${paper.title}`}
    >
      <div className='relative h-56 w-full overflow-hidden border-b border-slate-100/80 bg-[linear-gradient(180deg,rgba(248,250,252,0.98),rgba(241,245,249,0.92))]'>
        {paper.main_image_url ? (
          <>
            <img
              src={paper.main_image_url}
              alt='Paper Cover'
              className='h-full w-full object-cover transition-transform duration-1000 group-hover:scale-[1.03]'
            />
            <div className='absolute inset-0 bg-[linear-gradient(180deg,rgba(248,250,252,0.18),transparent_28%,rgba(15,23,42,0.22))]' />
          </>
        ) : (
          <div className='flex h-full flex-col items-center justify-center bg-[radial-gradient(circle_at_top,rgba(99,102,241,0.10),transparent_40%),linear-gradient(180deg,#f8fafc_0%,#eef2f7_100%)]'>
            <div className='mb-4 flex h-14 w-14 items-center justify-center rounded-[1.3rem] border border-white/80 bg-white/70 shadow-scholar-sm'>
              <ImageIcon className='h-6 w-6 text-slate-300' />
            </div>
            <span className='text-[0.64rem] font-bold uppercase tracking-[0.24em] text-slate-400'>Manuscript visual unavailable</span>
          </div>
        )}
      </div>

      <div className='flex flex-1 flex-col p-6'>
        <div className='mb-4 flex flex-wrap gap-2'>
          <span className='rounded-full border border-indigo-100 bg-indigo-50/80 px-3 py-1 text-[0.62rem] font-bold uppercase tracking-[0.22em] text-indigo-600'>
            {paper.venue.toUpperCase()} {paper.year} {paper.track ? `· ${paper.track}` : ''}
          </span>
          <span className={`rounded-full border px-3 py-1 text-[0.62rem] font-bold uppercase tracking-[0.22em] ${verdictClassName}`}>
            {paper.verdict}
          </span>
        </div>

        <div className='mb-4 min-h-[6rem]'>
          <h3 className='line-clamp-3 text-[1.22rem] font-black leading-[1.2] tracking-tight text-slate-900 transition-colors group-hover:text-indigo-600'>
            {paper.title}
          </h3>
        </div>

        <div className='mb-7 min-h-[5.8rem] space-y-3'>
          {authorsText ? (
            <div className='flex items-start gap-2.5 text-[0.84rem] font-medium leading-6 text-slate-600'>
              <Users className='mt-1 h-4 w-4 flex-shrink-0 text-indigo-300' />
              <span className='line-clamp-2'>{authorsText}</span>
            </div>
          ) : (
            <div className='flex items-center gap-2 text-[0.72rem] italic text-slate-300'>
              <div className='h-3 w-3 animate-pulse rounded-full bg-slate-100' />
              <span>Fetching Contributors...</span>
            </div>
          )}

          {affiliationsText ? (
            <div className='font-scholar flex items-start gap-2.5 text-[0.88rem] italic leading-6 text-slate-400'>
              <Landmark className='mt-1 h-4 w-4 flex-shrink-0 text-slate-300' />
              <span className='line-clamp-2'>{affiliationsText}</span>
            </div>
          ) : (
            <div className='flex items-center gap-2 text-[0.72rem] italic text-slate-300'>
              <div className='h-3 w-16 animate-pulse rounded bg-slate-50' />
            </div>
          )}
        </div>

        <div className='mt-auto border-t border-slate-100/90 pt-5'>
          <div className='grid gap-3'>
            <button
              type='button'
              onClick={(event) => {
                event.stopPropagation();
                onOpenPaper(paper);
              }}
              className='inline-flex w-full items-center justify-between gap-3 rounded-[1.4rem] border border-slate-200 bg-white px-5 py-3.5 text-[0.76rem] font-bold uppercase tracking-[0.22em] text-slate-800 transition-all hover:border-indigo-500 hover:bg-indigo-600 hover:text-white active:scale-[0.99]'
            >
              <span>Explore manuscript</span>
              <ArrowRight className='h-4 w-4 transition-transform group-hover:translate-x-1' />
            </button>
            <a
              href={paperZoteroPageUrl(paper.paper_id)}
              target='_blank'
              rel='noreferrer'
              onClick={(event) => {
                event.stopPropagation();
              }}
              className='inline-flex w-full items-center justify-between gap-3 rounded-[1.4rem] border border-slate-200 bg-slate-50/80 px-5 py-3.5 text-[0.76rem] font-bold uppercase tracking-[0.22em] text-slate-700 transition-all hover:border-indigo-200 hover:bg-white hover:text-indigo-600 active:scale-[0.99]'
            >
              <span>Save to Zotero</span>
              <BookMarked className='h-4 w-4' />
            </a>
          </div>
        </div>
      </div>
    </motion.article>
  );
}

export function PaperResultCardSkeleton() {
  return (
    <article className='skeleton-card-sheen relative flex h-full min-h-[37rem] flex-col overflow-hidden rounded-[2rem] border border-white/85 bg-white/92 shadow-[0_18px_40px_rgba(15,23,42,0.06)]'>
      <div className='relative h-56 overflow-hidden border-b border-slate-100/90 bg-[linear-gradient(180deg,#f8fafc_0%,#eef2f7_100%)]'>
        <div className='absolute inset-x-5 top-5 flex items-center justify-between'>
          <div className='skeleton-pulse h-7 w-28 rounded-full' />
          <div className='skeleton-pulse h-7 w-20 rounded-full' />
        </div>
        <div className='absolute left-1/2 top-1/2 h-20 w-20 -translate-x-1/2 -translate-y-1/2 rounded-[1.7rem] border border-white/80 bg-white/60 shadow-scholar-sm' />
      </div>

      <div className='flex flex-1 flex-col p-6'>
        <div className='mb-5 flex gap-2'>
          <div className='skeleton-pulse h-6 w-32 rounded-full' />
          <div className='skeleton-pulse h-6 w-20 rounded-full' />
        </div>

        <div className='mb-4 space-y-3'>
          {['w-[88%]', 'w-[76%]', 'w-[58%]'].map((width) => (
            <div key={width} className={`skeleton-pulse h-4 rounded-full ${width}`} />
          ))}
        </div>

        <div className='mb-7 min-h-[5.6rem] space-y-3'>
          <div className='flex items-start gap-2.5'>
            <div className='skeleton-pulse mt-1 h-4 w-4 rounded-full' />
            <div className='flex-1 space-y-2'>
              <div className='skeleton-pulse h-3.5 w-[92%] rounded-full' />
              <div className='skeleton-pulse h-3.5 w-[66%] rounded-full' />
            </div>
          </div>
          <div className='flex items-start gap-2.5'>
            <div className='skeleton-pulse mt-1 h-4 w-4 rounded-full' />
            <div className='flex-1 space-y-2'>
              <div className='skeleton-pulse h-3.5 w-[86%] rounded-full' />
              <div className='skeleton-pulse h-3.5 w-[58%] rounded-full' />
            </div>
          </div>
        </div>

        <div className='mt-auto border-t border-slate-100/90 pt-5'>
          <div className='skeleton-pulse h-[3.15rem] rounded-[1.4rem]' />
        </div>
      </div>
    </article>
  );
}
