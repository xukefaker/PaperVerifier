export type WorkspaceMobilePane = 'chat' | 'manuscript';

export type WorkspaceUiState = {
  desktopSplitPct: number;
  mobilePane: WorkspaceMobilePane;
  rationaleOpen: boolean;
};

const STORAGE_KEYS = {
  desktopSplitPct: 'chemverify.workspace.desktop_split_pct',
  mobilePane: 'chemverify.workspace.mobile_pane',
  rationaleOpen: 'chemverify.workspace.rationale_open',
} as const;

const DEFAULT_STATE: WorkspaceUiState = {
  desktopSplitPct: 40,
  mobilePane: 'chat',
  rationaleOpen: true,
};

function clampDesktopSplitPct(value: number): number {
  if (!Number.isFinite(value)) {
    return DEFAULT_STATE.desktopSplitPct;
  }
  return Math.max(28, Math.min(72, value));
}

function canUseStorage(): boolean {
  return typeof window !== 'undefined' && typeof window.localStorage !== 'undefined';
}

export function loadWorkspaceUiState(): WorkspaceUiState {
  if (!canUseStorage()) {
    return DEFAULT_STATE;
  }

  try {
    const desktopSplitPct = clampDesktopSplitPct(
      Number(window.localStorage.getItem(STORAGE_KEYS.desktopSplitPct) ?? DEFAULT_STATE.desktopSplitPct),
    );
    const mobilePaneValue = window.localStorage.getItem(STORAGE_KEYS.mobilePane);
    const rationaleOpenValue = window.localStorage.getItem(STORAGE_KEYS.rationaleOpen);

    return {
      desktopSplitPct,
      mobilePane: mobilePaneValue === 'manuscript' ? 'manuscript' : DEFAULT_STATE.mobilePane,
      rationaleOpen: rationaleOpenValue == null ? DEFAULT_STATE.rationaleOpen : rationaleOpenValue === 'true',
    };
  } catch {
    return DEFAULT_STATE;
  }
}

export function saveDesktopSplitPct(value: number): void {
  if (!canUseStorage()) {
    return;
  }
  try {
    window.localStorage.setItem(STORAGE_KEYS.desktopSplitPct, String(clampDesktopSplitPct(value)));
  } catch {}
}

export function saveMobilePane(value: WorkspaceMobilePane): void {
  if (!canUseStorage()) {
    return;
  }
  try {
    window.localStorage.setItem(STORAGE_KEYS.mobilePane, value);
  } catch {}
}

export function saveRationaleOpen(value: boolean): void {
  if (!canUseStorage()) {
    return;
  }
  try {
    window.localStorage.setItem(STORAGE_KEYS.rationaleOpen, String(value));
  } catch {}
}
