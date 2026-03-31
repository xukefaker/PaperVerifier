const STORAGE_KEYS = {
  currentProjectId: 'papersearchagent.workspace.current_project_id',
} as const;

function canUseStorage(): boolean {
  return typeof window !== 'undefined' && typeof window.localStorage !== 'undefined';
}

export function loadCurrentProjectId(): string | null {
  if (!canUseStorage()) {
    return null;
  }
  try {
    const value = window.localStorage.getItem(STORAGE_KEYS.currentProjectId)?.trim();
    return value ? value : null;
  } catch {
    return null;
  }
}

export function saveCurrentProjectId(projectId: string): void {
  if (!canUseStorage()) {
    return;
  }
  try {
    window.localStorage.setItem(STORAGE_KEYS.currentProjectId, projectId);
  } catch {}
}

export function clearCurrentProjectId(): void {
  if (!canUseStorage()) {
    return;
  }
  try {
    window.localStorage.removeItem(STORAGE_KEYS.currentProjectId);
  } catch {}
}
