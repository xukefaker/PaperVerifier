import { NextResponse } from 'next/server';

export type AppErrorCode =
  | 'bad_request'
  | 'not_found'
  | 'conflict'
  | 'provider_error'
  | 'not_ready'
  | 'internal_error';

export type AppErrorBody = {
  error: {
    code: AppErrorCode;
    message: string;
    fix?: string;
  };
};

const statusByCode: Record<AppErrorCode, number> = {
  bad_request: 400,
  not_found: 404,
  conflict: 409,
  provider_error: 502,
  not_ready: 503,
  internal_error: 500,
};

export function appError(code: AppErrorCode, message: string, fix?: string, status?: number) {
  const body: AppErrorBody = { error: { code, message, ...(fix ? { fix } : {}) } };
  return NextResponse.json(body, { status: status ?? statusByCode[code] });
}

export function errorMessage(error: unknown, fallback: string) {
  return error instanceof Error && error.message ? error.message : fallback;
}
