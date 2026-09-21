// The single place this app talks to the server.
//
// Every request goes to the app's own origin: in production FastAPI serves
// the built bundle, and in development Vite proxies the API paths. That
// means no CORS configuration exists to get wrong, and no API base URL to
// mis-deploy.

import type {
  BatchResult,
  CsvRow,
  Health,
  JobAccepted,
  JobStatus,
  SingleResult,
  Token,
  UserInfo,
  Verdict,
} from "./types";

const TOKEN_KEY = "leaf.token";

/** Thrown for any non-2xx response, carrying the server's own message. */
export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }

  get isUnauthorized(): boolean {
    return this.status === 401;
  }
}

// sessionStorage rather than localStorage: the token dies with the tab, so a
// shared machine does not keep a live session around. Neither survives XSS —
// an httpOnly cookie would, and is the upgrade if this ever holds real data.
export const tokenStore = {
  get(): string | null {
    try {
      return sessionStorage.getItem(TOKEN_KEY);
    } catch {
      return null;
    }
  },
  set(token: string): void {
    try {
      sessionStorage.setItem(TOKEN_KEY, token);
    } catch {
      /* private mode; the session simply will not persist across reloads */
    }
  },
  clear(): void {
    try {
      sessionStorage.removeItem(TOKEN_KEY);
    } catch {
      /* nothing to do */
    }
  },
};

function authHeaders(): HeadersInit {
  const token = tokenStore.get();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function failure(response: Response): Promise<ApiError> {
  let detail = response.statusText;
  try {
    const body = await response.json();
    if (typeof body?.detail === "string") {
      detail = body.detail;
    } else if (Array.isArray(body?.detail)) {
      // FastAPI validation errors arrive as a list of objects.
      detail = body.detail.map((d: { msg?: string }) => d.msg ?? "invalid").join("; ");
    }
  } catch {
    /* not JSON; the status text will have to do */
  }
  return new ApiError(detail, response.status);
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { ...authHeaders(), ...(init.headers ?? {}) },
  });
  if (!response.ok) throw await failure(response);
  return (await response.json()) as T;
}

// ── Authentication ───────────────────────────────────────────────────────

export async function login(username: string, password: string): Promise<Token> {
  // The token endpoint is an OAuth2 password form, not JSON.
  const form = new URLSearchParams({ username, password });
  const response = await fetch("/auth/token", {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: form,
  });
  if (!response.ok) throw await failure(response);

  const token = (await response.json()) as Token;
  tokenStore.set(token.access_token);
  return token;
}

export function logout(): void {
  tokenStore.clear();
}

export const whoami = (): Promise<UserInfo> => request<UserInfo>("/auth/me");

export const fetchHealth = (): Promise<Health> => request<Health>("/health");

// ── Prediction ───────────────────────────────────────────────────────────

function thresholdQuery(review?: number, unknown?: number): string {
  const params = new URLSearchParams();
  if (review !== undefined) params.set("review_threshold", String(review));
  if (unknown !== undefined) params.set("unknown_threshold", String(unknown));
  const query = params.toString();
  return query ? `?${query}` : "";
}

export function predictOne(
  file: File,
  review?: number,
  unknown?: number,
): Promise<SingleResult> {
  const body = new FormData();
  body.append("file", file);
  return request<SingleResult>(`/predict${thresholdQuery(review, unknown)}`, {
    method: "POST",
    body,
  });
}

export function predictBatchNow(
  file: File,
  review?: number,
  unknown?: number,
): Promise<BatchResult> {
  const body = new FormData();
  body.append("file", file);
  return request<BatchResult>(`/predict/batch${thresholdQuery(review, unknown)}`, {
    method: "POST",
    body,
  });
}

// ── Jobs ─────────────────────────────────────────────────────────────────

export function submitJob(
  file: File,
  review?: number,
  unknown?: number,
): Promise<JobAccepted> {
  const body = new FormData();
  body.append("file", file);
  return request<JobAccepted>(`/jobs${thresholdQuery(review, unknown)}`, {
    method: "POST",
    body,
  });
}

export const fetchJob = (id: string): Promise<JobStatus> =>
  request<JobStatus>(`/jobs/${id}`);

export const listJobs = (limit = 10): Promise<{ jobs: JobStatus[] }> =>
  request<{ jobs: JobStatus[] }>(`/jobs?limit=${limit}`);

export const deleteJob = (id: string): Promise<unknown> =>
  request<unknown>(`/jobs/${id}`, { method: "DELETE" });

export async function fetchJobResultsCsv(id: string): Promise<string> {
  const response = await fetch(`/jobs/${id}/results`, { headers: authHeaders() });
  if (!response.ok) throw await failure(response);
  return response.text();
}

// ── CSV ──────────────────────────────────────────────────────────────────

/**
 * Parse the results CSV.
 *
 * Handles quoted fields because a filename may legitimately contain a comma;
 * splitting on "," alone would silently shift every later column.
 */
export function parseCsv(text: string): CsvRow[] {
  const lines = text.trim().split(/\r?\n/);
  if (lines.length < 2) return [];

  const header = splitCsvLine(lines[0] ?? "");
  return lines.slice(1).map((line) => {
    const cells = splitCsvLine(line);
    const get = (name: string): string => {
      const index = header.indexOf(name);
      return index >= 0 ? (cells[index] ?? "") : "";
    };
    const confidence = get("confidence");
    return {
      filename: get("filename"),
      verdict: (get("verdict") || "error") as Verdict,
      label: get("label"),
      confidence: confidence === "" ? null : Number(confidence),
      needs_review: get("needs_review").toLowerCase() === "true",
      note: get("note"),
    };
  });
}

function splitCsvLine(line: string): string[] {
  const cells: string[] = [];
  let current = "";
  let quoted = false;

  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    if (quoted) {
      if (char === '"') {
        if (line[i + 1] === '"') {
          current += '"'; // an escaped quote inside a quoted field
          i += 1;
        } else {
          quoted = false;
        }
      } else {
        current += char;
      }
    } else if (char === '"') {
      quoted = true;
    } else if (char === ",") {
      cells.push(current);
      current = "";
    } else {
      current += char;
    }
  }
  cells.push(current);
  return cells;
}
