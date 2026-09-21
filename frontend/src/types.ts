// Mirrors medicinal_leaf/api/schemas.py. Keep the two in step — these field
// names are the wire contract, and a rename here is a breaking change.

export type Verdict =
  | "classified"
  | "needs_review"
  | "unable_to_classify"
  | "error";

export type JobState =
  | "queued"
  | "running"
  | "succeeded"
  | "failed"
  | "cancelled";

export const TERMINAL_STATES: readonly JobState[] = [
  "succeeded",
  "failed",
  "cancelled",
];

export interface Thresholds {
  review_threshold: number;
  unknown_threshold: number;
}

export interface PredictionResult {
  filename: string;
  verdict: Verdict;
  /** Null when the service declined to name a species (FR-14). */
  label: string | null;
  /** Null only when the image could not be read at all. */
  confidence: number | null;
  needs_review: boolean;
  probabilities: Record<string, number>;
  note: string | null;
}

export interface SingleResult {
  result: PredictionResult;
  thresholds: Thresholds;
}

export interface BatchSummary {
  total: number;
  classified: number;
  needs_review: number;
  unable_to_classify: number;
  errors: number;
}

export interface BatchResult {
  results: PredictionResult[];
  summary: BatchSummary;
  thresholds: Thresholds;
}

export interface JobAccepted {
  job_id: string;
  state: JobState;
  total: number;
  status_url: string;
  results_url: string;
}

export interface JobStatus {
  job_id: string;
  state: JobState;
  filename: string;
  total: number;
  processed: number;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  summary: BatchSummary;
  thresholds: Thresholds;
  error: string | null;
}

export interface Health {
  status: string;
  model_loaded: boolean;
  backbone: string | null;
  classes: string[];
  thresholds: Thresholds;
  version: string;
}

export interface UserInfo {
  username: string;
  kind: string;
  auth_enabled: boolean;
}

export interface Token {
  access_token: string;
  token_type: string;
  expires_in: number;
  username: string;
}

/** One row of a bulk job's CSV, after parsing. */
export interface CsvRow {
  filename: string;
  verdict: Verdict;
  label: string;
  confidence: number | null;
  needs_review: boolean;
  note: string;
}
