/**
 * Browser-side view of the broker contract, mirroring ../messages.schema.json.
 *
 * The frontend never speaks to Redis: it receives these payloads from the web-app
 * backend as Server-Sent Events, so only the event half of the contract is modelled
 * here. Optional fields are `?` rather than `| null` because a writer omits an unset
 * optional instead of sending an explicit null — see the `to_dict` note in
 * ../python/messages.py.
 */

/** Coarse pipeline phase. Monotonic: a job never moves backwards. */
export type Stage =
  | "queued"
  | "resolving"
  | "submitted"
  | "rendering"
  | "downloading"
  | "finalizing";

/**
 * Which worker pool a job is for (`Task` in ../python/streams.py). A job without one is
 * `generate`.
 */
export type Task = "generate" | "character_swap" | "character_swap_util";

/** `completed`, `failed` and `cancelled` are terminal. */
export type JobStatusName =
  "queued" | "running" | "completed" | "failed" | "cancelled";

export type ErrorCode =
  | "shot_not_found"
  | "invalid_shot"
  | "provider_rejected"
  | "provider_timeout"
  | "provider_unavailable"
  | "quota_exceeded"
  | "content_rejected"
  | "download_failed"
  | "internal_error"
  | "cancelled";

/**
 * Where the finished clip is.
 *
 * The index signature is deliberate: a provider may report artifacts this contract has
 * never heard of (a thumbnail, a second variant), and they arrive as ordinary members.
 * `knownOutputFields` lets a UI list the rest generically instead of dropping them.
 */
export interface Output {
  filename: string;
  /** Fetchable URL when the provider or a bucket exposes one; may expire. */
  url?: string;
  /** Path relative to the Research pipeline's output dir, when downloaded locally. */
  path?: string;
  bytes?: number;
  duration_seconds?: number;
  resolution?: string;
  [key: string]: unknown;
}

export const knownOutputFields = [
  "filename",
  "url",
  "path",
  "bytes",
  "duration_seconds",
  "resolution",
] as const;

/** Output members this contract version does not name, for generic display. */
export function extraOutputFields(output: Output): [string, unknown][] {
  const known = new Set<string>(knownOutputFields);
  return Object.entries(output).filter(
    ([key, value]) => !known.has(key) && value != null,
  );
}

export interface ProviderInfo {
  name: string;
  model?: string;
  /** Provider-side task id, for chasing a render in their dashboard. */
  request_id?: string;
}

export interface ErrorInfo {
  code: ErrorCode;
  /** Safe to render; the writer guarantees no keys or raw provider bodies. */
  message: string;
  retryable: boolean;
}

interface EventBase {
  contract_version: string;
  /** Correlation id shared by every event about one job. */
  job_id: string;
  project_id?: string;
  shot_id?: string;
  emitted_at: string;
}

/** Advisory. Dropping one makes the UI jumpier but never changes the outcome. */
export interface ProgressEvent extends EventBase {
  kind: "event.progress";
  stage: Stage;
  percent: number;
  message?: string;
  attempt: number;
}

/** Terminal and authoritative, even if progress events were lost. */
export interface CompletedEvent extends EventBase {
  kind: "event.completed";
  output: Output;
  provider?: ProviderInfo;
  started_at?: string;
  finished_at?: string;
}

/** Terminal. Published only once the worker has exhausted its retries. */
export interface FailedEvent extends EventBase {
  kind: "event.failed";
  error: ErrorInfo;
  attempts: number;
}

export type JobEvent = ProgressEvent | CompletedEvent | FailedEvent;

/**
 * Snapshot of a job, from `GET /api/jobs/:id`. Backed by the Redis hash described as
 * `$defs.JobStatus` in the schema. A late subscriber reads this first, then tails
 * events from `last_event_id` so it misses nothing and repeats nothing.
 */
export interface JobSnapshot {
  job_id: string;
  status: JobStatusName;
  project_id: string;
  shot_id: string;
  /** Absent on jobs enqueued by a 1.1 producer, which were all `generate`. */
  task?: Task | string;
  stage?: Stage;
  percent?: number;
  message?: string;
  attempt?: number;
  output?: Output;
  /** What actually ran, which may differ from what the shot file asked for. */
  provider?: ProviderInfo;
  error?: ErrorInfo;
  requested_by?: string;
  created_at: string;
  updated_at: string;
  /** Opaque; hand back as `Last-Event-ID` to resume a tail with no gap. */
  last_event_id?: string;
}

/** SSE event names used by `GET /api/jobs/:id/events`. */
export const SSE_EVENTS = {
  progress: "progress",
  completed: "completed",
  failed: "failed",
  /** Sent when the server's read blocks with nothing new, to keep proxies from idling out. */
  heartbeat: "heartbeat",
} as const;

export const TERMINAL_STATUSES: readonly JobStatusName[] = [
  "completed",
  "failed",
  "cancelled",
];

export const isTerminal = (status: JobStatusName): boolean =>
  TERMINAL_STATUSES.includes(status);

export const isCompleted = (event: JobEvent): event is CompletedEvent =>
  event.kind === "event.completed";

export const isFailed = (event: JobEvent): event is FailedEvent =>
  event.kind === "event.failed";

export const isProgress = (event: JobEvent): event is ProgressEvent =>
  event.kind === "event.progress";
