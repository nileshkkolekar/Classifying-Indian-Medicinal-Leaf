import { useCallback, useEffect, useState } from "react";

import {
  ApiError,
  deleteJob,
  fetchJob,
  fetchJobResultsCsv,
  listJobs,
  parseCsv,
  predictBatchNow,
  submitJob,
} from "../api";
import { TERMINAL_STATES, type BatchSummary, type JobStatus } from "../types";
import ResultsTable, { fromCsv, fromPredictions, type Row } from "./ResultsTable";

const POLL_MS = 1000;

interface Props {
  reviewThreshold: number;
  unknownThreshold: number;
}

function Summary({ summary }: { summary: BatchSummary }) {
  const flagged =
    summary.needs_review + summary.unable_to_classify + summary.errors;
  return (
    <>
      <dl className="metrics">
        <div className="metric">
          <dt>Images</dt>
          <dd>{summary.total}</dd>
        </div>
        <div className="metric">
          <dt>Classified</dt>
          <dd>{summary.classified}</dd>
        </div>
        <div className="metric">
          <dt>Needs review</dt>
          <dd>{summary.needs_review}</dd>
        </div>
        <div className="metric">
          <dt>Unable</dt>
          <dd>{summary.unable_to_classify}</dd>
        </div>
        <div className="metric">
          <dt>Errors</dt>
          <dd>{summary.errors}</dd>
        </div>
      </dl>
      {flagged > 0 ? (
        <div className="alert warn">
          {flagged} of {summary.total} results need a human look — highlighted below.
        </div>
      ) : (
        summary.total > 0 && (
          <div className="alert ok">
            All {summary.total} images classified above the review threshold.
          </div>
        )
      )}
    </>
  );
}

export default function BulkJobs({ reviewThreshold, unknownThreshold }: Props) {
  const [queued, setQueued] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [activeId, setActiveId] = useState<string | null>(null);
  const [job, setJob] = useState<JobStatus | null>(null);
  const [rows, setRows] = useState<Row[] | null>(null);
  const [csv, setCsv] = useState<string | null>(null);

  const [immediate, setImmediate] = useState<{
    rows: Row[];
    summary: BatchSummary;
  } | null>(null);

  const [history, setHistory] = useState<JobStatus[]>([]);

  const refreshHistory = useCallback(async () => {
    try {
      setHistory((await listJobs(10)).jobs);
    } catch {
      /* history is a convenience; never let it break the page */
    }
  }, []);

  useEffect(() => {
    void refreshHistory();
  }, [refreshHistory]);

  // Poll while a job runs. The interval is torn down on a terminal state or
  // when the component unmounts, so navigating away stops the requests.
  useEffect(() => {
    if (!activeId) return;
    let cancelled = false;

    async function tick() {
      try {
        const status = await fetchJob(activeId!);
        if (cancelled) return;
        setJob(status);

        if (TERMINAL_STATES.includes(status.state)) {
          clearInterval(timer);
          void refreshHistory();
          if (status.state !== "failed") {
            const text = await fetchJobResultsCsv(activeId!);
            if (cancelled) return;
            setCsv(text);
            setRows(fromCsv(parseCsv(text)));
          }
        }
      } catch (exception) {
        if (cancelled) return;
        clearInterval(timer);
        setError(
          exception instanceof ApiError ? exception.message : "Lost contact with the job.",
        );
      }
    }

    const timer = window.setInterval(() => void tick(), POLL_MS);
    void tick();

    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [activeId, refreshHistory]);

  function reset() {
    setActiveId(null);
    setJob(null);
    setRows(null);
    setCsv(null);
    setImmediate(null);
    setError(null);
  }

  async function handleFile(file: File) {
    setBusy(true);
    setError(null);
    setImmediate(null);
    try {
      if (queued) {
        const accepted = await submitJob(file, reviewThreshold, unknownThreshold);
        setActiveId(accepted.job_id);
      } else {
        const result = await predictBatchNow(file, reviewThreshold, unknownThreshold);
        setImmediate({
          rows: fromPredictions(result.results),
          summary: result.summary,
        });
      }
    } catch (exception) {
      setError(exception instanceof ApiError ? exception.message : "Upload failed.");
    } finally {
      setBusy(false);
    }
  }

  async function cancel() {
    if (!activeId) return;
    try {
      await deleteJob(activeId);
    } finally {
      reset();
      void refreshHistory();
    }
  }

  function download() {
    if (!csv || !activeId) return;
    // A blob URL keeps the CSV out of the address bar and works for any size.
    const url = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `leaf_results_${activeId.slice(0, 8)}.csv`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  const running = job !== null && !TERMINAL_STATES.includes(job.state);
  const progress = job && job.total > 0 ? job.processed / job.total : 0;

  return (
    <div className="stack">
      {!activeId && !immediate && (
        <div className="panel stack">
          <div className="row">
            <label style={{ margin: 0 }}>
              <input
                type="radio"
                name="mode"
                checked={queued}
                onChange={() => setQueued(true)}
              />{" "}
              Queued — large archives
            </label>
            <label style={{ margin: 0 }}>
              <input
                type="radio"
                name="mode"
                checked={!queued}
                onChange={() => setQueued(false)}
              />{" "}
              Immediate — small archives
            </label>
          </div>

          <p className="hint">
            {queued
              ? "Runs in the background and is collected when it finishes — the only way through thousands of images without the request timing out."
              : "Answered in a single response. Capped far lower."}
          </p>

          <div>
            <label htmlFor="bulk-file">ZIP archive</label>
            <input
              id="bulk-file"
              type="file"
              accept=".zip,application/zip"
              onChange={(e) => {
                const chosen = e.target.files?.[0];
                if (chosen) void handleFile(chosen);
              }}
            />
          </div>

          {busy && <p className="hint">{queued ? "Uploading and queueing…" : "Classifying…"}</p>}
          {error && <div className="alert">{error}</div>}
        </div>
      )}

      {/* ── A queued job ───────────────────────────────────────────── */}
      {job && (
        <div className="panel stack">
          <div className="row" style={{ justifyContent: "space-between" }}>
            <span className={`badge ${job.state}`}>{job.state}</span>
            <span className="hint">
              {job.processed} of {job.total} images
            </span>
          </div>

          {running && (
            <>
              <div className="meter">
                <span style={{ width: `${progress * 100}%` }} />
              </div>
              <div className="row">
                <button className="btn secondary" onClick={() => void cancel()}>
                  Cancel job
                </button>
              </div>
            </>
          )}

          {job.state === "failed" && <div className="alert">{job.error}</div>}
          {job.state === "cancelled" && (
            <div className="alert warn">
              Cancelled. Rows completed before cancelling are still below.
            </div>
          )}

          <Summary summary={job.summary} />

          {rows && (
            <>
              <ResultsTable rows={rows} />
              <div className="row">
                <button className="btn" onClick={download} disabled={!csv}>
                  Download CSV
                </button>
                <button className="btn secondary" onClick={reset}>
                  Classify another archive
                </button>
              </div>
            </>
          )}

          {!running && !rows && (
            <button className="btn secondary" onClick={reset}>
              Start over
            </button>
          )}
        </div>
      )}

      {/* ── An immediate batch ─────────────────────────────────────── */}
      {immediate && (
        <div className="panel stack">
          <Summary summary={immediate.summary} />
          <ResultsTable rows={immediate.rows} />
          <button className="btn secondary" onClick={reset}>
            Classify another archive
          </button>
        </div>
      )}

      {/* ── History ────────────────────────────────────────────────── */}
      {!activeId && history.length > 0 && (
        <div className="panel">
          <h3 style={{ margin: "0 0 10px", fontSize: "0.95rem" }}>Recent jobs</h3>
          {history.map((entry) => (
            <div className="job-row" key={entry.job_id}>
              <span className="id">{entry.job_id.slice(0, 8)}</span>
              <span>{entry.filename}</span>
              <span className={`badge ${entry.state}`}>{entry.state}</span>
              <button
                className="btn secondary"
                onClick={() => {
                  reset();
                  setActiveId(entry.job_id);
                }}
              >
                Open
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
