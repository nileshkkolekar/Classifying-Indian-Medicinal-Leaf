import { useCallback, useEffect, useState } from "react";

import { ApiError, fetchHealth, logout, tokenStore, whoami } from "./api";
import BulkJobs from "./components/BulkJobs";
import Login from "./components/Login";
import SinglePredict from "./components/SinglePredict";
import type { Health, UserInfo } from "./types";

type Tab = "single" | "bulk";
type Phase = "loading" | "locked" | "ready";

export default function App() {
  const [phase, setPhase] = useState<Phase>("loading");
  const [health, setHealth] = useState<Health | null>(null);
  const [user, setUser] = useState<UserInfo | null>(null);
  const [tab, setTab] = useState<Tab>("single");
  const [review, setReview] = useState(0.7);
  const [unknown, setUnknown] = useState(0.3);

  const bootstrap = useCallback(async () => {
    // /health is public, so it answers even when we are not signed in and
    // tells us whether a model is loaded at all.
    try {
      const status = await fetchHealth();
      setHealth(status);
      setReview(status.thresholds.review_threshold);
      setUnknown(status.thresholds.unknown_threshold);
    } catch {
      /* the session check below will surface the real problem */
    }

    try {
      setUser(await whoami());
      setPhase("ready");
    } catch (exception) {
      if (exception instanceof ApiError && exception.isUnauthorized) {
        // A stale token is worse than none: clear it so the form starts clean.
        tokenStore.clear();
        setPhase("locked");
      } else {
        setPhase("locked");
      }
    }
  }, []);

  useEffect(() => {
    void bootstrap();
  }, [bootstrap]);

  if (phase === "loading") {
    return (
      <div className="shell">
        <p className="hint">Connecting…</p>
      </div>
    );
  }

  if (phase === "locked") {
    return <Login onAuthenticated={() => void bootstrap()} />;
  }

  return (
    <div className="shell">
      <header className="masthead">
        <div>
          <h1>Medicinal Leaf Classification</h1>
          <p className="disclaimer">
            Aloevera · Amla · Mint · Neem · Tulsi — educational and botanical use
            only, not medical guidance.
          </p>
        </div>

        <div className="session">
          <span>
            <span className={`status-dot ${health?.model_loaded ? "ready" : ""}`} />
            {health?.model_loaded ? health.backbone : "no model loaded"}
          </span>
          {user?.auth_enabled && (
            <>
              <span>{user.username}</span>
              <button
                className="btn secondary"
                onClick={() => {
                  logout();
                  setPhase("locked");
                }}
              >
                Sign out
              </button>
            </>
          )}
        </div>
      </header>

      {health && !health.model_loaded && (
        <div className="alert warn" style={{ marginTop: 18 }}>
          The API is running but no model is loaded, so predictions will fail.
          Train one with <code>leaf-train fit</code>, or point
          <code> MLC_AWS__CHECKPOINT_URI</code> at a checkpoint in S3.
        </div>
      )}

      <div className="tabs" role="tablist">
        <button
          role="tab"
          aria-selected={tab === "single"}
          onClick={() => setTab("single")}
        >
          Single image
        </button>
        <button role="tab" aria-selected={tab === "bulk"} onClick={() => setTab("bulk")}>
          Bulk ZIP
        </button>
      </div>

      <div className="panel" style={{ marginBottom: 18 }}>
        <div className="row" style={{ gap: 24 }}>
          <div style={{ flex: "1 1 220px" }}>
            <label htmlFor="review">
              Flag for review below {(review * 100).toFixed(0)}%
            </label>
            <input
              id="review"
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={review}
              onChange={(e) => setReview(Number(e.target.value))}
              style={{ width: "100%" }}
            />
          </div>
          <div style={{ flex: "1 1 220px" }}>
            <label htmlFor="unknown">
              Decline to answer below {(unknown * 100).toFixed(0)}%
            </label>
            <input
              id="unknown"
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={unknown}
              onChange={(e) => setUnknown(Number(e.target.value))}
              style={{ width: "100%" }}
            />
          </div>
        </div>
        {unknown > review && (
          <div className="alert" style={{ marginTop: 12 }}>
            The decline threshold cannot exceed the review threshold.
          </div>
        )}
      </div>

      {tab === "single" ? (
        <SinglePredict
          reviewThreshold={review}
          unknownThreshold={Math.min(unknown, review)}
        />
      ) : (
        <BulkJobs
          reviewThreshold={review}
          unknownThreshold={Math.min(unknown, review)}
        />
      )}
    </div>
  );
}
