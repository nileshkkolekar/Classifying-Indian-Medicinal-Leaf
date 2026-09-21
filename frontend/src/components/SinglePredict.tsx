import { useEffect, useState } from "react";

import { ApiError, predictOne } from "../api";
import type { SingleResult } from "../types";

interface Props {
  reviewThreshold: number;
  unknownThreshold: number;
}

export default function SinglePredict({ reviewThreshold, unknownThreshold }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<SingleResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  // Object URLs are a real allocation; revoke the old one on every change or
  // the tab leaks a copy of every image the user tried.
  useEffect(() => {
    if (!file) {
      setPreview(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  async function classify(chosen: File) {
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      setResult(await predictOne(chosen, reviewThreshold, unknownThreshold));
    } catch (exception) {
      setError(
        exception instanceof ApiError ? exception.message : "Request failed.",
      );
    } finally {
      setBusy(false);
    }
  }

  function handleChange(chosen: File | null) {
    setFile(chosen);
    if (chosen) void classify(chosen);
  }

  const prediction = result?.result;
  const ranked = prediction
    ? Object.entries(prediction.probabilities).sort((a, b) => b[1] - a[1])
    : [];

  return (
    <div className="stack">
      <div className="panel stack">
        <div>
          <label htmlFor="single-file">Leaf photograph</label>
          <input
            id="single-file"
            type="file"
            accept="image/jpeg,image/png"
            onChange={(e) => handleChange(e.target.files?.[0] ?? null)}
          />
        </div>
        {busy && <p className="hint">Classifying…</p>}
        {error && <div className="alert">{error}</div>}
      </div>

      {prediction && (
        <div className="panel">
          <div className="predict-grid">
            <div>
              {preview && <img className="preview" src={preview} alt={prediction.filename} />}
            </div>

            <div>
              <span className={`badge ${prediction.verdict}`}>
                {prediction.verdict.replace(/_/g, " ")}
              </span>

              <h2 className="verdict-headline">
                {prediction.label ?? "No confident match"}
              </h2>

              {prediction.confidence !== null && (
                <>
                  <p className="hint" style={{ marginBottom: 6 }}>
                    Confidence {(prediction.confidence * 100).toFixed(1)}%
                  </p>
                  <div className="meter">
                    <span
                      style={{
                        width: `${Math.min(Math.max(prediction.confidence, 0), 1) * 100}%`,
                      }}
                    />
                  </div>
                </>
              )}

              {prediction.note && (
                <div
                  className={`alert ${prediction.verdict === "error" ? "" : "warn"}`}
                  style={{ marginTop: 14 }}
                >
                  {prediction.note}
                </div>
              )}

              {ranked.length > 0 && (
                <div className="dist">
                  {ranked.map(([species, probability]) => (
                    <div className="dist-row" key={species}>
                      <span>{species}</span>
                      <span className="meter">
                        <span style={{ width: `${probability * 100}%` }} />
                      </span>
                      <span className="value">{(probability * 100).toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              )}

              <p className="hint" style={{ marginTop: 14 }}>
                Review below {(result.thresholds.review_threshold * 100).toFixed(0)}% ·
                declines below {(result.thresholds.unknown_threshold * 100).toFixed(0)}%
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
