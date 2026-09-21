import type { CsvRow, PredictionResult, Verdict } from "../types";

/** The shape both the immediate and queued paths converge on. */
export interface Row {
  filename: string;
  verdict: Verdict;
  label: string | null;
  confidence: number | null;
  needsReview: boolean;
  note: string;
}

export const fromPredictions = (results: PredictionResult[]): Row[] =>
  results.map((r) => ({
    filename: r.filename,
    verdict: r.verdict,
    label: r.label,
    confidence: r.confidence,
    needsReview: r.needs_review,
    note: r.note ?? "",
  }));

export const fromCsv = (rows: CsvRow[]): Row[] =>
  rows.map((r) => ({
    filename: r.filename,
    verdict: r.verdict,
    label: r.label || null,
    confidence: r.confidence,
    needsReview: r.needs_review,
    note: r.note,
  }));

const percent = (value: number | null): string =>
  value === null ? "—" : `${(value * 100).toFixed(1)}%`;

export default function ResultsTable({ rows }: { rows: Row[] }) {
  if (rows.length === 0) {
    return <p className="hint">No rows to show.</p>;
  }

  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th scope="col">File</th>
            <th scope="col">Species</th>
            <th scope="col">Confidence</th>
            <th scope="col">Verdict</th>
            <th scope="col">Note</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            // Filenames repeat across folders inside an archive, so the index
            // is part of the key.
            <tr key={`${row.filename}-${index}`} className={`flag-${row.verdict}`}>
              <td className="file">{row.filename}</td>
              <td>{row.label ?? "—"}</td>
              <td className="num">{percent(row.confidence)}</td>
              <td>
                <span className={`badge ${row.verdict}`}>
                  {row.verdict.replace(/_/g, " ")}
                </span>
              </td>
              <td className="hint">{row.note}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
