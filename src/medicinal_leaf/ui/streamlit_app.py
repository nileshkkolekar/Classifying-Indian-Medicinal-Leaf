"""Streamlit UI: single-image and bulk ZIP leaf classification.

    streamlit run src/medicinal_leaf/ui/streamlit_app.py

Talks to the FastAPI service over HTTP rather than loading a model directly.
That keeps one copy of the classification policy (thresholds, flagging,
declining) in the API, where it is tested — a second implementation here
would drift.

Because the calls are server-to-server from the Streamlit process, no browser
CORS configuration is involved.
"""

from __future__ import annotations

import io
import time
from typing import Any

import httpx
import pandas as pd
import streamlit as st

from medicinal_leaf.config.settings import Settings, load_settings

#: How often the browser re-polls a running job.
POLL_SECONDS = 1.0
ACTIVE_JOB_KEY = "active_job_id"
TERMINAL_STATES = {"succeeded", "failed", "cancelled"}

IMAGE_TYPES = ["jpg", "jpeg", "png"]

#: Colour and label per verdict, used for the badge and the table highlight.
VERDICT_STYLE: dict[str, tuple[str, str]] = {
    "classified": ("#1a7f37", "Classified"),
    "needs_review": ("#bf8700", "Needs review"),
    "unable_to_classify": ("#9a6700", "Unable to classify"),
    "error": ("#cf222e", "Error"),
}

ROW_TINT: dict[str, str] = {
    "classified": "",
    "needs_review": "background-color: rgba(191, 135, 0, 0.18)",
    "unable_to_classify": "background-color: rgba(154, 103, 0, 0.22)",
    "error": "background-color: rgba(207, 34, 46, 0.18)",
}


@st.cache_resource
def get_settings() -> Settings:
    """Configuration defaults; cached so every rerun does not re-read YAML."""
    return load_settings()


# ── API client ───────────────────────────────────────────────────────────


def fetch_health(base_url: str, timeout: float) -> dict[str, Any] | None:
    """Ask the API whether it is up and has a model. ``None`` if unreachable."""
    try:
        response = httpx.get(f"{base_url.rstrip('/')}/health", timeout=timeout)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError:
        return None


def post_upload(
    base_url: str,
    path: str,
    filename: str,
    payload: bytes,
    content_type: str,
    thresholds: dict[str, float],
    timeout: float,
) -> tuple[dict[str, Any] | None, str | None]:
    """POST one file. Returns ``(body, error_message)`` — exactly one is set.

    The API's own rejection messages (file too large, not a ZIP, no model
    loaded) are surfaced verbatim; they are written to be read by a user.
    """
    try:
        response = httpx.post(
            f"{base_url.rstrip('/')}{path}",
            files={"file": (filename, payload, content_type)},
            params=thresholds,
            timeout=timeout,
        )
    except httpx.HTTPError as exc:
        return None, f"Could not reach the API at {base_url}: {exc}"

    if response.is_success:
        return response.json(), None

    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    if isinstance(detail, list):  # FastAPI validation errors
        detail = "; ".join(str(item.get("msg", item)) for item in detail)
    return None, f"{response.status_code}: {detail}"


def api_get(base_url: str, path: str, timeout: float) -> tuple[Any | None, str | None]:
    """GET a JSON endpoint, returning ``(body, error_message)``."""
    try:
        response = httpx.get(f"{base_url.rstrip('/')}{path}", timeout=timeout)
    except httpx.HTTPError as exc:
        return None, f"Could not reach the API at {base_url}: {exc}"
    if response.is_success:
        return response.json(), None
    return None, _detail(response)


def api_delete(base_url: str, path: str, timeout: float) -> tuple[Any | None, str | None]:
    try:
        response = httpx.delete(f"{base_url.rstrip('/')}{path}", timeout=timeout)
    except httpx.HTTPError as exc:
        return None, f"Could not reach the API at {base_url}: {exc}"
    if response.is_success:
        return response.json(), None
    return None, _detail(response)


def fetch_results_csv(base_url: str, job_id: str, timeout: float) -> tuple[str | None, str | None]:
    """Download a finished job's results as raw CSV text."""
    try:
        response = httpx.get(f"{base_url.rstrip('/')}/jobs/{job_id}/results", timeout=timeout)
    except httpx.HTTPError as exc:
        return None, f"Could not reach the API at {base_url}: {exc}"
    if response.is_success:
        return response.text, None
    return None, _detail(response)


def _detail(response: httpx.Response) -> str:
    """Pull the API's own error message out of a failed response."""
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    if isinstance(detail, list):  # FastAPI validation errors
        detail = "; ".join(str(item.get("msg", item)) for item in detail)
    return f"{response.status_code}: {detail}"


# ── Rendering ────────────────────────────────────────────────────────────


def verdict_badge(verdict: str) -> str:
    colour, label = VERDICT_STYLE.get(verdict, ("#57606a", verdict))
    return (
        f"<span style='background:{colour};color:#fff;padding:2px 10px;"
        f"border-radius:12px;font-size:0.85rem;font-weight:600'>{label}</span>"
    )


def results_frame(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Flatten API results into the table shown and exported (FR-11)."""
    rows = [
        {
            "File": item["filename"],
            "Predicted species": item.get("label") or "—",
            "Confidence": item.get("confidence"),
            "Verdict": item["verdict"],
            "Needs review": bool(item.get("needs_review")),
            "Note": item.get("note") or "",
        }
        for item in results
    ]
    return pd.DataFrame(rows, columns=list(rows[0]) if rows else [])


def results_frame_from_csv(text: str) -> pd.DataFrame:
    """Same display shape as :func:`results_frame`, from a job's CSV.

    A queued job streams its rows to disk rather than returning JSON, so the
    two paths converge here and share one table renderer.
    """
    columns = ["File", "Predicted species", "Confidence", "Verdict", "Needs review", "Note"]
    if not text.strip():
        return pd.DataFrame(columns=columns)

    raw = pd.read_csv(io.StringIO(text), dtype=str).fillna("")
    if raw.empty:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(
        {
            "File": raw["filename"],
            "Predicted species": raw["label"].replace("", "—"),
            "Confidence": pd.to_numeric(raw["confidence"], errors="coerce"),
            "Verdict": raw["verdict"],
            "Needs review": raw["needs_review"].str.lower().isin(["true", "1"]),
            "Note": raw["note"],
        }
    )


def style_results(frame: pd.DataFrame) -> Any:
    """Tint every row that a reviewer needs to look at (FR-12)."""

    def _row_style(row: pd.Series) -> list[str]:
        return [ROW_TINT.get(row["Verdict"], "")] * len(row)

    return frame.style.apply(_row_style, axis=1).format({"Confidence": "{:.1%}"}, na_rep="—")


def render_summary_row(summary: dict[str, Any]) -> None:
    """The five verdict counts, shared by both bulk paths."""
    columns = st.columns(5)
    columns[0].metric("Images", summary.get("total", 0))
    columns[1].metric("Classified", summary.get("classified", 0))
    columns[2].metric("Needs review", summary.get("needs_review", 0))
    columns[3].metric("Unable to classify", summary.get("unable_to_classify", 0))
    columns[4].metric("Errors", summary.get("errors", 0))


def render_active_job(base_url: str, job_id: str, timeout: float) -> None:
    """Poll a queued job and show progress, then its results.

    Each poll is one fast script rerun rather than a blocking loop, so the
    page stays responsive and the Cancel button actually works.
    """
    body, error = api_get(base_url, f"/jobs/{job_id}", timeout)
    # Both halves are independently optional, so test the body too - a
    # successful response with no payload would otherwise crash below.
    if error or body is None:
        st.error(error or "The API returned no data for this job.")
        if st.button("Start over"):
            st.session_state.pop(ACTIVE_JOB_KEY, None)
            st.rerun()
        return

    state = body["state"]
    processed, total = body.get("processed", 0), body.get("total", 0)

    if state not in TERMINAL_STATES:
        fraction = processed / total if total else 0.0
        st.progress(
            min(max(fraction, 0.0), 1.0),
            text=f"{state.title()} — {processed} of {total} images ({fraction:.0%})",
        )
        render_summary_row(body.get("summary", {}))
        st.caption(f"Job `{job_id}` · polling every {POLL_SECONDS:.0f}s")

        if st.button("Cancel job", type="secondary"):
            api_delete(base_url, f"/jobs/{job_id}", timeout)
            st.session_state.pop(ACTIVE_JOB_KEY, None)
            st.rerun()

        time.sleep(POLL_SECONDS)
        st.rerun()
        return

    # ── Finished ─────────────────────────────────────────────────────────
    if state == "failed":
        st.error(body.get("error") or "The job failed.")
    elif state == "cancelled":
        st.warning("Job cancelled. Any rows completed before cancelling are still available.")

    if state in {"succeeded", "cancelled"}:
        csv_text, csv_error = fetch_results_csv(base_url, job_id, timeout)
        if csv_error:
            st.error(csv_error)
        elif csv_text is not None:
            summary = body.get("summary", {})
            render_summary_row(summary)

            flagged = (
                summary.get("needs_review", 0)
                + summary.get("unable_to_classify", 0)
                + summary.get("errors", 0)
            )
            if flagged:
                st.warning(f"{flagged} of {summary.get('total', 0)} results need a human look.")
            elif summary.get("total"):
                st.success(f"All {summary['total']} images classified above the review threshold.")

            frame = results_frame_from_csv(csv_text)
            st.dataframe(style_results(frame), use_container_width=True, hide_index=True)
            st.download_button(
                "Download results as CSV",
                data=csv_text.encode("utf-8"),
                file_name=f"leaf_results_{job_id[:8]}.csv",
                mime="text/csv",
            )

    if st.button("Classify another archive"):
        st.session_state.pop(ACTIVE_JOB_KEY, None)
        st.rerun()


def render_job_history(base_url: str, timeout: float) -> None:
    """Earlier jobs, so a finished run can be collected later."""
    body, error = api_get(base_url, "/jobs?limit=10", timeout)
    if error or not body or not body.get("jobs"):
        return

    with st.expander(f"Recent jobs ({len(body['jobs'])})"):
        for job in body["jobs"]:
            row = st.columns([3, 2, 2, 2])
            row[0].markdown(f"`{job['job_id'][:8]}` {job['filename']}")
            row[1].markdown(verdict_badge_for_state(job["state"]), unsafe_allow_html=True)
            row[2].caption(f"{job.get('processed', 0)}/{job.get('total', 0)} images")
            if job["state"] in TERMINAL_STATES and row[3].button(
                "Open", key=f"open_{job['job_id']}"
            ):
                st.session_state[ACTIVE_JOB_KEY] = job["job_id"]
                st.rerun()


def verdict_badge_for_state(state: str) -> str:
    colour = {
        "succeeded": "#1a7f37",
        "running": "#0969da",
        "queued": "#57606a",
        "failed": "#cf222e",
        "cancelled": "#bf8700",
    }.get(state, "#57606a")
    return (
        f"<span style='background:{colour};color:#fff;padding:1px 8px;"
        f"border-radius:10px;font-size:0.78rem'>{state}</span>"
    )


def render_single(body: dict[str, Any], image_bytes: bytes) -> None:
    result = body["result"]
    thresholds = body["thresholds"]

    left, right = st.columns([1, 1.3])
    with left:
        st.image(image_bytes, caption=result["filename"], use_container_width=True)

    with right:
        st.markdown(verdict_badge(result["verdict"]), unsafe_allow_html=True)

        if result.get("label"):
            st.markdown(f"## {result['label']}")
        else:
            st.markdown("## No confident match")

        confidence = result.get("confidence")
        if confidence is not None:
            st.metric("Confidence", f"{confidence:.1%}")
            st.progress(min(max(confidence, 0.0), 1.0))

        if result.get("note"):
            if result["verdict"] == "error":
                st.error(result["note"])
            else:
                st.warning(result["note"])

        st.caption(
            f"Review below {thresholds['review_threshold']:.0%} · "
            f"declines below {thresholds['unknown_threshold']:.0%}"
        )

    probabilities = result.get("probabilities") or {}
    if probabilities:
        with st.expander("Full probability distribution"):
            frame = (
                pd.DataFrame(
                    {"Species": list(probabilities), "Probability": list(probabilities.values())}
                )
                .sort_values("Probability", ascending=False)
                .reset_index(drop=True)
            )
            st.dataframe(
                frame.style.format({"Probability": "{:.2%}"}),
                use_container_width=True,
                hide_index=True,
            )


def render_batch(body: dict[str, Any]) -> None:
    results = body["results"]
    summary = body["summary"]

    render_summary_row(summary)

    flagged = summary["needs_review"] + summary["unable_to_classify"] + summary["errors"]
    if flagged:
        st.warning(
            f"{flagged} of {summary['total']} results need a human look — highlighted below."
        )
    else:
        st.success(f"All {summary['total']} images classified above the review threshold.")

    frame = results_frame(results)
    st.dataframe(style_results(frame), use_container_width=True, hide_index=True)

    st.download_button(
        "Download results as CSV",
        data=frame.to_csv(index=False).encode("utf-8"),
        file_name="leaf_classification_results.csv",
        mime="text/csv",
    )


# ── Page ─────────────────────────────────────────────────────────────────


def main() -> None:
    settings = get_settings()
    st.set_page_config(page_title="Medicinal Leaf Classification", page_icon="🌿", layout="wide")

    st.title("🌿 Indian Medicinal Leaf Classification")
    st.caption(
        "Identify Aloevera, Amla, Mint, Neem and Tulsi from a photograph. "
        "Educational and botanical use only — not medical guidance."
    )

    with st.sidebar:
        st.header("Settings")
        base_url = st.text_input("API base URL", value=settings.serving.api_base_url)

        health = fetch_health(base_url, settings.serving.request_timeout_seconds)
        if health is None:
            st.error("API unreachable. Start it with `leaf-api`.")
        elif not health["model_loaded"]:
            st.warning("API is up, but no model is loaded. Train one with `leaf-train fit`.")
        else:
            st.success(f"Model ready — {health.get('backbone') or 'unknown backbone'}")
            st.caption("Classes: " + ", ".join(health.get("classes", [])))

        st.divider()
        st.subheader("Review policy")
        st.caption("Thresholds are configurable per request (FR-13).")

        defaults = (health or {}).get("thresholds") or {
            "review_threshold": settings.serving.review_threshold,
            "unknown_threshold": settings.serving.unknown_threshold,
        }
        review_threshold = st.slider(
            "Flag for review below",
            0.0,
            1.0,
            float(defaults["review_threshold"]),
            0.01,
            help="Predictions under this confidence are flagged for a human check.",
        )
        unknown_threshold = st.slider(
            "Decline to answer below",
            0.0,
            1.0,
            float(defaults["unknown_threshold"]),
            0.01,
            help="Under this confidence the service reports no species at all.",
        )
        if unknown_threshold > review_threshold:
            st.error("The decline threshold cannot exceed the review threshold.")

        st.divider()
        st.caption(
            f"Limits — image {settings.serving.max_image_bytes / 1048576:.0f} MB, "
            f"archive {settings.serving.max_archive_bytes / 1048576:.0f} MB, "
            f"{settings.serving.max_zip_entries} images per ZIP."
        )

    thresholds = {
        "review_threshold": review_threshold,
        "unknown_threshold": min(unknown_threshold, review_threshold),
    }
    timeout = settings.serving.request_timeout_seconds

    single_tab, bulk_tab = st.tabs(["Single image", "Bulk ZIP upload"])

    with single_tab:
        st.subheader("Classify one image")
        upload = st.file_uploader("Leaf photograph", type=IMAGE_TYPES, key="single")
        if upload is not None:
            payload = upload.getvalue()
            with st.spinner("Classifying…"):
                body, error = post_upload(
                    base_url,
                    "/predict",
                    upload.name,
                    payload,
                    upload.type or "image/jpeg",
                    thresholds,
                    timeout,
                )
            if error:
                st.error(error)
            elif body:
                render_single(body, payload)

    with bulk_tab:
        st.subheader("Classify a ZIP of images")

        active_job = st.session_state.get(ACTIVE_JOB_KEY)
        if active_job:
            render_active_job(base_url, active_job, timeout)
        else:
            choice = st.radio(
                "Processing",
                ["Queued — large archives", "Immediate — small archives"],
                horizontal=True,
                help=(
                    "Queued work runs in the background and is collected when it "
                    "finishes, which is the only way to get through thousands of "
                    "images without the request timing out. Immediate returns in "
                    "one response, and is capped much lower."
                ),
            )
            queued = choice.startswith("Queued")

            if queued:
                st.caption(
                    f"Up to {settings.queue.max_zip_entries:,} images per archive. "
                    f"The browser upload is capped at {settings.queue.ui_max_upload_mb:,} MB "
                    "because Streamlit holds it in memory — for anything larger, "
                    "POST the archive to /jobs directly."
                )
            else:
                st.caption(
                    f"Up to {settings.serving.max_zip_entries} images and "
                    f"{settings.serving.max_archive_bytes / 1048576:.0f} MB, answered in "
                    "one response."
                )

            archive = st.file_uploader("ZIP archive", type=["zip"], key="bulk")
            if archive is not None:
                spinner = "Uploading and queueing…" if queued else "Classifying archive…"
                with st.spinner(spinner):
                    body, error = post_upload(
                        base_url,
                        "/jobs" if queued else "/predict/batch",
                        archive.name,
                        archive.getvalue(),
                        "application/zip",
                        thresholds,
                        timeout,
                    )
                if error:
                    st.error(error)
                elif body and queued:
                    # Hand off to the polling view on the next run.
                    st.session_state[ACTIVE_JOB_KEY] = body["job_id"]
                    st.rerun()
                elif body:
                    render_batch(body)

            render_job_history(base_url, timeout)


if __name__ == "__main__":
    main()
