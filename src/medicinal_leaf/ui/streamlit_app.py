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

from typing import Any

import httpx
import pandas as pd
import streamlit as st

from medicinal_leaf.config.settings import Settings, load_settings

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


def style_results(frame: pd.DataFrame) -> Any:
    """Tint every row that a reviewer needs to look at (FR-12)."""

    def _row_style(row: pd.Series) -> list[str]:
        return [ROW_TINT.get(row["Verdict"], "")] * len(row)

    return frame.style.apply(_row_style, axis=1).format({"Confidence": "{:.1%}"}, na_rep="—")


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

    columns = st.columns(5)
    columns[0].metric("Images", summary["total"])
    columns[1].metric("Classified", summary["classified"])
    columns[2].metric("Needs review", summary["needs_review"])
    columns[3].metric("Unable to classify", summary["unable_to_classify"])
    columns[4].metric("Errors", summary["errors"])

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
        st.caption(
            "Every image is scored individually. Low-confidence results are "
            "highlighted so they can be routed for manual review."
        )
        archive = st.file_uploader("ZIP archive", type=["zip"], key="bulk")
        if archive is not None:
            with st.spinner("Classifying archive…"):
                body, error = post_upload(
                    base_url,
                    "/predict/batch",
                    archive.name,
                    archive.getvalue(),
                    "application/zip",
                    thresholds,
                    timeout,
                )
            if error:
                st.error(error)
            elif body:
                render_batch(body)


if __name__ == "__main__":
    main()
