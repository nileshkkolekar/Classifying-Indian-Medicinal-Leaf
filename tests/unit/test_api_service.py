"""Upload limits and the classify / flag / decline policy.

These are the rules that decide whether a user is shown a species name, a
warning, or nothing at all — and the limits standing between the service and
a malicious archive. Both deserve testing without HTTP in the way.
"""

from __future__ import annotations

import zipfile

import pytest

from medicinal_leaf.api.schemas import Verdict
from medicinal_leaf.api.service import (
    UploadRejectedError,
    ZipLimits,
    classify_entries,
    decide_verdict,
    decode_image,
    display_name,
    safe_zip_entries,
    summarize,
    to_result,
)
from tests.conftest import StubPredictor

THRESHOLDS = {"review_threshold": 0.70, "unknown_threshold": 0.30}


# ── Verdict policy ───────────────────────────────────────────────────────


def test_high_confidence_is_classified():
    verdict, note = decide_verdict(0.95, **THRESHOLDS)
    assert verdict is Verdict.CLASSIFIED
    assert note is None


def test_middling_confidence_needs_review():
    verdict, note = decide_verdict(0.55, **THRESHOLDS)
    assert verdict is Verdict.NEEDS_REVIEW
    assert "review threshold" in note


def test_low_confidence_declines_to_answer():
    verdict, note = decide_verdict(0.10, **THRESHOLDS)
    assert verdict is Verdict.UNABLE_TO_CLASSIFY
    assert "floor" in note


def test_thresholds_are_inclusive_at_the_bar():
    """Exactly at the threshold counts as clearing it."""
    assert decide_verdict(0.70, **THRESHOLDS)[0] is Verdict.CLASSIFIED
    assert decide_verdict(0.30, **THRESHOLDS)[0] is Verdict.NEEDS_REVIEW


def test_thresholds_are_not_hardcoded():
    """FR-13: the same score lands differently under a different policy."""
    assert decide_verdict(0.75, review_threshold=0.70, unknown_threshold=0.30)[0] is (
        Verdict.CLASSIFIED
    )
    assert decide_verdict(0.75, review_threshold=0.90, unknown_threshold=0.30)[0] is (
        Verdict.NEEDS_REVIEW
    )


def test_declining_withholds_the_label(stub_predictor):
    """FR-14: no species name is offered when the model is unsure."""
    prediction = stub_predictor.predict(None)
    result = to_result("leaf.jpg", prediction, review_threshold=1.0, unknown_threshold=0.99)

    assert result.verdict is Verdict.UNABLE_TO_CLASSIFY
    assert result.label is None
    # The evidence is still returned for anyone who wants to apply their own policy.
    assert result.confidence == pytest.approx(0.95)
    assert result.probabilities


def test_flagged_result_keeps_its_label(stub_predictor):
    """A low-confidence guess is shown, but marked."""
    result = to_result(
        "leaf.jpg", stub_predictor.predict(None), review_threshold=0.99, unknown_threshold=0.10
    )
    assert result.verdict is Verdict.NEEDS_REVIEW
    assert result.label == "Tulsi"
    assert result.needs_review is True


# ── Name sanitising ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("leaf.jpg", "leaf.jpg"),
        ("folder/leaf.jpg", "folder/leaf.jpg"),
        ("../../etc/passwd.jpg", "etc/passwd.jpg"),
        ("/absolute/leaf.jpg", "absolute/leaf.jpg"),
        ("windows\\style\\leaf.jpg", "windows/style/leaf.jpg"),
        ("..", "unnamed"),
    ],
)
def test_display_name_strips_traversal(raw, expected):
    assert display_name(raw) == expected


# ── Image decoding ───────────────────────────────────────────────────────


def test_decode_valid_jpeg(jpeg_bytes):
    image = decode_image(jpeg_bytes)
    assert image.mode == "RGB"
    assert image.size == (64, 48)


def test_decode_rejects_garbage():
    with pytest.raises(ValueError):
        decode_image(b"this is definitely not an image")


# ── Archive safety ───────────────────────────────────────────────────────


def test_reads_every_image_in_the_archive(make_zip, jpeg_bytes):
    data = make_zip({f"leaf_{i}.jpg": jpeg_bytes for i in range(3)})
    entries = safe_zip_entries(data, ZipLimits())

    assert len(entries) == 3
    assert all(payload == jpeg_bytes for _, payload in entries)


def test_ignores_non_image_members(make_zip, jpeg_bytes):
    data = make_zip(
        {
            "leaf.jpg": jpeg_bytes,
            "notes.txt": b"ignore me",
            "__MACOSX/._leaf.jpg": b"junk",
            ".hidden.jpg": jpeg_bytes,
        }
    )
    entries = safe_zip_entries(data, ZipLimits())
    assert [name for name, _ in entries] == ["leaf.jpg"]


def test_rejects_archive_with_no_images(make_zip):
    data = make_zip({"readme.txt": b"nothing here"})
    with pytest.raises(UploadRejectedError, match="no images"):
        safe_zip_entries(data, ZipLimits())


def test_rejects_something_that_is_not_a_zip():
    with pytest.raises(UploadRejectedError, match="Not a readable ZIP"):
        safe_zip_entries(b"plain bytes", ZipLimits())


def test_rejects_too_many_entries(make_zip, jpeg_bytes):
    data = make_zip({f"leaf_{i}.jpg": jpeg_bytes for i in range(5)})
    with pytest.raises(UploadRejectedError, match="over the 2 per-upload limit") as excinfo:
        safe_zip_entries(data, ZipLimits(max_entries=2))
    assert excinfo.value.status_code == 413


def test_rejects_oversize_member(make_zip, jpeg_bytes):
    data = make_zip({"leaf.jpg": jpeg_bytes})
    with pytest.raises(UploadRejectedError, match="per-image limit") as excinfo:
        safe_zip_entries(data, ZipLimits(max_file_bytes=10))
    assert excinfo.value.status_code == 413


def test_rejects_oversize_total(make_zip, jpeg_bytes):
    data = make_zip({f"leaf_{i}.jpg": jpeg_bytes for i in range(3)})
    with pytest.raises(UploadRejectedError, match="expands to") as excinfo:
        safe_zip_entries(data, ZipLimits(max_uncompressed_bytes=100))
    assert excinfo.value.status_code == 413


def test_rejects_a_zip_bomb(make_zip):
    """Highly compressible padding is the signature of a decompression bomb."""
    data = make_zip({"bomb.jpg": b"\x00" * 500_000})

    with pytest.raises(UploadRejectedError, match="zip bomb") as excinfo:
        safe_zip_entries(data, ZipLimits(max_compression_ratio=10.0))
    assert excinfo.value.status_code == 413


def test_ratio_check_tolerates_normal_images(make_zip, jpeg_bytes):
    """A real JPEG barely compresses, so it must not trip the bomb check."""
    data = make_zip({"leaf.jpg": jpeg_bytes}, compression=zipfile.ZIP_STORED)
    assert len(safe_zip_entries(data, ZipLimits(max_compression_ratio=10.0))) == 1


# ── Batch classification ─────────────────────────────────────────────────


def test_classifies_every_entry(stub_predictor, jpeg_bytes):
    entries = [(f"leaf_{i}.jpg", jpeg_bytes) for i in range(4)]
    results = classify_entries(stub_predictor, entries, **THRESHOLDS)

    assert len(results) == 4
    assert all(r.verdict is Verdict.CLASSIFIED for r in results)
    assert [r.filename for r in results] == [name for name, _ in entries]


def test_decodable_images_go_through_the_model_in_one_batch(stub_predictor, jpeg_bytes):
    entries = [(f"leaf_{i}.jpg", jpeg_bytes) for i in range(4)]
    classify_entries(stub_predictor, entries, **THRESHOLDS)
    assert stub_predictor.batch_calls == 1


def test_one_corrupt_file_does_not_sink_the_batch(stub_predictor, jpeg_bytes):
    entries = [
        ("good_1.jpg", jpeg_bytes),
        ("broken.jpg", b"not an image at all"),
        ("good_2.jpg", jpeg_bytes),
    ]
    results = classify_entries(stub_predictor, entries, **THRESHOLDS)

    assert len(results) == 3
    by_name = {r.filename: r for r in results}
    assert by_name["broken.jpg"].verdict is Verdict.ERROR
    assert by_name["broken.jpg"].needs_review is True
    assert by_name["broken.jpg"].label is None
    assert by_name["good_1.jpg"].verdict is Verdict.CLASSIFIED
    # Order is preserved even though the failure was handled out of band.
    assert [r.filename for r in results] == [name for name, _ in entries]


def test_empty_batch_is_harmless(stub_predictor):
    assert classify_entries(stub_predictor, [], **THRESHOLDS) == []


# ── Summary ──────────────────────────────────────────────────────────────


def test_summary_counts_each_verdict(stub_predictor, jpeg_bytes):
    confident = StubPredictor(confidence=0.99)
    entries = [("a.jpg", jpeg_bytes), ("b.jpg", b"broken")]
    results = classify_entries(confident, entries, **THRESHOLDS)

    summary = summarize(results)
    assert summary.total == 2
    assert summary.classified == 1
    assert summary.errors == 1
    assert summary.flagged == 1
