"""Spec adr-foundation: DESIGN.md's key decisions become numbered, reviewable ADRs."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

XFAIL = pytest.mark.xfail(
    strict=True, reason="spec:adr-foundation (remove this marker when implemented)"
)
ADR_DIR = Path(__file__).resolve().parents[1] / "docs" / "adr"
STATUSES = {"Proposed", "Accepted", "Superseded", "Deprecated"}


def _adrs() -> list[Path]:
    return sorted(
        p for p in ADR_DIR.glob("[0-9][0-9][0-9][0-9]-*.md") if not p.name.startswith("0000-")
    )


@XFAIL
def test_template_and_index_exist():
    assert (ADR_DIR / "0000-template.md").is_file()
    assert (ADR_DIR / "README.md").is_file()


@XFAIL
def test_one_adr_per_design_decision_numbered_contiguously():
    nums = [int(p.name[:4]) for p in _adrs()]
    assert len(nums) >= 9, "DESIGN.md section 11 lists nine decisions"
    assert nums == list(range(1, len(nums) + 1))


@XFAIL
def test_each_adr_has_status_and_the_three_sections():
    assert _adrs(), "no ADRs found in docs/adr"
    for p in _adrs():
        text = p.read_text()
        assert re.match(rf"# ADR-{p.name[:4]}: \S", text), (
            f"{p.name}: title must be '# ADR-NNNN: ...'"
        )
        m = re.search(r"^Status: (\w+)", text, re.M)
        assert m and m.group(1) in STATUSES, f"{p.name}: bad Status line"
        for section in ("## Context", "## Decision", "## Consequences"):
            assert section in text, f"{p.name}: missing {section}"
        assert len(text.split()) >= 150, f"{p.name}: too thin"


@XFAIL
def test_index_links_every_adr_and_design_doc_points_to_the_folder():
    index = (ADR_DIR / "README.md").read_text()
    for p in _adrs():
        assert p.name in index, f"index does not link {p.name}"
    assert "docs/adr" in (ADR_DIR.parents[1] / "DESIGN.md").read_text()


@XFAIL
def test_no_unmeasured_performance_claims():
    assert _adrs(), "no ADRs found in docs/adr"
    for p in _adrs():
        text = p.read_text()
        assert not re.search(r"outperform", text, re.I), (
            f"{p.name}: do not claim to outperform anything"
        )
        assert not re.search(r"\d+(\.\d+)?\s?x\s+(faster|slower)", text, re.I), (
            f"{p.name}: numbers need a captured run in results/"
        )
