"""Tests for the standardized benchmark suite."""

from __future__ import annotations

import pytest

from benchmarks.suite import (
    CATEGORY_DESCRIPTIONS,
    REQUIRED_CATEGORIES,
    SUITE,
    validate_suite,
)


def test_suite_covers_required_categories() -> None:
    assert set(REQUIRED_CATEGORIES) <= set(SUITE)


def test_all_categories_have_descriptions() -> None:
    assert all(category in CATEGORY_DESCRIPTIONS for category in SUITE)


@pytest.mark.parametrize("category", sorted(SUITE))
def test_category_has_nonempty_prompts(category: str) -> None:
    prompts = SUITE[category]
    assert len(prompts) >= 2
    assert all(isinstance(p, str) and p.strip() for p in prompts)


def test_no_duplicate_prompts_across_suite() -> None:
    all_prompts = [prompt for prompts in SUITE.values() for prompt in prompts]
    assert len(all_prompts) == len(set(all_prompts))


def test_validate_suite_rejects_missing_category() -> None:
    with pytest.raises(ValueError):
        validate_suite({"summarization": ["some prompt"]})


def test_validate_suite_rejects_empty_category() -> None:
    with pytest.raises(ValueError):
        validate_suite({**SUITE, "classification": []})


def test_validate_suite_rejects_duplicate_prompts() -> None:
    with pytest.raises(ValueError):
        validate_suite({"summarization": ["same", "same"]})


def test_validate_suite_accepts_bundled_suite() -> None:
    validate_suite()  # should not raise
