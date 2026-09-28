"""Standardized prompt suite for reproducible cross-model benchmarks.

The suite is a fixed, versioned set of prompts organized by task category.
Keeping the prompts immutable is what makes benchmark runs comparable over
time and across machines.
"""

from __future__ import annotations

REQUIRED_CATEGORIES: tuple[str, ...] = (
    "summarization",
    "classification",
    "code_generation",
)

CATEGORY_DESCRIPTIONS: dict[str, str] = {
    "summarization": "Condense a source text into key points",
    "classification": "Assign inputs to fixed categories",
    "code_generation": "Produce executable program source code",
}

SUITE: dict[str, list[str]] = {
    "summarization": [
        "Summarize the following meeting notes into three bullet points, "
        "keeping every decision and action owner:\n\n"
        "Meeting: Q3 planning. Attendees: Ana (engineering), Ben (design), "
        "Cam (ops).\n"
        "Decisions: (1) ship the beta by Dec 15; (2) freeze new features "
        "after Nov 1; (3) Cam owns the rollout runbook.\n"
        "Action items: Ana drafts the API contract by Oct 10.",
        "Write a two-sentence summary of this abstract, neutral and "
        "factual:\n\n"
        "This paper presents a lightweight retrieval pipeline for small "
        "language models running on edge devices. The authors show that a "
        "hybrid BM25-dense index matches large-model accuracy at a fraction "
        "of the memory footprint on three open-source benchmarks.",
        "Condense this support email thread into one actionable paragraph "
        "with the next step and its owner:\n\n"
        "Client: The staging deployment failed again.\n"
        "Engineer: I fixed the TLS config on the load balancer.\n"
        "Manager: Please verify on staging and close the incident if green.",
    ],
    "classification": [
        "Classify the sentiment of the following review as POSITIVE, "
        "NEGATIVE or NEUTRAL. Reply with only the label.\n\n"
        "Review: The download worked fine, but the app crashed twice and "
        "support never replied.",
        "Classify this support ticket into one of: BILLING, TECHNICAL, "
        "FEATURE_REQUEST, OTHER. Reply with only the label.\n\n"
        "Ticket: The invoice was charged twice this month and I cannot find "
        "the refund tab.",
        "Assign each company to its industry sector: TECHNOLOGY, FINANCE, "
        "HEALTHCARE, ENERGY, RETAIL. Reply as a comma-separated list.\n\n"
        "Companies: OpenAI, Pfizer, ExxonMobil, Goldman Sachs, Walmart.",
    ],
    "code_generation": [
        "Write a Python function that returns the median of a list of "
        "numbers. The list may be empty: return None in that case. Do not "
        "use the statistics module. Include a short docstring.",
        "Write a bash one-liner that lists the 5 largest files under "
        "/var/log, sorted by size, printing each with its size in "
        "human-readable form.",
        "Write a SQL query that returns, for each month of 2025, the total "
        "revenue and the number of orders, ordered by month.\n\n"
        "Tables:\n"
        "orders(id INTEGER, created_at TIMESTAMP, total NUMERIC)\n"
        "order_items(order_id INTEGER, product_id INTEGER, qty INTEGER, "
        "price NUMERIC)",
    ],
}


def validate_suite(suite: dict[str, list[str]] | None = None) -> None:
    """Raise ValueError if a suite is malformed.

    The default validates the bundled :data:`SUITE`. The check mirrors what
    tests enforce: required categories present, no empty prompts, no
    duplicates.
    """
    data = suite if suite is not None else SUITE

    missing = [c for c in REQUIRED_CATEGORIES if c not in data]
    if missing:
        raise ValueError(f"suite missing required categories: {missing}")

    for category, prompts in data.items():
        if not prompts:
            raise ValueError(f"category {category!r} has no prompts")
        if any(not isinstance(p, str) or not p.strip() for p in prompts):
            raise ValueError(f"category {category!r} contains empty prompts")
        seen: set[str] = set()
        for prompt in prompts:
            if prompt in seen:
                raise ValueError(f"category {category!r} contains duplicate prompts")
            seen.add(prompt)
