"""
Deterministic checks for explicit format constraints stated in a user prompt.

The constraints are parsed from the prompt text itself (e.g. "in exactly one
sentence of no more than 15 words"), so the same code path works for predefined
scenarios and for custom prompts. Nothing here calls an LLM.
"""

from __future__ import annotations

import re
from typing import Any

_NUM_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "a single": 1, "single": 1,
}
_NUM = r"(\d{1,3}|one|two|three|four|five|six|seven|eight|nine|ten|a single|single)"
_BULLET_RE = re.compile(r"^\s*(?:[-*•–]|\d{1,2}[.)])\s+\S")


def _to_int(token: str) -> int | None:
    token = (token or "").strip().lower()
    if token.isdigit():
        return int(token)
    return _NUM_WORDS.get(token)


def count_words(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'’\-/]*", text or ""))


def count_sentences(text: str) -> int:
    """Count sentences by terminal punctuation; a trailing fragment counts as one."""
    t = re.sub(r"\s+", " ", (text or "").strip())
    if not t:
        return 0
    # Protect common abbreviations / decimals so they do not split sentences
    t = re.sub(r"\b(e\.g|i\.e|etc|vs|mr|mrs|dr|no)\.", lambda m: m.group(1), t, flags=re.I)
    t = re.sub(r"(\d)\.(\d)", r"\1\2", t)
    parts = [p for p in re.split(r"(?<=[.!?])[\"')\]]*\s+", t) if re.search(r"\w", p)]
    return len(parts)


def split_lines(text: str) -> tuple[list[str], list[str]]:
    """Return (bullet_lines, non_bullet_lines), ignoring blank lines."""
    bullets, other = [], []
    for ln in (text or "").splitlines():
        if not ln.strip():
            continue
        (bullets if _BULLET_RE.match(ln) else other).append(ln.strip())
    return bullets, other


def parse_constraints(prompt: str) -> list[dict[str, Any]]:
    """Extract explicit, checkable format constraints from the prompt."""
    p = re.sub(r"\s+", " ", (prompt or "").lower())
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(kind: str, expected: Any, label: str) -> None:
        if kind in seen or expected is None:
            return
        seen.add(kind)
        out.append({"type": kind, "expected": expected, "label": label})

    # --- words ---
    m = re.search(rf"(?:no more than|at most|maximum of|a maximum of|up to|within)\s+{_NUM}\s+words?", p)
    if m:
        n = _to_int(m.group(1))
        add("max_words", n, f"at most {n} words")
    m = re.search(rf"(?:fewer than|less than|under)\s+{_NUM}\s+words?", p)
    if m:
        n = _to_int(m.group(1))
        add("max_words", (n - 1) if n else None, f"fewer than {n} words")
    m = re.search(rf"{_NUM}\s+words?\s+or\s+(?:fewer|less)", p)
    if m:
        n = _to_int(m.group(1))
        add("max_words", n, f"at most {n} words")
    m = re.search(rf"exactly\s+{_NUM}\s+words?", p)
    if m:
        n = _to_int(m.group(1))
        add("exact_words", n, f"exactly {n} words")
    if re.search(r"(?:only|just|exactly|with|in)\s+(?:one|a single|1)\s+word\b", p):
        add("max_words", 1, "one word only")

    # --- sentences ---
    m = re.search(rf"(?:no more than|at most|maximum of|up to)\s+{_NUM}\s+(?:short |brief |simple )?sentences?", p)
    if m:
        n = _to_int(m.group(1))
        add("max_sentences", n, f"at most {n} sentence(s)")
    m = re.search(rf"(?:exactly|only|just)\s+{_NUM}\s+(?:short |brief |simple |complete )?sentences?", p)
    if m:
        n = _to_int(m.group(1))
        add("exact_sentences", n, f"exactly {n} sentence(s)")
    elif "max_sentences" not in seen:
        m = re.search(rf"\bin\s+{_NUM}\s+(?:short |brief |simple |complete )?sentences?", p)
        if m:
            n = _to_int(m.group(1))
            add("exact_sentences", n, f"{n} sentence(s)")

    # --- bullets ---
    m = re.search(rf"(?:exactly|only|just)\s+{_NUM}\s+bullet", p) or re.search(
        rf"\b{_NUM}\s+bullet\s+points?", p
    )
    if m:
        n = _to_int(m.group(1))
        add("exact_bullets", n, f"exactly {n} bullet points")
        if re.search(r"no other text|nothing else|only .*bullet", p):
            add("only_bullets", True, "no text outside the bullet points")

    return out


def check_constraints(prompt: str, response: str) -> dict[str, Any]:
    """Evaluate every parsed constraint against the response."""
    constraints = parse_constraints(prompt)
    resp = response or ""
    words = count_words(resp)
    sentences = count_sentences(resp)
    bullets, other = split_lines(resp)
    results: list[dict[str, Any]] = []
    for c in constraints:
        kind, exp = c["type"], c["expected"]
        if kind == "max_words":
            observed, passed = words, words <= exp
        elif kind == "exact_words":
            observed, passed = words, words == exp
        elif kind == "max_sentences":
            observed, passed = sentences, sentences <= exp
        elif kind == "exact_sentences":
            observed, passed = sentences, sentences == exp
        elif kind == "exact_bullets":
            observed, passed = len(bullets), len(bullets) == exp
        elif kind == "only_bullets":
            observed, passed = len(other), len(other) == 0
        else:  # pragma: no cover - defensive
            observed, passed = None, True
        results.append({**c, "observed": observed, "passed": bool(passed)})
    violated = [r for r in results if not r["passed"]]
    return {
        "has_constraints": bool(results),
        "results": results,
        "all_passed": not violated,
        "violated": violated,
        "stats": {
            "words": words,
            "sentences": sentences,
            "bullet_lines": len(bullets),
            "non_bullet_lines": len(other),
            "characters": len(resp),
        },
    }


def describe_violations(check: dict[str, Any]) -> list[str]:
    """Human-readable, factual statements about failed constraints."""
    out = []
    for r in check.get("violated") or []:
        kind = r["type"]
        if kind in {"max_words", "exact_words"}:
            out.append(f"The reply has {r['observed']} words; the prompt requires {r['label']}.")
        elif kind in {"max_sentences", "exact_sentences"}:
            out.append(f"The reply has {r['observed']} sentence(s); the prompt requires {r['label']}.")
        elif kind == "exact_bullets":
            out.append(f"The reply has {r['observed']} bullet point(s); the prompt requires {r['label']}.")
        elif kind == "only_bullets":
            out.append(f"The reply has {r['observed']} line(s) of text outside the bullet points; the prompt allows none.")
    return out
