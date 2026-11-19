"""Rule-based checker for the measurable parts of ASD-STE100 Simplified Technical English.

Rules checked (ASD-STE100 Issue 9, Part 1):
    R4.1/R5.1  sentence length   - procedural <= 20 words, descriptive <= 25 words
    R6.6       paragraph length  - at most 6 sentences per paragraph
    R3.1       voice             - use the active voice (passive is detected heuristically)
    R5.3       one topic         - flag sentences that chain more than one clause with ';'
Plus the Flesch Reading Ease score as an overall readability signal.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

_CITATION = re.compile(r"\[[^\]]*\]")  # citation markers are not words
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(])")
_BE = r"(?:is|are|was|were|be|been|being|am)"
_IRREGULAR = (
    "built|done|made|given|taken|seen|shown|known|driven|written|held|paid|sold|"
    "won|set|put|cut|led|met|brought|bought|found|kept|left|lost|run|spent|grown"
)
_PASSIVE = re.compile(rf"\b{_BE}\s+(?:\w+ly\s+)?(?:\w+ed|{_IRREGULAR})\b", re.I)
_IMPERATIVE = re.compile(r"^(?:\d+\.\s*|[-*]\s*)?(?:Use|Do|Make|Check|Install|Add|Remove|Set|Keep|Apply|Compare)\b")


@dataclass
class STEReport:
    n_sentences: int
    mean_sentence_words: float
    pct_sentences_within_limit: float  # % meeting the 20/25-word limit
    pct_passive: float
    pct_paragraphs_within_limit: float  # % paragraphs with <= 6 sentences
    pct_multi_clause: float
    flesch_reading_ease: float
    compliance: float  # mean of the four rule pass-rates, 0-100

    def as_dict(self) -> dict:
        return asdict(self)


def _syllables(word: str) -> int:
    """Vowel-group heuristic: count vowel runs, drop a silent final 'e', minimum 1."""
    word = word.lower()
    groups = re.findall(r"[aeiouy]+", word)
    n = len(groups) - (1 if word.endswith("e") and not word.endswith("le") and len(groups) > 1 else 0)
    return max(1, n)


def _sentences(paragraph: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(paragraph) if s.strip()]


def _words(sentence: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9'’.%-]*", _CITATION.sub("", sentence))


def check(text: str) -> STEReport:
    """Score a response against the STE rules above."""
    # Strip markdown emphasis/headings so they do not count as words.
    body = re.sub(r"[*_#`>]", "", text)
    paragraphs = [p for p in re.split(r"\n\s*\n|\n(?=\s*(?:[-*]|\d+\.)\s)", body) if p.strip()]
    sents = [s for p in paragraphs for s in _sentences(p.replace("\n", " "))]
    sents = [s for s in sents if _words(s)]
    if not sents:
        return STEReport(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    lengths = [len(_words(s)) for s in sents]
    within = [n <= (20 if _IMPERATIVE.match(s) else 25) for s, n in zip(sents, lengths)]
    passive = [bool(_PASSIVE.search(s)) for s in sents]
    multi = [s.count(";") > 0 for s in sents]
    para_ok = [len(_sentences(p.replace("\n", " "))) <= 6 for p in paragraphs]

    words = [w for s in sents for w in _words(s)]
    syll = sum(_syllables(w) for w in words)
    flesch = 206.835 - 1.015 * (len(words) / len(sents)) - 84.6 * (syll / max(1, len(words)))

    pct = lambda xs: 100.0 * sum(xs) / len(xs)  # noqa: E731
    rules = [pct(within), 100 - pct(passive), pct(para_ok), 100 - pct(multi)]
    return STEReport(
        n_sentences=len(sents),
        mean_sentence_words=sum(lengths) / len(lengths),
        pct_sentences_within_limit=pct(within),
        pct_passive=pct(passive),
        pct_paragraphs_within_limit=pct(para_ok),
        pct_multi_clause=pct(multi),
        flesch_reading_ease=flesch,
        compliance=sum(rules) / len(rules),
    )
