"""Faithfulness checks over an (original, compressed) pair.

Heuristic and dependency-free -- a smoke alarm, not a proof. Ordered by
severity: negation loss (flipped meaning), negation scope loss (meaning
reattached to the wrong word), number loss (a rewritten claim), entity loss
(a reattributed one).
"""

from __future__ import annotations

import re

__all__ = [
    "NEGATION_FORCE_TOKENS",
    "NEGATION_WORDS",
    "find_entities",
    "find_negation_scopes",
    "find_negations",
    "find_numbers",
    "verify",
]

# fmt: off
#: Words whose disappearance can invert a statement's meaning.
NEGATION_WORDS: frozenset[str] = frozenset({
    "not", "no", "never", "none", "neither", "nor", "except", "unless",
    "without", "cannot", "nothing", "nobody", "nowhere",
    "lack", "lacks", "lacking", "absent",
})
# "rather"/"instead" excluded: they mark substitution, not negation.

#: Handed to backends that accept a "never drop these" token list. Derived from
#: NEGATION_WORDS so the checked and the protected vocabulary cannot drift.
NEGATION_FORCE_TOKENS: tuple[str, ...] = ("n't", *sorted(NEGATION_WORDS))
# fmt: on

# fmt: off
#: Skipped when looking for what a negation applies to. A function word carries
#: no claim of its own, so losing one is not scope loss: "not when you import"
#: scopes over "import", and "not at module scope" over "module". Deliberately
#: no verbs beyond the auxiliaries -- "cannot be waived" scopes over "waived".
_SCOPE_SKIP: frozenset[str] = frozenset({
    # determiners and quantifiers
    "a", "an", "the", "any", "this", "that", "these", "those", "all", "both",
    "each", "every", "some", "much", "many",
    # possessives and pronouns
    "its", "his", "her", "their", "our", "your", "my", "it", "they", "we",
    "you", "he", "she", "i", "them", "us", "me", "him", "there",
    # prepositions and subordinators
    "of", "in", "on", "at", "to", "for", "by", "with", "from", "as", "into",
    "onto", "over", "under", "about", "between", "through", "during", "when",
    "while", "where", "if", "because", "since", "although", "though", "after",
    "before", "than", "then", "so", "and", "or", "but", "that's",
    # auxiliaries and copulas
    "is", "are", "was", "were", "be", "been", "being", "am", "do", "does",
    "did", "has", "have", "had", "can", "will", "would", "should", "could",
    "may", "might", "must", "shall",
    # degree and comparison: "never more than 3 seconds" scopes over the
    # measurement, and "not only slow" over "slow"
    "more", "less", "fewer", "greater", "longer", "shorter", "later",
    "earlier", "sooner", "further", "only", "just", "even", "still", "yet",
    "always", "ever", "very", "too", "quite", "also", "other", "otherwise",
})
# fmt: on

#: What ends the region a negation can scope over. Semicolons and colons count:
#: they separate independent clauses as firmly as a full stop does.
_SENTENCE_END_RE = re.compile(r"[.!?;:\n]")

_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_KEPT_RE = re.compile(r"[a-z0-9]+(?:['&.-][a-z0-9]+)*")
_SPLIT_RE = re.compile(r"['&.-]")

# Ints, decimals, thousands-separated figures, percentages, and dotted versions.
_NUMBER_RE = re.compile(
    r"""
    (?<![\w.])
    [vV]?\d+
    (?:[.,]\d+)*
    %?
    (?![\w])
    """,
    re.VERBOSE,
)

# A capitalised word; dots only between alphanumerics ("U.S.A", "Node.js").
_CAPWORD = r"[A-Z][A-Za-z0-9'&-]*(?:\.[A-Za-z0-9][A-Za-z0-9'&-]*)*"

# Lowercase words that sit inside a name ("Bank of America"). "the"/"and"
# excluded: they glue sentences far more often than names.
_NAME_CONNECTORS = r"of|de|del|da|di|van|von|der|la|le"
# Horizontal whitespace only: a name never straddles a line break.
_ENTITY_RE = re.compile(
    rf"\b{_CAPWORD}(?:[ \t]+(?:{_NAME_CONNECTORS})[ \t]+{_CAPWORD}|[ \t]+{_CAPWORD})*"
)
_ACRONYM_RE = re.compile(r"\b[A-Z]{2,}(?:\d+)?\b")

# fmt: off
# Capitalised words that usually just start a sentence rather than name anything.
_SENTENCE_STARTERS = frozenset({
    "a", "an", "the", "this", "that", "these", "those", "it", "its", "there",
    "they", "we", "you", "he", "she", "i", "if", "when", "while", "but", "and",
    "or", "so", "for", "in", "on", "at", "to", "as", "by", "with", "from",
    "however", "because", "since", "although", "though", "after", "before",
    "during", "each", "every", "some", "many", "most", "all", "both",
    "no", "not", "never", "unless", "without", "here", "how", "what", "why",
    "who", "which", "then", "than", "now", "also", "note", "use", "using",
    "see", "do", "does", "did", "is", "are", "was", "were", "be", "can",
    "cannot", "will", "would", "should", "could", "may", "might", "must",
    "let", "make", "get", "keep", "your", "our", "their", "my",
})
# fmt: on

_MAX_REPORTED = 6


def _norm_number(raw: str) -> str:
    """Normalise a number so ``1,000`` and ``1000`` compare equal.

    The ``v`` of a version literal is kept: "roll back to v2" and "2 retries"
    are different claims, and stripping it made one vouch for the other. Case
    still folds, so ``V2`` and ``v2`` are one number.
    """
    return raw.lower().replace(",", "")


def find_negations(text: str) -> dict[str, int]:
    """Count negation cues, treating ``-n't`` contractions as their own cue."""
    counts: dict[str, int] = {}
    lowered = text.lower()
    for match in _WORD_RE.finditer(lowered):
        word = match.group(0)
        if word.endswith("n't"):
            counts["n't"] = counts.get("n't", 0) + 1
            continue
        if word in NEGATION_WORDS:
            counts[word] = counts.get(word, 0) + 1
    # "cant"/"wont" style apostrophe-less contractions are ambiguous; skipped.
    return counts


def find_negation_scopes(text: str) -> list[tuple[str, str]]:
    """Pair each negation cue with the word it scopes over.

    Pinning a negation keeps the cue but not what it applies to, and a cue that
    outlives its scope silently re-attaches to whatever word the compressor left
    next to it: "is not automatic: accounts" becomes "not accounts". Counting
    cues cannot see that, because the cue is still there.

    Determiners between the two are skipped -- "not a network partition" scopes
    over "network", and losing the "a" damages nothing. A cue immediately
    followed by another cue has no scope of its own and is left out.

    The search stops at the end of the sentence. "Is it ready? No. The migration
    proceeds" must not hand "No" the next sentence's subject: a one-word answer
    scopes over nothing, and pairing it across the boundary invents a warning.
    """
    lowered = text.lower()
    matches = list(_WORD_RE.finditer(lowered))
    pairs: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        word = match.group(0)
        if not (word.endswith("n't") or word in NEGATION_WORDS):
            continue
        cursor = match.end()
        for candidate_match in matches[index + 1 :]:
            if _SENTENCE_END_RE.search(lowered[cursor : candidate_match.start()]):
                break  # a new sentence; whatever it says is not this cue's scope
            cursor = candidate_match.end()
            candidate = candidate_match.group(0)
            if candidate in _SCOPE_SKIP:
                continue
            if candidate.endswith("n't") or candidate in NEGATION_WORDS:
                break  # the next cue owns that word as its own scope
            pairs.append((word, candidate))
            break
    return pairs


def find_numbers(text: str) -> dict[str, int]:
    """Count numeric literals, normalised so formatting differences do not matter."""
    counts: dict[str, int] = {}
    for match in _NUMBER_RE.finditer(text):
        key = _norm_number(match.group(0))
        counts[key] = counts.get(key, 0) + 1
    return counts


def find_entities(text: str) -> list[str]:
    """Best-effort proper-noun extraction: capitalised runs and acronyms."""
    entities: list[str] = []
    seen: set[str] = set()

    for match in _ENTITY_RE.finditer(text):
        phrase = match.group(0).strip(" .,-")
        if not phrase:
            continue
        words = phrase.split()
        # Drop leading sentence scaffolding: "The Acme Corp" -> "Acme Corp".
        stripped = 0
        while len(words) > 1 and words[0].lower() in _SENTENCE_STARTERS:
            words = words[1:]
            stripped += 1
        if len(words) == 1:
            # A lone capital counts unless it is merely opening a sentence.
            opens_sentence = stripped == 0 and _starts_sentence(text, match.start())
            if opens_sentence or words[0].lower() in _SENTENCE_STARTERS:
                continue
        phrase = " ".join(words)
        key = phrase.lower()
        if key not in seen:
            seen.add(key)
            entities.append(phrase)

    for match in _ACRONYM_RE.finditer(text):
        acronym = match.group(0)
        key = acronym.lower()
        if key not in seen and len(acronym) >= 2:
            seen.add(key)
            entities.append(acronym)

    return entities


def _starts_sentence(text: str, index: int) -> bool:
    """Whether the token at ``index`` is the first word of a sentence."""
    i = index - 1
    while i >= 0 and text[i] in " \t":
        i -= 1
    if i < 0:
        return True
    return text[i] in ".!?\n:;#•-*>"  # "#": a heading opens a line


def _fmt(items: list[str]) -> str:
    """Join pre-formatted items, capping the list so a warning stays readable."""
    shown = ", ".join(items[:_MAX_REPORTED])
    if len(items) <= _MAX_REPORTED:
        return shown
    return f"{shown} (+{len(items) - _MAX_REPORTED} more)"


def verify(original: str, compressed: str) -> list[str]:
    """Return human-readable faithfulness warnings, most severe first.

    An empty list means every check passed. The checks never raise; a document
    that trips nothing is not thereby proven faithful, only un-suspicious.
    """
    warnings: list[str] = []

    # 1. Negation --------------------------------------------------------
    orig_neg = find_negations(original)
    comp_neg = find_negations(compressed)
    lost_neg = []
    for word, count in orig_neg.items():
        kept = comp_neg.get(word, 0)
        if kept < count:
            lost_neg.append(f"{word!r} ({count}× → {kept}×)")
    if lost_neg:
        warnings.append(f"negation lost: {_fmt(lost_neg)} — meaning may be inverted")

    # 2. Negation scope --------------------------------------------------
    # A cue that outlived what it applies to is not caught above: it is still
    # present, so the counts balance. Only cues that survived are considered --
    # a dropped one is already reported as inversion, and reporting it twice
    # would bury the more severe warning.
    comp_words = {m.group(0) for m in _WORD_RE.finditer(compressed.lower())}
    stranded: list[str] = []
    for cue, scope in find_negation_scopes(original):
        if cue in comp_words and scope not in comp_words:
            pair = f"{cue!r} (scope {scope!r})"
            if pair not in stranded:
                stranded.append(pair)
    if stranded:
        warnings.append(
            f"negation kept without its scope: {_fmt(stranded)} — it now applies to "
            "whatever followed it"
        )

    # 3. Numbers ---------------------------------------------------------
    orig_num = find_numbers(original)
    comp_num = find_numbers(compressed)
    missing_numbers = [n for n, count in orig_num.items() if comp_num.get(n, 0) < count]
    if missing_numbers:
        warnings.append("numbers missing from output: " + _fmt([repr(n) for n in missing_numbers]))

    # 4. Entities --------------------------------------------------------
    # Tokenise the output once, rather than scanning the whole document with a
    # fresh regex for every word of every entity.
    kept = _word_set(compressed.lower())
    missing_entities = []
    survived: dict[frozenset[str], list[tuple[list[str], str]]] = {}
    for entity in find_entities(original):
        words = [w for w in (_bare(w) for w in entity.split()) if w]
        if not words:
            continue
        remaining = frozenset(w for w in words if w in kept)
        if not remaining:
            missing_entities.append(entity)
            continue
        survived.setdefault(remaining, []).append((words, entity))
    if missing_entities:
        warnings.append(
            "entities missing from output: " + _fmt([repr(e) for e in missing_entities])
        )

    # 5. Entity ambiguity -------------------------------------------------
    # Clipping "Acme Corporation" to "Acme" is harmless; clipping both "Bank of
    # America" and "Bank of England" to "Bank" is not. The words that survived
    # are the same, so the output can no longer say which one it means. Only a
    # collision is reported -- a lone entity has nothing to be confused with.
    collisions = []
    for remaining, entries in survived.items():
        distinct: list[tuple[list[str], str]] = []
        for words, entity in entries:
            if not any(_same_name(words, other) for other, _ in distinct):
                distinct.append((words, entity))
        if len(distinct) > 1:
            collisions.append(
                f"{' / '.join(repr(e) for _, e in distinct)} — both reduced to "
                f"{' '.join(sorted(remaining))!r}"
            )
    if collisions:
        warnings.append("entities now indistinguishable: " + _fmt(collisions))

    return warnings


def _same_name(first: list[str], second: list[str]) -> bool:
    """Whether two entity word lists are spellings of one name.

    "Acme Corp" abbreviates "Acme Corporation", "CLI's" is "CLI", and a later
    mention often drops the suffix entirely. None of those is a second
    organisation that compression made ambiguous, so none should be reported as
    a collision. Words are compared position by position, either one allowed to
    be a truncation of the other, and the shorter list may stop early.

    "Bank of America" and "Bank of England" diverge at a word where neither is a
    truncation of the other, which is what makes them two names.
    """
    short, long = (first, second) if len(first) <= len(second) else (second, first)
    # Not strict: the shorter list stopping early is the "Acme" / "Acme Corp" case.
    return all(a.startswith(b) or b.startswith(a) for a, b in zip(short, long, strict=False))


def _bare(word: str) -> str:
    """Lowercase a word and drop its possessive, so "CLI's" matches "CLI"."""
    return re.sub(r"'s$", "", word.strip(".,'&-").lower())


def _word_set(text_lower: str) -> set[str]:
    """Every word of ``text_lower``, plus the pieces of hyphenated/dotted ones.

    Both forms are needed so "Acme" is found inside "acme-2" and "Node.js" is
    only found when the whole name survived.
    """
    kept: set[str] = set()
    for token in _KEPT_RE.findall(text_lower):
        kept.add(_bare(token))
        kept.update(part for part in _SPLIT_RE.split(token) if part)
    return kept
