"""Split long documents into chunks a backend can handle, and put them back.

Cuts at the coarsest boundary that fits (paragraph, line, sentence, then words),
records every separator so rejoining restores the layout, and routes fenced code
and tables around the compressor entirely.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .base import CompressionResult, CompressorBackend, count_tokens

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = [
    "CODE_EXTENSIONS",
    "COMPOUND_NUMBER_RE",
    "DEFAULT_CHUNK_TOKENS",
    "FENCE_RE",
    "IDENTIFIER_RE",
    "INLINE_CODE_RE",
    "LINE_PREFIX_RE",
    "URL_RE",
    "Chunk",
    "chunk_document",
    "code_regions",
    "compress_document",
    "looks_like_code",
    "protect_spans",
    "rejoin",
    "restore_spans",
    "split_sentences",
    "word_cost",
]

#: Comfortably inside the 512 word-piece window of the default LLMLingua-2 model.
DEFAULT_CHUNK_TOKENS = 450

#: A fenced block: ``` or ~~~ opener through its matching closer (or end of text).
FENCE_RE = re.compile(
    r"^[ \t]*(```|~~~)[^\n]*\n.*?(?:^[ \t]*\1[ \t]*$|\Z)",
    re.DOTALL | re.MULTILINE,
)
#: An inline code span; CommonMark lets it wrap, bounded to one line break so
#: a stray backtick cannot swallow a paragraph.
INLINE_CODE_RE = re.compile(r"`[^`\n]*(?:\n[^`\n]*)?`")
URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>()\[\]]+|\S+@\S+\.\S+")

#: Block types re-emitted byte-for-byte. Compressing a table's cells destroys
#: its alignment and the column-to-value mapping.
VERBATIM_CODE_TOKENS = frozenset({"fence", "code_block", "html_block"})

#: Extensions whose contents are source, not prose. Compressing these produces
#: syntactically invalid output, so grug passes them through untouched.
CODE_EXTENSIONS = frozenset(
    {
        ".py",
        ".pyi",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".rs",
        ".go",
        ".java",
        ".kt",
        ".c",
        ".h",
        ".cc",
        ".cpp",
        ".hpp",
        ".cs",
        ".rb",
        ".php",
        ".swift",
        ".scala",
        ".sh",
        ".bash",
        ".zsh",
        ".fish",
        ".ps1",
        ".sql",
        ".r",
        ".jl",
        ".lua",
        ".pl",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".cfg",
        ".xml",
        ".proto",
        ".tf",
        ".dockerfile",
        ".makefile",
        ".gradle",
        ".cmake",
        ".vim",
        ".el",
    }
)

#: Lines that only code writes. One of these anywhere in a run is what makes it
#: code rather than indented prose.
_CODE_STRONG_RE = re.compile(
    r"""
      ^\#!                                          # shebang
    | ^\s*(?:def|class|import|from|return|elif|while|try|except|finally|
            lambda|async|await|yield|raise|assert|const|let|var|func|fn|pub|
            impl|struct|enum|interface|package|require|module|namespace|
            template|typedef|extern|static|public|private|defn|defmacro|
            fun|val|sub|end|done|fi|esac|then|do)\b
    | ^\s*(?:SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP|FROM|WHERE|JOIN|
            GROUP|ORDER|HAVING|UNION|WITH)\b                # SQL
    | ^\s*(?:FROM|RUN|COPY|ADD|WORKDIR|ENTRYPOINT|CMD|ENV|EXPOSE|ARG|LABEL|
            VOLUME|USER|SHELL|HEALTHCHECK)\s+\S             # Dockerfile
    | ^\s*[@]\w                                     # decorator
    | ^\s*[{\[]                                     # object or array opener
    | ^\s*\([a-z][\w./-]*[\s)]                       # s-expression, not a prose aside
    | ^\s*[)\]}]                                    # closing bracket line
    | ^\s*</?[A-Za-z][\w:-]*[\s/>]                   # markup tag
    | ^[\w.$-]+:[ \t]*$                             # make target, label
    | [;{}]\s*$                                     # statement terminator
    | ;\s*(?:do|then)\s*$                           # shell block opener
    | ^\s*[\w.\[\]"'$-]+(?:\s+[\w.\[\]"'$-]+)*\s*[-+*/|&^]?=[^=]   # assignment
    | =>|->|::|<-|\|\||&&|!==|===                    # operators prose does not use
    | \w\(.*\)\s*[:{]?\s*$                         # call or signature ending a line
    """,
    re.VERBOSE,
)

#: Weak on its own: two spaces or a tab is how most languages indent, but it is
#: also how a markdown list continues, so this never convicts alone.
_CODE_WEAK_RE = re.compile(r"^(?:[ ]{2,}|\t+)\S")


def _code_line(line: str) -> bool:
    return bool(_CODE_STRONG_RE.search(line) or _CODE_WEAK_RE.search(line))


#: How many consecutive code-looking lines make a block worth protecting.
CODE_RUN_LINES = 2
VERBATIM_MARKDOWN_TOKENS = frozenset({"table_open"})

#: Line-leading markdown structure. Protected so the marker returns verbatim
#: while the text after it still compresses.
LINE_PREFIX_RE = re.compile(
    r"^[ \t]*(?:"
    r"#{1,6}[ \t]+"
    r"|[-*+][ \t]+"
    r"|\d+[.)][ \t]+"
    r"|>[ \t]*"
    r"|(?:-{3,}|\*{3,}|_{3,})[ \t]*$"
    r")",
    re.MULTILINE,
)

#: Numbers whose meaning lives in an internal separator: 9.6, 1,250, 3-5.
#: Backends keep the digits but drop the punctuation, splitting one into two.
COMPOUND_NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:[.,:/-]\d+)+%?(?![\w])")

#: Names whose meaning lives in an internal separator, the way a number's does:
#: us-east-1, v2.1.0-rc3, text/plain, read:write, utf-8. A word-piece tokenizer
#: splits these on the punctuation and a compressor then drops it, so "node-07"
#: comes back as "node 07" -- a different host, silently.
#:
#: The token must mix letters and digits. Punctuation alone does not make an
#: identifier -- "and/or", "input/output", "e.g." and "U.S.A" are English, and
#: pinning them would cost ratio on every document for no faithfulness gain.
#: Neither does a bare hyphen: "api-gateway" and "sign-off" are the same shape,
#: so no pattern separates them, and the pair that matters ("node-07") has a
#: digit. A bare number is :data:`COMPOUND_NUMBER_RE`'s job, not this one's.
#: Letters and digits together: "node-07", "v2.1.0-rc3", "utf-8", "log4j-2.17.1".
_IDENT_MIXED = (
    r"(?=[A-Za-z0-9./:-]*[A-Za-z])"
    r"(?=[A-Za-z0-9./:-]*\d)"
    r"[A-Za-z0-9]+(?:[./:-][A-Za-z0-9]+)+%?"
)
#: Dotted names with no digit at all: "TRAINING.md", "notes.grug.md", "Node.js".
#: Every segment must be at least two characters, which is what separates a
#: filename from the dotted abbreviations that pepper English -- "e.g.", "i.e.",
#: "U.S.A" are all single-letter segments. A letter is required so that "12.50"
#: stays a number.
_IDENT_DOTTED = r"(?=[A-Za-z0-9.]*[A-Za-z])[A-Za-z0-9]{2,}(?:\.[A-Za-z0-9]{2,})+"

IDENTIFIER_RE = re.compile(rf"(?<![\w./:-])(?:{_IDENT_MIXED}|{_IDENT_DOTTED})(?![\w-])")

# Placeholder for a protected span. A plain ASCII word, because a word-piece
# tokenizer mangles private-use codepoints and splits on punctuation. One-letter
# tag so the digits that follow parse unambiguously.
_PH_PREFIX = "GRUGSPAN"
_PH_SUFFIX = "X"
_PH_RE_TEMPLATE = _PH_PREFIX + "%s" + r"(\d+)" + _PH_SUFFIX
# Case-insensitive: a backend may change case, and a lost placeholder silently
# deletes the span it stands for.
PLACEHOLDER_RE = re.compile(_PH_RE_TEMPLATE % r"[a-z]", re.IGNORECASE)

_BLOCK_SPLIT_RE = re.compile(r"(\n[ \t]*\n\s*)")
_BLANK_LINE_RE = re.compile(r"\n[ \t]*\n")
_NEWLINE_RE = re.compile(r"\n")
_WORD_SPLIT_RE = re.compile(r"\s+")

# Words that end in a period without ending a sentence.
_ABBREVIATIONS = frozenset(
    {
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "sr",
        "jr",
        "st",
        "mt",
        "inc",
        "ltd",
        "co",
        "corp",
        "dept",
        "est",
        "vs",
        "etc",
        "e.g",
        "i.e",
        "cf",
        "al",
        "fig",
        "vol",
        "pp",
        "ch",
        "approx",
        "min",
        "max",
        "avg",
        "sec",
        "ref",
        "eq",
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sep",
        "sept",
        "oct",
        "nov",
        "dec",
    }
)

_SENT_BOUNDARY_RE = re.compile(r"([.!?]+[\"')\]]*)(\s+)")


def protect_spans(text: str, *patterns: re.Pattern[str], tag: str = "c") -> tuple[str, list[str]]:
    """Replace every match of ``patterns`` with an opaque placeholder.

    Args:
        text: Text to rewrite.
        *patterns: Compiled patterns whose matches are stashed, applied in order.
        tag: Single-letter namespace; nested passes must use different tags
            so their index spaces cannot collide.

    Returns:
        The rewritten text and the stash of original spans, to be handed back
        to :func:`restore_spans` with the same ``tag``.
    """
    stash: list[str] = []

    def _stash(match: re.Match[str]) -> str:
        span = match.group(0)
        # A later pattern can span an earlier placeholder -- "`-b`/`backend=`"
        # becomes "GRUGSPANc4X/GRUGSPANc5X", which reads as one identifier.
        # Stashing it would nest the two, and :func:`restore_spans` unwraps only
        # one level, so both spans would be lost. Leave it to the pattern that
        # already claimed it.
        if contains_placeholder(span):
            return span
        stash.append(span)
        return f"{_PH_PREFIX}{tag}{len(stash) - 1}{_PH_SUFFIX}"

    for pattern in patterns:
        text = pattern.sub(_stash, text)
    return text, stash


@functools.lru_cache(maxsize=8)
def _placeholder_re(tag: str) -> re.Pattern[str]:
    return re.compile(_PH_RE_TEMPLATE % re.escape(tag), re.IGNORECASE)


def restore_spans(text: str, stash: list[str], *, tag: str = "c") -> str:
    """Restore spans stashed by :func:`protect_spans`; other tags are left alone."""
    if not stash:
        return text

    def _pop(match: re.Match[str]) -> str:
        index = int(match.group(1))
        return stash[index] if index < len(stash) else match.group(0)

    return _placeholder_re(tag).sub(_pop, text)


def contains_placeholder(token: str) -> bool:
    """Whether ``token`` holds a protected-span placeholder."""
    return _PH_PREFIX in token.upper()


def split_sentences(text: str) -> list[str]:
    """Split a line of prose into sentences, keeping trailing punctuation.

    Abbreviations ("e.g.", "Dr.", "vs.") and decimals do not end sentences.
    """
    if not text.strip():
        return [text] if text else []

    sentences: list[str] = []
    start = 0
    for match in _SENT_BOUNDARY_RE.finditer(text):
        end = match.end(1)
        if _is_abbreviation(text, match.start(1)):
            continue
        sentences.append(text[start:end])
        start = match.end(2)
    tail = text[start:]
    if tail:
        sentences.append(tail)
    return [s for s in sentences if s] or [text]


def _is_abbreviation(text: str, punct_index: int) -> bool:
    """Whether the punctuation at ``punct_index`` closes an abbreviation."""
    if text[punct_index] != ".":
        return False
    word_start = punct_index
    while word_start > 0 and (text[word_start - 1].isalnum() or text[word_start - 1] == "."):
        word_start -= 1
    word = text[word_start:punct_index].lower().rstrip(".")
    if not word:
        return False
    if word in _ABBREVIATIONS:
        return True
    # Single letters are initials ("J. R. R. Tolkien"); digits are decimals.
    return len(word) == 1 and word.isalpha()


@functools.lru_cache(maxsize=8192)
def word_cost(word: str) -> int:
    return max(1, count_tokens(" " + word))


@dataclass
class Chunk:
    """One unit of work handed to a backend, plus how to stitch it back."""

    text: str
    #: Exact separator that followed this chunk in the source document.
    sep: str = ""
    #: Whitespace that preceded this chunk. Held outside the compressed text
    #: because backends are free to strip their output.
    prefix: str = ""
    #: ``False`` for fenced code, which is re-emitted verbatim.
    compressible: bool = True
    #: Spans (inline code, URLs) swapped out before compression.
    stash: list[str] = field(default_factory=list)


def chunk_document(
    text: str,
    max_tokens: int = DEFAULT_CHUNK_TOKENS,
    *,
    preserve_code: bool = True,
    preserve_inline_code: bool = True,
    preserve_markdown: bool = True,
    preserve_numbers: bool = True,
    preserve_identifiers: bool = True,
) -> list[Chunk]:
    """Cut ``text`` into chunks of at most ``max_tokens``, structure recorded.

    Args:
        text: The document.
        max_tokens: Soft ceiling per chunk; only a single over-long sentence
            can exceed it, and then it is hard-split.
        preserve_code: Route fenced code blocks around the compressor.
        preserve_inline_code: Swap inline code spans and URLs for placeholders
            so a backend cannot rewrite their contents.
        preserve_markdown: Route tables around the compressor, and protect
            line-leading structure (heading hashes, bullets, ordered-list
            numbers, blockquote markers, horizontal rules).
        preserve_numbers: Protect numbers with internal separators, which a
            backend keeps the digits of but not the punctuation between them.
        preserve_identifiers: Protect names whose internal separator is
            load-bearing (``us-east-1``, ``text/plain``, ``v2.1.0-rc3``).
    """
    if max_tokens < 1:
        raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
    if not text:
        return []

    token_types: frozenset[str] = frozenset()
    if preserve_code:
        token_types |= VERBATIM_CODE_TOKENS
    if preserve_markdown:
        token_types |= VERBATIM_MARKDOWN_TOKENS

    spans = [INLINE_CODE_RE, URL_RE] if preserve_inline_code else []
    if preserve_markdown:
        spans.insert(0, LINE_PREFIX_RE)
    # Identifiers first: they are the longer, more specific match. Run the other
    # way round, COMPOUND_NUMBER_RE claims the "2.17.1" out of "log4j-2.17.1"
    # and leaves the name behind it unprotected.
    if preserve_identifiers:
        spans.append(IDENTIFIER_RE)
    if preserve_numbers:
        spans.append(COMPOUND_NUMBER_RE)

    ranges = _verbatim_ranges(text, token_types)
    if preserve_code:
        ranges = _merge_ranges(ranges + code_regions(text))

    chunks: list[Chunk] = []
    for segment, is_verbatim in _segments_from_ranges(text, ranges):
        if is_verbatim:
            chunks.append(Chunk(text=segment, sep="", compressible=False))
            continue
        chunks.extend(_chunk_prose(segment, max_tokens, spans))
    return chunks


@functools.lru_cache(maxsize=1)
def _parser() -> Any:
    """CommonMark + GFM tables. Not ``gfm-like``: that needs the linkify extra."""
    from markdown_it import MarkdownIt

    # Inline parsing is ~35% of the parse and we only read block token maps.
    return MarkdownIt("commonmark").enable("table").disable("inline")


def _line_starts(text: str) -> list[int]:
    return [0, *(m.end() for m in _NEWLINE_RE.finditer(text))]


def looks_like_code(text: str, filename: str | None = None) -> bool:
    """Whether ``text`` is source code rather than prose.

    The extension decides when there is one -- it is the strongest signal we
    get. Otherwise this counts code-shaped lines, and deliberately leans
    towards saying yes: treating prose as code costs some compression, while
    treating code as prose corrupts it.
    """
    if filename:
        suffix = Path(filename).suffix.lower()
        if suffix in CODE_EXTENSIONS:
            return True
        name = Path(filename).name.lower()
        if name in {"makefile", "dockerfile", "rakefile", "justfile"}:
            return True

    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False
    if lines[0].startswith("#!"):
        return True
    # Strong signals only: a document of indented prose is not source.
    coded = sum(1 for ln in lines if _CODE_STRONG_RE.search(ln))
    return coded / len(lines) >= 0.5


def code_regions(text: str, min_lines: int = CODE_RUN_LINES) -> list[tuple[int, int]]:
    """Character ranges of unfenced code embedded in prose.

    Markdown fences and indented blocks are found by the parser; this catches
    the case the parser cannot see, where source sits in plain text with no
    marker around it. Only runs of ``min_lines`` consecutive code-shaped lines
    qualify, so an occasional prose line ending in a brace is not enough.
    """
    starts = _line_starts(text)
    lines = text.splitlines()
    flags = [bool(ln.strip()) and _code_line(ln) for ln in lines]
    strong = [bool(ln.strip()) and bool(_CODE_STRONG_RE.search(ln)) for ln in lines]

    ranges: list[tuple[int, int]] = []
    index = 0
    while index < len(flags):
        if not flags[index]:
            index += 1
            continue
        end = index
        while end + 1 < len(flags) and (flags[end + 1] or not lines[end + 1].strip()):
            end += 1
        while end > index and not lines[end].strip():
            end -= 1
        # Indentation alone is not enough: a run must contain something only
        # code writes, or an indented paragraph in a list looks like a block.
        if end - index + 1 >= min_lines and any(strong[index : end + 1]):
            begin = starts[index]
            stop = starts[end] + len(lines[end])
            ranges.append((begin, stop))
        index = end + 1
    return ranges


def _verbatim_ranges(text: str, token_types: frozenset[str]) -> list[tuple[int, int]]:
    """Character ranges of blocks that bypass the compressor.

    Parsed rather than pattern-matched, so a pipe inside a fence is not a table
    row and a fence nested in a list item is still found. If the parse fails we
    fall back to fences alone rather than losing the document.
    """
    if not token_types:
        return []
    try:
        tokens = _parser().parse(text)
    except Exception:  # pragma: no cover - a parser bug must not lose the document
        return [m.span() for m in FENCE_RE.finditer(text)]

    starts = _line_starts(text)
    ranges: list[tuple[int, int]] = []
    for token in tokens:
        if token.map is None or token.type not in token_types:
            continue
        first, last = token.map
        begin = starts[first] if first < len(starts) else len(text)
        end = starts[last] if last < len(starts) else len(text)
        # A block's map can run through the blank line after it; leave that out.
        end = begin + len(text[begin:end].rstrip())
        if end > begin:
            ranges.append((begin, end))

    ranges.sort()
    merged: list[tuple[int, int]] = []
    for begin, end in ranges:
        if merged and begin <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((begin, end))
    return merged


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Sort and coalesce overlapping character ranges."""
    merged: list[tuple[int, int]] = []
    for begin, end in sorted(ranges):
        if merged and begin <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((begin, end))
    return merged


def _segments_from_ranges(text: str, ranges: list[tuple[int, int]]) -> list[tuple[str, bool]]:
    segments: list[tuple[str, bool]] = []
    cursor = 0
    for begin, end in ranges:
        if begin > cursor:
            segments.append((text[cursor:begin], False))
        segments.append((text[begin:end], True))
        cursor = end
    if cursor < len(text):
        segments.append((text[cursor:], False))
    return segments or [(text, False)]


def _chunk_prose(text: str, max_tokens: int, spans: list[re.Pattern[str]]) -> list[Chunk]:
    """Chunk a prose segment, cutting at the coarsest boundary that fits."""
    if not text.strip():
        return [Chunk(text=text, sep="", compressible=False)] if text else []

    # Backends strip their output, so keep the outer whitespace outside the chunk.
    lead = text[: len(text) - len(text.lstrip())]
    trail = text[len(text.rstrip()) :]

    units = _to_units(text.strip(), max_tokens)
    packed = _pack(units, max_tokens)

    chunks: list[Chunk] = []
    for body, sep in packed:
        if not body.strip():
            chunks.append(Chunk(text=body, sep=sep, compressible=False))
            continue
        stash: list[str] = []
        if spans:
            body, stash = protect_spans(body, *spans)
        chunks.append(Chunk(text=body, sep=sep, compressible=True, stash=stash))

    chunks[0].prefix = lead + chunks[0].prefix
    chunks[-1].sep += trail
    return chunks


def _to_units(text: str, max_tokens: int) -> list[tuple[str, str]]:
    """Break prose into (text, following-separator) pairs no larger than needed.

    Recursion order is paragraph -> line -> sentence -> words. Each level only
    descends when the piece above it does not fit.
    """
    units: list[tuple[str, str]] = []
    parts = _BLOCK_SPLIT_RE.split(text)
    # ``parts`` alternates block, separator, block, separator, ... block.
    for index in range(0, len(parts), 2):
        block = parts[index]
        sep = parts[index + 1] if index + 1 < len(parts) else ""
        if not block:
            if sep:
                if units:
                    _extend_sep(units, sep)
                else:
                    units.append(("", sep))
            continue
        pieces = _split_block(block, max_tokens)
        for offset, (piece, piece_sep) in enumerate(pieces):
            is_last = offset == len(pieces) - 1
            units.append((piece, sep if is_last else piece_sep))
    return units


def _extend_sep(units: list[tuple[str, str]], sep: str) -> None:
    text, existing = units[-1]
    units[-1] = (text, existing + sep)


def _split_block(block: str, max_tokens: int) -> list[tuple[str, str]]:
    if count_tokens(block) <= max_tokens:
        return [(block, "")]

    pieces: list[tuple[str, str]] = []
    lines = block.split("\n")
    for offset, line in enumerate(lines):
        sep = "\n" if offset < len(lines) - 1 else ""
        if count_tokens(line) <= max_tokens:
            pieces.append((line, sep))
            continue
        sentences = split_sentences(line)
        for s_offset, sentence in enumerate(sentences):
            s_sep = " " if s_offset < len(sentences) - 1 else sep
            if count_tokens(sentence) <= max_tokens:
                pieces.append((sentence, s_sep))
            else:
                fragments = _hard_split(sentence, max_tokens)
                for f_offset, fragment in enumerate(fragments):
                    pieces.append((fragment, " " if f_offset < len(fragments) - 1 else s_sep))
    return pieces


def _hard_split(sentence: str, max_tokens: int) -> list[str]:
    """Last resort: cut a single over-long sentence on word boundaries."""
    words = _WORD_SPLIT_RE.split(sentence.strip())
    fragments: list[str] = []
    current: list[str] = []
    running = 0
    for word in words:
        cost = word_cost(word)
        if current and running + cost > max_tokens:
            fragments.append(" ".join(current))
            current, running = [], 0
        current.append(word)
        running += cost
    if current:
        fragments.append(" ".join(current))
    return fragments or [sentence]


def _pack(units: list[tuple[str, str]], max_tokens: int) -> list[tuple[str, str]]:
    """Greedily merge consecutive units while they fit inside ``max_tokens``."""
    packed: list[tuple[str, str]] = []
    buffer = ""
    buffer_tokens = 0
    pending_sep = ""

    for text, sep in units:
        cost = count_tokens(text)
        # A blank line is a hard boundary: backends collapse newlines, so the
        # chunker has to own paragraph structure.
        too_big = buffer and buffer_tokens + cost > max_tokens
        if buffer and (too_big or _BLANK_LINE_RE.search(pending_sep)):
            packed.append((buffer, pending_sep))
            buffer, buffer_tokens = "", 0
            pending_sep = ""
        buffer = buffer + pending_sep + text if buffer else text
        buffer_tokens += cost
        pending_sep = sep
    if buffer or pending_sep:
        packed.append((buffer, pending_sep))
    return packed


def rejoin(chunks: list[Chunk], outputs: list[str]) -> str:
    """Reassemble compressed chunk outputs using the recorded separators."""
    if len(chunks) != len(outputs):
        raise ValueError(f"got {len(outputs)} outputs for {len(chunks)} chunks")
    return "".join(
        chunk.prefix + restore_spans(output, chunk.stash) + chunk.sep
        for chunk, output in zip(chunks, outputs, strict=True)
    )


def compress_document(
    text: str,
    backend: CompressorBackend,
    rate: float = 0.5,
    *,
    max_tokens: int = DEFAULT_CHUNK_TOKENS,
    preserve_code: bool = True,
    preserve_inline_code: bool = True,
    preserve_markdown: bool = True,
    preserve_numbers: bool = True,
    preserve_identifiers: bool = True,
    **kwargs: Any,
) -> CompressionResult:
    """Chunk ``text``, compress each chunk with ``backend``, and reassemble.

    Compressible chunks go through :meth:`CompressorBackend.compress_batch`, so
    a backend that can batch does. Code blocks and blank runs bypass the
    backend entirely.
    """
    chunks = chunk_document(
        text,
        max_tokens,
        preserve_code=preserve_code,
        preserve_inline_code=preserve_inline_code,
        preserve_markdown=preserve_markdown,
        preserve_numbers=preserve_numbers,
        preserve_identifiers=preserve_identifiers,
    )
    if not chunks:
        return CompressionResult.build(text, text, backend.name, metadata={"chunks": 0})

    live = [i for i, chunk in enumerate(chunks) if chunk.compressible]
    outputs = [chunk.text for chunk in chunks]
    warnings: list[str] = []
    backend_metadata: list[dict[str, Any]] = []

    if live:
        results = backend.compress_batch([chunks[i].text for i in live], rate=rate, **kwargs)
        if len(results) != len(live):
            raise RuntimeError(
                f"{backend.name}.compress_batch returned {len(results)} results "
                f"for {len(live)} chunks"
            )
        for index, result in zip(live, results, strict=True):
            outputs[index] = result.text
            warnings.extend(result.warnings)
            if result.metadata:
                backend_metadata.append(result.metadata)

    compressed = rejoin(chunks, outputs)
    return CompressionResult.build(
        text,
        compressed,
        backend.name,
        warnings=warnings,
        metadata={
            "chunks": len(chunks),
            "compressed_chunks": len(live),
            "code_blocks_preserved": sum(
                1 for c in chunks if not c.compressible and c.text.strip()
            ),
            "max_chunk_tokens": max_tokens,
            "backend_metadata": _merge_metadata(backend_metadata),
        },
    )


def _merge_metadata(items: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Fold per-chunk metadata across the chunks of one document.

    Ints sum as counters and lists concatenate in document order -- a list is a
    record of what happened to each chunk ("pinned_back"), so keeping only the
    first chunk's would report one restored word for a document where twenty
    were put back. Anything else keeps the first value seen, because a scalar
    like the model name or device is the same for every chunk.
    """
    merged: dict[str, Any] = {}
    for item in items:
        for key, value in item.items():
            if isinstance(value, bool):
                merged.setdefault(key, value)
            elif isinstance(value, int):
                merged[key] = merged.get(key, 0) + value
            elif isinstance(value, list):
                merged.setdefault(key, []).extend(value)
            else:
                merged.setdefault(key, value)
    return merged
