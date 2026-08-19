"""The built-in English pack: word classes, pleasantry phrases, negation vetoes.

Word classes are ordered cheapest-to-lose first; the priorities are spaced so
a custom class can slot between two built-ins.
"""

from __future__ import annotations

from ...verify import NEGATION_WORDS
from .core import Language, PhraseRule, RuleSet, WordClassRule, register_language

__all__ = ["ENGLISH"]

# fmt: off
FILLERS = frozenset({
    "basically", "actually", "really", "very", "quite", "simply", "just",
    "essentially", "literally", "obviously", "clearly", "indeed", "truly",
    "extremely", "totally", "utterly", "fairly", "somewhat", "kind", "sort",
    "well", "so", "then", "thus", "hence", "therefore", "moreover",
    "furthermore", "additionally", "please", "kindly", "thanks", "thank",
    "hello", "hi", "regards", "sincerely",
})

ARTICLES = frozenset({"a", "an", "the"})

COPULAS = frozenset({
    "is", "are", "was", "were", "am", "be", "been", "being",
    "has", "have", "had", "does", "do", "did", "get", "gets", "got",
})

FUNCTION_WORDS = frozenset({
    "of", "to", "in", "on", "at", "for", "with", "from", "by", "as",
    "that", "which", "who", "whom", "whose", "into", "onto", "upon",
    "and", "also", "about", "over", "under", "between", "through",
    "during", "while", "when", "where", "than", "such", "some", "any",
})

PRONOUNS = frozenset({
    "it", "its", "they", "them", "their", "theirs", "we", "our", "ours",
    "us", "you", "your", "yours", "he", "him", "his", "she", "her", "hers",
    "i", "my", "me", "mine", "this", "these", "those", "there", "here",
})
# fmt: on

#: Phrases that carry no information at all, with optional terser replacements.
PLEASANTRIES: tuple[tuple[str, str], ...] = (
    (r"\bit is important to note that\b", ""),
    (r"\bit is important to (?:understand|remember|realize) that\b", ""),
    (r"\bit is important to\b", ""),
    (r"\bit should be noted that\b", ""),
    (r"\bit is worth noting that\b", ""),
    (r"\bplease note that\b", ""),
    (r"\bplease be aware that\b", ""),
    (r"\bplease note\b", ""),
    (r"\bas you (?:can see|may know|are aware)\b", ""),
    (r"\bneedless to say\b", ""),
    (r"\bas a matter of fact\b", ""),
    (r"\bkeep in mind that\b", ""),
    (r"\bbear in mind that\b", ""),
    (r"\bthe fact that\b", "that"),
    (r"\bdue to the fact that\b", "because"),
    (r"\bin spite of the fact that\b", "although"),
    (r"\bin order to\b", "to"),
    (r"\bin the event that\b", "if"),
    (r"\bat this point in time\b", "now"),
    (r"\bfor the purpose of\b", "for"),
    (r"\bwith (?:regard|respect) to\b", "about"),
    (r"\bin terms of\b", "in"),
    (r"\ba (?:large |great )?number of\b", "many"),
    (r"\bin the near future\b", "soon"),
    (r"\bon a regular basis\b", "regularly"),
    (r"\bhope this helps\b", ""),
    (r"\bfeel free to\b", ""),
    (r"\blet me know if\b", "if"),
)

ENGLISH = register_language(
    Language(
        code="en",
        rules=RuleSet(
            PhraseRule("pleasantries", PLEASANTRIES),
            WordClassRule("fillers", FILLERS, priority=10),
            WordClassRule("articles", ARTICLES, priority=20),
            WordClassRule("copulas", COPULAS, priority=30),
            WordClassRule("function-words", FUNCTION_WORDS, priority=40),
            WordClassRule("pronouns", PRONOUNS, priority=50),
        ),
        never_drop=NEGATION_WORDS,
    )
)
