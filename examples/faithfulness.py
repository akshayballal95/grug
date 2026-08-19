#!/usr/bin/env python3
"""What the verifier is for, on every backend you have installed.

    python examples/faithfulness.py

Losing detail makes a prompt vaguer. Losing a negation makes it *wrong* -- in
fluent, confident prose that nothing downstream will question. This script
shows the failure, the defaults that prevent it, and the check that catches it
when they do not.
"""

from __future__ import annotations

from _shared import rule

import grug

# Sentences whose meaning lives entirely in one small word.
LOAD_BEARING = [
    "Bills scale with volume, not price.",
    "A forced migration cannot be rolled back once the ledger is frozen.",
    "No customer was billed twice and no invoice was lost.",
    "Do not raise the flush interval without running a load test.",
    "The historical assignment is not exposed through the API.",
]


def available_backends() -> list[str]:
    """Backends usable with no arguments.

    ``available`` only means the dependencies are installed. ``modern`` needs a
    checkpoint name, so it is installed and unusable at the same time.
    """
    return [
        row["name"]
        for row in grug.backend_info()
        if row["available"] and not row["requires_configuration"]
    ]


def main() -> int:
    rule("1. The problem, demonstrated")
    original = "Bills scale with volume, not price."
    naive = "bills scale volume price"
    print(f"  original  : {original}")
    print(f"  naive     : {naive}")
    print("  The compression is 47% smaller and asserts the opposite of the source.")
    print("  grug.verify() catches it:")
    for warning in grug.verify(original, naive):
        print(f"    WARN {warning}")

    rule("2. The three checks")
    cases = [
        ("negation", "The build did not pass.", "build passed"),
        ("number", "We ran 1,250 accounts through the trial.", "ran accounts through trial"),
        ("entity", "Acme Corporation reported the regression.", "reported regression"),
        ("clean", "We ran 1,250 accounts.", "ran 1,250 accounts"),
    ]
    for label, source, compressed in cases:
        found = grug.verify(source, compressed)
        print(f"  {label:<9} {found if found else 'no warnings'}")

    rule("3. Load-bearing negations through every installed backend")
    backends = available_backends()
    print(f"  backends: {', '.join(backends)}\n")
    for sentence in LOAD_BEARING:
        print(f"  {sentence}")
        for name in backends:
            result = grug.compress(sentence, rate=0.3, backend=name)
            flag = "WARN" if result.warnings else "ok  "
            print(f"    {flag} {name:<8} {result.text}")
            for warning in result.warnings:
                print(f"         -> {warning}")
        print()

    rule("4. Turning the safety off, to show it is doing something")
    from grug.backends.rules import RulesBackend

    sentence = "The migration is not automatic and no data is deleted."
    safe = RulesBackend().compress(sentence, rate=0.2)
    unsafe = RulesBackend(keep_words=set()).compress(sentence, rate=0.2, drop_pleasantries=False)
    print(f"  default (negations pinned) : {safe.text}")
    print(f"  warnings                   : {safe.warnings or 'none'}")
    print()
    print("  rules excludes negations from its candidate list outright; lingua2")
    print("  pins them via force_tokens -- a strong default, not a guarantee,")
    print("  which is why the verifier exists.")
    print(f"  (same sentence, other options off: {unsafe.text})")

    rule("5. CI gating")
    print(
        "  grug compress prompt.md --rate 0.4\n"
        "    exit 0  clean\n"
        "    exit 1  error\n"
        "    exit 2  compressed, but the verifier flagged something\n\n"
        "  Treat exit 2 as 'a human should read this diff', not as a failure."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
