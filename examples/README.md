# grug examples

Runnable scripts. Each one is standalone — run it from anywhere:

```bash
python examples/rules_backend.py
```

| Script | Needs | What it shows |
| --- | --- | --- |
| [`rules_backend.py`](rules_backend.py) | `pip install grugify` | The dependency-free backend: rate sweep, what it refuses to drop, batching, composing rules, adding a language. |
| [`compare_backends.py`](compare_backends.py) | `pip install grugify` | Every runnable backend on the same document, side by side — tokens, ratio, seconds, warnings. Pass `--model` to include the classifier. |
| [`faithfulness.py`](faithfulness.py) | `pip install grugify` | Why the verifier exists: the negation problem, the three checks, and load-bearing sentences run through every backend. |

`sample_doc.md` is the shared input — a short incident report with a fenced code
block, several number formats, inline code, a URL, and negations that carry the
meaning of their paragraphs. `_shared.py` is a tiny printing helper, not part of
the grug API.

## Flags

`compare_backends.py` takes options:

```bash
python examples/compare_backends.py --rates 0.7 0.5 0.3
python examples/compare_backends.py --model akshayballal/grug-mmbert-small-meetingbank
```

The first classifier run downloads the checkpoint; later runs load from the
Hugging Face cache in a second or two.

## The same thing from the CLI

```bash
grug backends                                        # what is installed
grug compress examples/sample_doc.md --rate 0.5      # writes sample_doc.grug.md
grug compress examples/sample_doc.md --json -q | jq '{ratio, warnings}'
grug verify examples/sample_doc.md examples/sample_doc.grug.md
```

## Expected shape of the results

On `sample_doc.md` (406 tokens, of which 70 are an incompressible code block):

| backend | rate 0.8 | rate 0.5 | rate 0.3 |
| --- | --- | --- | --- |
| `rules` | 0.79 | 0.73 | 0.73 |
| `classifier` | ~0.8 | ~0.55 | ~0.4 |

`rules` floors out because it only deletes words its rules nominate. The
classifier keeps going because it scores every word. Both report the ratio they
actually achieved. Numbers will shift with your document; the shape will not.
