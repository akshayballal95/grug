# Reproducing the compressor

grug's `lingua2` backend uses a checkpoint someone else trained. This is how to
train your own — corpus, labels, encoder — with the same commands the project
uses.

```bash
pip install 'grug[train]'
```

That pulls torch, transformers and datasets. Plain `pip install grug` stays
free of all three, and `import grug` never imports torch even with the extra
installed.

## The three stages

```bash
grug train prepare --out data/                                  # corpus -> labels
grug train run --data data/ --out ckpt/ --model answerdotai/ModernBERT-base
grug train evaluate --model ckpt/ docs/*.md                     # ratio + faithfulness
```

Then use it like any other backend:

```python
import grug
from grug.backends.modern import ModernBackend

comp = grug.Compressor(ModernBackend(model_name="ckpt/"))
comp.compress(document, rate=0.4)
```

## Stage 1 — `prepare`

Downloads [`microsoft/MeetingBank-LLMCompressed`][corpus]: 5,169 meeting
transcripts with GPT-4 compressions, released by the LLMLingua-2 authors.

The corpus ships (original, compressed) **text pairs, not labels**, so the
labels are derived here with the paper's alignment algorithm — a sliding-window
fuzzy match that handles the three things a teacher does to its input:

| Problem | Example | Handled by |
| --- | --- | --- |
| Ambiguity | `program` appears three times | searching outward from the last match |
| Variation | `consenting` → `Consent` | prefix and ratio matching |
| Reordering | `properties within … inclusion` → `properties inclusion` | left search after right fails |

Two quality metrics then discard the worst examples, as in the paper: the top
5% by **variation rate** (words in the output absent from the input — a
hallucination signal) and the top 10% by **alignment gap** (`HR − MR`, how much
of the compression the labels failed to explain).

On the first 25 rows: 285 chunk pairs, 245 kept, **keep rate 0.31** — which
matches the paper's reported ~3.1× compression, a useful check that the
alignment is faithful.

> The corpus is **CC-BY-NC-SA-4.0**. A model trained on it inherits a
> non-commercial constraint. Check this before shipping a checkpoint.

### Skipping stage 1

Deriving labels costs ~8 minutes of single-threaded alignment. On a rented GPU
that is 8 minutes of paying for an idle card, so the derived labels are cached:

```bash
grug train prepare --out data/ --from-hub akshayballal/grug-meetingbank-labels
```

That takes about 4 seconds, and falls back to deriving if the cache is missing
or you point it at a corpus of your own. To publish your own:

```bash
grug train prepare --out data/ --push-to-hub you/your-labels
```

## Stage 2 — `run`

Fine-tunes an encoder into a binary preserve/discard token classifier:

```
h = f_θ(x);  p(xᵢ) = softmax(W hᵢ + b);  L = mean CrossEntropy
```

Defaults follow the paper — 10 epochs, AdamW, lr 1e-5, batch 10. Every
sub-word of a word carries that word's label, which matches inference, where
the backend averages sub-word probabilities to score the word.

| Flag | Default | Notes |
| --- | --- | --- |
| `--model` | `answerdotai/ModernBERT-base` | any fast-tokenizer encoder |
| `--epochs` / `--lr` / `--batch-size` | `10` / `1e-5` / `10` | the paper's settings |
| `--max-length` | `512` | raise toward 8192 for a modern encoder |
| `--cs-weight` | `0.0` | see below |

`--cs-weight` adds [MOOSComp][moos]'s inter-class cosine similarity loss, which
penalises preserve and discard representations for being similar in the final
layer. Encoders over-smooth: representations converge with depth, and the
classifier reads the layer where the two classes are hardest to tell apart.
Leave it at `0.0` to reproduce LLMLingua-2 exactly; raise it to experiment.

## Stage 3 — `evaluate`

Token accuracy tells you the classifier learned the labels. It does not tell
you the compressed prompt is still *true*, so evaluation reports faithfulness
as a first-class metric, using the same verifier that runs in production:

```
rate=0.50  ratio=0.52  clean=0.86  0.19s/doc
```

`clean` is the fraction of documents `grug.verify()` had no complaint about,
broken down by negation loss, negation scope loss, number loss and entity loss.
A checkpoint that beats the baseline on ratio but loses more negations is worse
for grug's purpose, and this is where you would see that.

## Choosing an encoder

The binding constraint on LLMLingua-2 is its 512 word-piece window — the reason
grug chunks at 450 tokens. Every modern encoder gives 8192.

| Encoder | Params | Ctx | Notes |
| --- | --- | --- | --- |
| `answerdotai/ModernBERT-base` | 149M | 8192 | English **+ code** — best fit for docs and READMEs |
| `jhu-clsp/mmBERT-base` | 307M | 8192 | multilingual, drop-in for XLM-R at half the size |
| `jhu-clsp/mmBERT-small` | 140M | 8192 | replaces the mBERT "small" variant |
| `EuroBERT/EuroBERT-210m` | 210M | 8192 | 15 languages |
| `LiquidAI/LFM2.5-Encoder-350M` | 354M | 8192 | fastest on CPU |

Note that the public corpus is pre-chunked to ≤512 tokens, because that is what
GPT-4 and the old encoders needed. So retraining on it measures the **encoder**
gain, not the long-context gain — you only unlock that by re-distilling a
corpus with long examples.

## Smoke run

The whole pipeline on a slice, in about a minute on CPU:

```bash
grug train prepare --limit 25 --out /tmp/g/data
grug train run --data /tmp/g/data --out /tmp/g/ckpt \
  --model answerdotai/ModernBERT-base --epochs 1 --batch-size 4 --device cpu
```

[corpus]: https://huggingface.co/datasets/microsoft/MeetingBank-LLMCompressed
[moos]: https://arxiv.org/abs/2504.16786
