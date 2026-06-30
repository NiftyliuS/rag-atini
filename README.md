# RagAtini

Semantic-velocity document chunker. RagAtini segments a document by embedding it
through a transformer, tracking how fast the meaning shifts from token to token
("semantic velocity"), cutting at the sharp transitions, and snapping each cut to
a real sentence/paragraph boundary with a neural splitter.

The result is a set of coherent, boundary-aligned chunks whose granularity is
tunable — from a few large sections to many small passages — without re-running
the expensive embedding pass.

## How it works

1. **Embed** the document through an embedding model (e.g. `nomic-ai/modernbert-embed-base`),
   meshing per-token vectors across overlapping windows so even long documents get
   one continuous vector per token.
2. **Smooth** the token-vector sequence (Gaussian), then compute **semantic velocity** —
   the norm of the difference between consecutive smoothed vectors. Velocity spikes
   where the topic shifts.
3. **Detect peaks** in the velocity curve (prominence-gated). Each peak is a candidate cut.
4. **Snap** each cut to the nearest real boundary using the `chonky` neural sentence
   splitter, so chunks never break mid-sentence or mid-word.

The expensive work (steps 1–2) is prominence-independent. Granularity (step 3) is a
cheap re-slice — see [`.to()`](#re-slicing-without-re-embedding).

## Installation

```bash
pip install torch transformers scipy numpy
# the chonky boundary splitter is pulled from the Hugging Face hub on first run
```

For the plotting helpers:

```bash
pip install umap-learn scikit-learn matplotlib
```

## Quick start

```python
import torch
from transformers import AutoTokenizer, AutoModel
from RagAtini import RagAtini

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "nomic-ai/modernbert-embed-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

ragatini = RagAtini(model, tokenizer)

with open("document.txt") as f:
    document = f.read()

response = ragatini.vectorize(document, f_sig=0.5, prominence=0.5)

for seg in response.segments:
    start, end = seg.text_coords
    print(f"[{start}:{end}] {seg.text[:80]}...")
```

Each `segment.text` is exactly `document[start:end]` — chunks are verbatim slices
of the source, never paraphrased or reconstructed.

## API

### `RagAtini(model, tokenizer, max_chunk_length=None, doc_prefix="search_document: ")`

Wraps an embedding model and tokenizer. `max_chunk_length` defaults to the
tokenizer's `model_max_length`. `doc_prefix` is prepended to each window before
embedding (use the prefix your embedding model expects).

### `vectorize(document, *, f_sig=1.0, prominence=0.5, overlap=False, min_chunk_size=100, ...) -> RagAtiniResponse`

Runs the full pipeline and returns a response containing the cuts.

| parameter | default | effect |
|---|---|---|
| `f_sig` | `1.0` | Smoothing scale. Lower = finer chunks (less smoothing, more peaks survive). `1.0` ≈ large sections, `0.5` ≈ balanced, `0.25` ≈ fine passages. |
| `prominence` | `0.5` | How sharp a velocity peak must be (relative to the local noise floor) to become a cut. Higher = fewer, stronger cuts. |
| `overlap` | `False` | If `True`, each chunk extends one boundary past its cut on both sides. Useful only at fine granularity, where it bridges evidence split across small chunks. |
| `min_chunk_size` | `100` | Minimum chunk length in characters. Slivers merge into the next chunk. |

### `RagAtiniResponse`

| attribute | type | description |
|---|---|---|
| `segments` | `list[RagAtiniTextSegment]` | The chunks, in document order. |
| `peaks` | `np.ndarray` | Token indices of the velocity peaks used as cuts. |
| `prominence`, `overlap`, `min_chunk_size` | | The settings this response was built with. |

`RagAtiniTextSegment` has `.text` (the verbatim chunk) and `.text_coords`
(`(start_char, end_char)` into the original document).

### Re-slicing without re-embedding

The costly pipeline runs once. To get a different granularity from the **same**
embedding pass, call `.to()` on a response — it re-detects peaks and rebuilds
segments, but reuses the cached velocity curve and boundary pool:

```python
response = ragatini.vectorize(document, f_sig=0.5)

coarse = response.to(prominence=4.0)   # fewer, larger chunks
fine   = response.to(prominence=0.1)   # more, smaller chunks
```

`.to()` returns a fresh response and does not mutate the original, so you can hold
several granularities at once (e.g. to build a hierarchy externally). Note that
`.to()` only varies `prominence`, `overlap`, and `min_chunk_size` — changing the
smoothing (`f_sig`) requires a new `vectorize` call, because it reshapes the
velocity curve itself.

## Visualizing

The `charts.py` helpers plot the velocity curve and the semantic trajectory.

```python
from charts import peak_velocity_chart, umap_chart_2d

velocity = response._request.velocity.cpu().numpy()
vectors  = response._request.vectors.cpu().numpy()   # meshed token vectors
peaks    = response.peaks

peak_velocity_chart(velocity, peaks)   # the velocity curve with cuts marked
umap_chart_2d(vectors, peaks)          # the semantic trajectory, colored by chunk
umap_chart_2d(vectors)                 # trajectory colored by token index (no cuts)
```

### Semantic velocity with cuts

The velocity curve is the heart of the method: peaks (circles) are where meaning
shifts fastest, and become chunk boundaries. The same document can be cut coarsely
or finely by changing the prominence threshold.

**`prominence=0.5`** — balanced chunking, cuts at every clear transition:

![Velocity peaks at prominence 0.5](plots/paper_peaks_0_5.png)

**`prominence=4.0`** — only the strongest transitions survive, yielding a handful
of large sections:

![Velocity peaks at prominence 4.0](plots/paper_peaks_4_0.png)

### Semantic trajectory

Projecting the smoothed token vectors to 2D (UMAP) shows the document as a
continuous path through meaning space. Color runs from the start of the document
to the end — the trajectory is one long thread, which is why chunking works by
cutting *along* it at the points of fastest change rather than by clustering.

![UMAP semantic trajectory](plots/paper_umap.png)

## Notes

- Chunks are verbatim character-range slices of the input. `split_text`-style usage
  returns `document[a:b]` exactly, so downstream evidence-locating (e.g. `.find()`)
  works on the original text.
- The boundary splitter (`chonky`) is a prose model. It refines cuts onto real
  sentence/paragraph boundaries even at fine granularity, so small chunks stay
  coherent rather than fragmenting mid-structure.
- For single-topic documents, the semantic trajectory is a 1D sequence with no
  cross-document clusters — chunking captures the structure by cutting the sequence,
  not by grouping. Cross-segment *topical* grouping can be derived from the segment
  vectors; cross-segment *referential* links (e.g. a method and its results) require
  sparse/lexical signals, not dense vectors.