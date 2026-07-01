from dataclasses import dataclass
import torch
import argparse
import chromadb.api.client
from chromadb.utils import embedding_functions
from chunking_evaluation import BaseChunker, GeneralEvaluation
from RagAtini import RagAtini

# --- chromadb>=1.5 raises NotFoundError when delete_collection hits a -------
# collection that doesn't exist yet; chunking_evaluation assumes the old
# silent behaviour. Patch the class method so every call is safe. ----------
_orig_delete = chromadb.api.client.Client.delete_collection


def _safe_delete(self, name, *a, **k):
    try:
        return _orig_delete(self, name, *a, **k)
    except Exception:
        return None


chromadb.api.client.Client.delete_collection = _safe_delete

# ========================================================== #

OPENAI_API_KEY = ""

if not OPENAI_API_KEY:
    print("Please fill in OPENAI_API_KEY for the benchmarks to run or use built in embedding model")


# ========================================================== #

@dataclass
class Benchmark:
    name: str
    prominence: float
    f_sig: float
    overlap: bool
    embedder: str = "openai"  # "openai" or "modernbert"
    retrieve: int = -1  # passed to GeneralEvaluation.run(retrieve=...)


BENCHMARKS = {
    "coarse": Benchmark("coarse", prominence=0.5, f_sig=1.0, overlap=False),
    "coarse-overlap": Benchmark("coarse-overlap", prominence=0.5, f_sig=1.0, overlap=True),
    "medium": Benchmark("b", prominence=0.5, f_sig=0.5, overlap=False),
    "medium-overlap": Benchmark("b", prominence=0.5, f_sig=0.5, overlap=True),
    "short": Benchmark("c", prominence=0.1, f_sig=0.25, overlap=False),
    "short-overlap": Benchmark("d", prominence=0.1, f_sig=0.25, overlap=True),
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ragAtini = RagAtini(
    vectorizer_model="nomic-ai/modernbert-embed-base",
    boundary_model="mirth/chonky_modernbert_base_1",
    device=DEVICE
)


class RagAtiniChunker(BaseChunker):
    def __init__(self, prominence: float, f_sig: float, overlap: bool, benchmark_name: str):
        self.prominence = prominence
        self.f_sig = f_sig
        self.overlap = overlap
        self.chunks = []
        self.benchmark_name = benchmark_name

    def split_text(self, text):
        resp = ragAtini.vectorize(
            text,
            prominence=self.prominence,
            f_sig=self.f_sig,
            overlap=self.overlap
        )
        chunks = [s.text for s in resp.segments]

        self.chunks += chunks
        return chunks

    def summary(self):
        chunks = self.chunks
        sizes = [len(c) for c in chunks]
        print(f"{self.benchmark_name}: chunks={len(chunks)} mean_chars={sum(sizes) / len(sizes):.0f}")
        return {
            "benchmark_name": self.benchmark_name,
            "chunks_count": len(chunks),
            "mean_chunk_chars": sum(sizes) / len(sizes)
        }


default_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-large"
)


def _metrics(res):
    # drop the giant per-query corpora_scores; keep scalar summaries as plain floats
    return {k: float(v) for k, v in res.items() if k != "corpora_scores"}


def print_table(rows):
    cols = ["benchmark", "retrieve", "iou_mean", "iou_std", "recall_mean", "recall_std",
            "precision_omega_mean", "precision_mean", "chunks_count", "mean_chunk_chars"]
    head = ["benchmark", "r", "iou", "iou_sd", "recall", "rec_sd", "precW", "prec", "chunks", "chars"]

    def cell(k, v):
        if k == "chunks_count":
            return str(int(v))
        if k == "mean_chunk_chars":
            return f"{v:.0f}"
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    table = [head] + [[cell(k, r.get(k)) for k in cols] for r in rows]
    widths = [max(len(row[i]) for row in table) for i in range(len(head))]

    def line(row):
        return "| " + " | ".join(v.ljust(w) for v, w in zip(row, widths)) + " |"

    print()
    print(line(head))
    print("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for row in table[1:]:
        print(line(row))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmarks", default=",".join(BENCHMARKS),
                   help="comma-separated or 'all', e.g. coarse,coarse-overlap")
    p.add_argument("--retrieve", type=int, default=5, help="5 = top-5, -1 = Min")
    args = p.parse_args()

    names = list(BENCHMARKS) if args.benchmarks == "all" else [n.strip() for n in args.benchmarks.split(",")]

    rows = []
    total = len(names)
    for i, name in enumerate(names, 1):
        bm = BENCHMARKS[name]
        print(f"[{i}/{total}] running {name} (r={args.retrieve}) ...", flush=True)
        chunker = RagAtiniChunker(bm.prominence, bm.f_sig, bm.overlap, name)
        res = GeneralEvaluation().run(chunker, default_ef, retrieve=args.retrieve)
        row = {"benchmark": name, "retrieve": args.retrieve, **chunker.summary(), **_metrics(res)}
        rows.append(row)
        print(f"[{i}/{total}] done    {name:<16} "
              f"IoU {row['iou_mean']:.4f}+-{row['iou_std']:.4f}  "
              f"Recall {row['recall_mean']:.3f}  PrecW {row['precision_omega_mean']:.3f}  "
              f"{int(row['chunks_count'])} chunks / {row['mean_chunk_chars']:.0f} chars", flush=True)

    print_table(rows)
    return rows


if __name__ == "__main__":
    main()
