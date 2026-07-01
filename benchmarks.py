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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmarks", default=",".join(BENCHMARKS),
                   help="comma-separated, e.g. coarse,coarse-overlap")
    p.add_argument("--retrieve", type=int, default=5, help="5 = top-5, -1 = Min")
    args = p.parse_args()

    names = list(BENCHMARKS) if args.benchmarks == "all" else [n.strip() for n in args.benchmarks.split(",")]

    rows = []
    for name in names:
        bm = BENCHMARKS[name]
        chunker = RagAtiniChunker(bm.prominence, bm.f_sig, bm.overlap, name)
        res = GeneralEvaluation().run(chunker, default_ef, retrieve=args.retrieve)
        print(name, "retrieve=%d" % args.retrieve, res)
        rows.append({**chunker.summary(), "retrieve": args.retrieve, **res})
    return rows


if __name__ == "__main__":
    main()
