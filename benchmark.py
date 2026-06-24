OPENAI_API_KEY = ""

import torch
from transformers import AutoTokenizer, AutoModel
from chromadb import Documents, EmbeddingFunction, Embeddings
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "nomic-ai/modernbert-embed-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(DEVICE)
ragAtini = RagAtini(model, tokenizer)

call_count = 0


class RagAtiniChunker(BaseChunker):
    def split_text(self, text):
        global call_count
        call_count += 1
        print(f"{call_count}")
        resp = ragAtini.vectorize(text, prominence=0.01, overlap=0, f_sig=0.25)
        return [s.text for s in resp.segments]


class ModernBertEF(EmbeddingFunction):
    def __init__(self, model, tokenizer, device, prefix="search_document: "):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.prefix = prefix

    def __call__(self, input: Documents) -> Embeddings:
        out = []
        with torch.no_grad():
            for doc in input:
                enc = self.tokenizer(self.prefix + doc, return_tensors="pt",
                                     truncation=True, max_length=8192).to(self.device)
                hidden = self.model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).float()
                emb = (hidden * mask).sum(1) / mask.sum(1)  # masked mean pool
                out.append(emb[0].cpu().tolist())
        return out


ef = ModernBertEF(model, tokenizer, DEVICE)

# Choose embedding function
default_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-large"
)

results = GeneralEvaluation().run(RagAtiniChunker(), default_ef)
print(results)
