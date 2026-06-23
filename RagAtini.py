import re
import bisect
import torch
import numpy as np
import copy
from dataclasses import dataclass
from typing import List, Tuple, Optional
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

chonky_tokenizer = AutoTokenizer.from_pretrained("mirth/chonky_modernbert_base_1", model_max_length=1024)
chonky_model = AutoModelForTokenClassification.from_pretrained(
    "mirth/chonky_modernbert_base_1", num_labels=2, id2label={0: "O", 1: "separator"}, label2id={"O": 0, "separator": 1}
)
chonky_model.eval()
chonky = pipeline(
    "ner", model=chonky_model, tokenizer=chonky_tokenizer, aggregation_strategy="simple", device=-1, batch_size=8
)


def find_boundaries(texts, min_score=0.6):
    if isinstance(texts, str):
        texts = [texts]
    results = chonky(texts)  # one batched call; list-per-input
    if texts and isinstance(results[0], dict):  # single input -> pipeline returns a flat list
        results = [results]
    return [[e["end"] for e in r if e["score"] > min_score] for r in results]


@dataclass
class RagSegment:
    vector: torch.Tensor
    text: Optional[str] = None
    text_coords: Optional[Tuple[int, int]] = None


@dataclass
class RagAtiniResponse:
    velocity: torch.Tensor
    peaks: np.ndarray
    token_ids: torch.Tensor
    token_vectors: torch.Tensor
    recoded_text: str
    segments: List[RagSegment]


class RagAtini:
    def __init__(self, model, tokenizer, max_chunk_length=None, doc_prefix="search_document: "):
        self.model = model
        self.device = next(self.model.parameters()).device if hasattr(self.model, "parameters") else "cpu"

        self.max_context_window = max_chunk_length if max_chunk_length else getattr(tokenizer, "model_max_length", 8192)
        assert self.max_context_window, "Failed to resolve max_context_window"

        self.tokenizer = copy.deepcopy(tokenizer)
        self.tokenizer.model_max_length = int(1e9)
        self.pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        self.cls_id = self.tokenizer.cls_token_id if self.tokenizer.cls_token_id is not None else self.tokenizer.bos_token_id
        self.sep_id = self.tokenizer.sep_token_id if self.tokenizer.sep_token_id is not None else self.tokenizer.eos_token_id

        self.prefix_ids = (self.tokenizer(doc_prefix, add_special_tokens=False)["input_ids"]
                           if doc_prefix else [])
        self.prefix_len = len(self.prefix_ids)

        self.default_stride = int(self.max_context_window * 0.25)
        self.sigma = max(10, self.max_context_window // 100)

        self.find_boundaries = find_boundaries
        if hasattr(self.model, "eval"):
            self.model.eval()

    def tokenize(self, text: str):
        enc = self.tokenizer(text, add_special_tokens=False, return_tensors="pt",
                             return_offsets_mapping=True)
        tokens = enc["input_ids"].to(self.device)
        offsets = enc["offset_mapping"][0].tolist()
        return tokens, offsets

    def get_token_offsets(self, document: str, offsets):
        token_to_char = [s for s, _ in offsets]
        token_to_char.append(len(document))  # sentinel end so token_to_char[last_token] is valid

        char_to_token = [0] * len(document)
        for i, (s, e) in enumerate(offsets):
            for c in range(s, e):
                char_to_token[c] = i

        return token_to_char, char_to_token, document

    def chunk_tokens(self, tokens, stride):
        window_size = self.max_context_window - self.prefix_len - 2
        tokens_len = tokens.size(0)

        prefix_tensor = torch.tensor(self.prefix_ids, dtype=torch.long, device=self.device)
        cls_tensor = torch.tensor([self.cls_id], dtype=torch.long, device=self.device)
        sep_tensor = torch.tensor([self.sep_id], dtype=torch.long, device=self.device)

        chunks = []
        for i in range(0, max(1, tokens_len), stride):
            chunk = tokens[i:i + window_size]

            chunk_ids = torch.cat([cls_tensor, prefix_tensor, chunk, sep_tensor])

            if chunk_ids.size(0) < self.max_context_window:
                pad_tensor = torch.full((self.max_context_window - chunk_ids.size(0),), self.pad_id,
                                        dtype=torch.long, device=self.device)
                chunk_ids = torch.cat([chunk_ids, pad_tensor])

            chunks.append(chunk_ids)

            if i + window_size >= tokens_len:
                break

        return torch.stack(chunks) if chunks else torch.empty((0, self.max_context_window), dtype=torch.long,
                                                              device=self.device)

    def process_chunks(self, chunks, batch_size: int):
        if chunks.size(0) == 0:
            hidden_dim = self.model.config.hidden_size if hasattr(self.model, "config") else 768
            return torch.empty((0, chunks.size(1), hidden_dim), device=self.device)

        all_outputs = []
        with torch.no_grad():
            for i in range(0, chunks.size(0), batch_size):
                batch = chunks[i:i + batch_size]
                attention_mask = (batch != self.pad_id).long()
                outputs = self.model(batch, attention_mask=attention_mask).last_hidden_state
                all_outputs.append(outputs)

        return torch.cat(all_outputs, dim=0)

    def apply_gaussian(self, vectors, sigma: int):
        tokens_len = vectors.size(0)
        if tokens_len == 0:
            return vectors

        vectors_np = vectors.float().cpu().numpy()
        smoothed_np = gaussian_filter1d(vectors_np, sigma=sigma, axis=0)

        return torch.tensor(smoothed_np, device=self.device, dtype=vectors.dtype)

    def calculate_velocity(self, vectors):
        tokens_len = vectors.size(0)
        if tokens_len < 2:
            return torch.zeros(tokens_len, device=self.device, dtype=vectors.dtype)

        velocity = torch.zeros(tokens_len, device=self.device, dtype=vectors.dtype)
        velocity[1:] = torch.norm(vectors[1:] - vectors[:-1], dim=1)
        return velocity

    def generate_chunk_masks(self, num_chunks: int, window_size: int, tokens_len: int, stride: int):
        if num_chunks == 0:
            return torch.empty((0, 0), device=self.device, dtype=torch.float32)

        chunk_masks = torch.zeros((num_chunks, window_size), device=self.device, dtype=torch.float32)
        base_window = 0.5 * (
                1.0 - torch.cos(2.0 * torch.pi * torch.arange(window_size, device=self.device) / (window_size - 1)))

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * stride
            end_idx = min(start_idx + window_size, tokens_len)
            chunk_len = end_idx - start_idx

            chunk_mask = base_window[:chunk_len].clone()

            if chunk_idx == 0:
                chunk_mask[:chunk_len // 2] = 1.0
            if chunk_idx == num_chunks - 1:
                chunk_mask[chunk_len // 2:] = 1.0
                taper_len = min(256, chunk_len)
                if taper_len > 1:
                    taper = 0.5 * (1.0 + torch.cos(
                        torch.pi * torch.arange(taper_len, device=self.device) / (taper_len - 1)))
                    chunk_mask[-taper_len:] = taper
                elif taper_len == 1:
                    chunk_mask[-1] = 0.0

            chunk_masks[chunk_idx, :chunk_len] = chunk_mask

        return chunk_masks

    def mesh_vectors(self, chunk_vectors, chunk_masks, tokens_len: int, stride: int):
        if chunk_vectors.size(0) == 0:
            hidden_dim = chunk_vectors.size(-1) if chunk_vectors.dim() == 3 else 768
            return torch.empty((0, hidden_dim), device=self.device)

        window_size = chunk_masks.size(1)
        hidden_dim = chunk_vectors.size(2)

        sum_vec = torch.zeros((tokens_len, hidden_dim), device=self.device, dtype=chunk_vectors.dtype)
        weight_vec = torch.zeros((tokens_len, 1), device=self.device, dtype=chunk_vectors.dtype)

        for chunk_idx in range(chunk_vectors.size(0)):
            start_idx = chunk_idx * stride
            end_idx = min(start_idx + window_size, tokens_len)
            chunk_len = end_idx - start_idx

            mask_expanded = chunk_masks[chunk_idx, :chunk_len].unsqueeze(-1)

            valid_vectors = chunk_vectors[chunk_idx, 1 + self.prefix_len:1 + self.prefix_len + chunk_len, :]

            sum_vec[start_idx:end_idx] += valid_vectors * mask_expanded
            weight_vec[start_idx:end_idx] += mask_expanded

        weight_vec[weight_vec == 0] = 1.0
        return sum_vec / weight_vec

    def detect_peaks(self, semantic_velocity, distance: int, prominence: float = 4.0):
        if isinstance(semantic_velocity, torch.Tensor):
            vel_np = semantic_velocity.cpu().numpy()
        else:
            vel_np = semantic_velocity

        median_vel = np.median(vel_np)
        mad = max(1e-6, np.median(np.abs(vel_np - median_vel)))
        min_prominence = mad * prominence

        peaks, _ = find_peaks(vel_np, distance=distance, prominence=min_prominence)

        return peaks

    def snap_to_boundary(self, boundaries, text, segment_start, segment_end, overlap: int = 0):
        def nearest(pos):
            i = bisect.bisect_left(boundaries, pos)
            cands = []
            if i < len(boundaries): cands.append(i)
            if i > 0: cands.append(i - 1)
            return min(cands, key=lambda j: abs(boundaries[j] - pos))

        b_start = nearest(segment_start)
        b_end = nearest(segment_end)

        b_start = max(0, b_start - overlap)
        b_end = min(len(boundaries) - 1, b_end + overlap)

        first_char = boundaries[b_start]
        last_char = boundaries[b_end]

        return first_char, last_char, text[first_char:last_char]

    def peak_adjacent_boundaries(self, peak_chars, recoded_text, radius=512):
        peak_chars = sorted(int(c) for c in peak_chars)
        n = len(recoded_text)

        text_segments = []
        segment_offsets = []
        for i, peak in enumerate(peak_chars):
            left = peak_chars[i - 1] if i > 0 else 0
            right = peak_chars[i + 1] if i + 1 < len(peak_chars) else n
            from_char = max(peak - radius, left)
            to_char = min(peak + radius, right)
            text_segments.append(recoded_text[from_char:to_char])
            segment_offsets.append(from_char)

        boundaries = self.find_boundaries(text_segments)

        refined = []
        for seps, base in zip(boundaries, segment_offsets):
            refined.extend(base + s for s in seps)

        return sorted(set(refined))

    def vectorize(self,
                  document: str,
                  internal_batch: int = 1,
                  stride: int = None,
                  f_sig: float = 1.0,
                  prominence: float = 4.0,
                  overlap: int = 0
                  ):
        stride = stride if stride else self.default_stride
        sigma = self.sigma * f_sig

        tokens, offsets = self.tokenize(document)
        tokens = tokens.squeeze(0)
        tokens_len = tokens.size(0)

        if tokens_len == 0:
            raise ValueError("Input document resulted in zero tokens. Cannot process empty documents.")

        token_to_char, char_to_token, recoded_text = self.get_token_offsets(document, offsets)
        chunks = self.chunk_tokens(tokens, stride)
        chunk_vectors = self.process_chunks(chunks, internal_batch)

        window_size = chunk_vectors.size(1) - self.prefix_len - 2
        num_chunks = chunk_vectors.size(0)

        chunk_masks = self.generate_chunk_masks(num_chunks, window_size, tokens_len, stride)
        meshed_vectors = self.mesh_vectors(chunk_vectors, chunk_masks, tokens_len, stride)
        smoothed_vectors = self.apply_gaussian(meshed_vectors, sigma)
        semantic_velocity = self.calculate_velocity(smoothed_vectors)
        semantic_peaks = self.detect_peaks(semantic_velocity, sigma, prominence)
        semantic_peak_chars = [token_to_char[p] for p in semantic_peaks]

        boundaries = sorted(
            [0] + self.peak_adjacent_boundaries(semantic_peak_chars, recoded_text) + [len(recoded_text)])

        segments = []
        offset = 0
        for peak in np.append(semantic_peaks, len(tokens) - 1):
            first_char, last_char, segment_text = self.snap_to_boundary(
                boundaries, recoded_text, token_to_char[offset], token_to_char[peak], overlap
            )

            segments.append(RagSegment(
                vector=meshed_vectors[offset:peak].mean(dim=0),
                text=segment_text,  # .strip(),
                text_coords=(first_char, last_char)
            ))
            offset = peak

        return RagAtiniResponse(
            velocity=semantic_velocity,
            peaks=semantic_peaks,
            token_ids=tokens,
            token_vectors=meshed_vectors,
            recoded_text=recoded_text,
            segments=segments
        )
