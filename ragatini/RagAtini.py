import bisect
import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from transformers import AutoTokenizer, AutoModel, AutoModelForTokenClassification, pipeline


def detect_peaks(semantic_velocity, distance: int, prominence: float = 0.5):
    if isinstance(semantic_velocity, torch.Tensor):
        vel_np = semantic_velocity.float().cpu().numpy()
    else:
        vel_np = semantic_velocity

    median_vel = np.median(vel_np)
    mad = max(1e-6, np.median(np.abs(vel_np - median_vel)))
    min_prominence = mad * prominence

    peaks, _ = find_peaks(vel_np, distance=distance, prominence=min_prominence)

    return peaks


def find_nearest_boundary(boundaries, pos):
    i = bisect.bisect_left(boundaries, pos)
    cands = []
    if i < len(boundaries): cands.append(i)
    if i > 0: cands.append(i - 1)
    return min(cands, key=lambda j: abs(boundaries[j] - pos))


def snap_to_boundary(boundaries: List[int], segment_start: int, segment_end: int, overlap: bool):
    b_start = find_nearest_boundary(boundaries, segment_start)
    b_end = find_nearest_boundary(boundaries, segment_end)

    if overlap:
        if boundaries[b_start] >= segment_start:
            b_start = max(0, b_start - 1)
        if boundaries[b_end] <= segment_end:
            b_end = min(len(boundaries) - 1, b_end + 1)

    first_char = boundaries[b_start]
    last_char = boundaries[b_end]

    return first_char, last_char


@dataclass
class RagAtiniTextSegment:
    text: Optional[str] = None
    text_coords: Optional[Tuple[int, int]] = None


@dataclass
class RagAtiniVectorizeRequest:
    sigma: int
    document: str
    boundaries: List[int]
    velocity: torch.Tensor
    vectors: torch.Tensor
    token_to_char: List[int]


class RagAtiniResponse:
    def __init__(self,
                 request: RagAtiniVectorizeRequest,
                 prominence: float,
                 overlap: bool,
                 min_chunk_size: int
                 ):
        self._request = request

        self.prominence = prominence
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size

        peaks, segments = self._build(prominence, overlap, min_chunk_size)
        self.peaks = peaks
        self.segments = segments

    def _segment(self, first_char, last_char):
        return RagAtiniTextSegment(
            text=self._request.document[first_char:last_char],
            text_coords=(first_char, last_char)
        )

    def _build(self, prominence: float, overlap: bool, min_chunk_size: int):
        min_chunk_size = max(min_chunk_size, 1)
        prominence = max(prominence, 0)

        peaks = detect_peaks(self._request.velocity, self._request.sigma, prominence)
        segments = []
        offset = 0
        last_peak = len(self._request.velocity) - 1
        for peak in np.append(peaks, last_peak):
            is_last_peak = peak == last_peak
            first_char, last_char = snap_to_boundary(
                boundaries=self._request.boundaries,
                segment_start=self._request.token_to_char[offset],
                segment_end=self._request.token_to_char[peak],
                overlap=overlap
            )

            if last_char - first_char < min_chunk_size:
                if is_last_peak and first_char != last_char:
                    if segments:
                        segments[-1] = self._segment(segments[-1].text_coords[0], last_char)
                    else:
                        segments.append(self._segment(first_char, last_char))
            else:
                segments.append(self._segment(first_char, last_char))
                offset = peak

        return peaks, segments

    def to(self, prominence: float = None, overlap: bool = None, min_chunk_size: int = None):
        prominence = self.prominence if prominence is None else prominence
        overlap = self.overlap if overlap is None else overlap
        min_chunk_size = self.min_chunk_size if min_chunk_size is None else min_chunk_size

        return RagAtiniResponse(
            request=self._request,
            prominence=prominence,
            overlap=overlap,
            min_chunk_size=min_chunk_size
        )


class RagAtini:
    def __init__(self,
                 vectorizer_model: str = "nomic-ai/modernbert-embed-base",
                 boundary_model: str = "mirth/chonky_modernbert_base_1",
                 doc_prefix="search_document: ",
                 boundary_radius=512,
                 device: str = 'cuda',
                 max_chunk_length=None,
                 trust_remote_code: bool = False):
        self.device = device if device.startswith('cuda') and torch.cuda.is_available() else 'cpu'

        self.tokenizer = AutoTokenizer.from_pretrained(vectorizer_model, trust_remote_code=trust_remote_code)
        self.max_context_window = max_chunk_length if max_chunk_length else getattr(self.tokenizer, "model_max_length",
                                                                                    8192)
        assert self.max_context_window, "Failed to resolve max_context_window"

        self.model = AutoModel.from_pretrained(vectorizer_model, trust_remote_code=trust_remote_code).to(self.device)
        self.model.eval()

        self.tokenizer.model_max_length = int(1e9)
        self.pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        self.cls_id = self.tokenizer.cls_token_id if self.tokenizer.cls_token_id is not None else self.tokenizer.bos_token_id
        self.sep_id = self.tokenizer.sep_token_id if self.tokenizer.sep_token_id is not None else self.tokenizer.eos_token_id

        self.prefix_ids = (self.tokenizer(doc_prefix, add_special_tokens=False)["input_ids"]
                           if doc_prefix else [])
        self.prefix_len = len(self.prefix_ids)

        self.default_stride = int(self.max_context_window * 0.25)
        self.sigma = max(10, self.max_context_window // 100)

        self.chonky = self.init_chonky(boundary_model)
        self.boundary_radius = boundary_radius

    def init_chonky(self, model_name):
        chonky_tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=1024)
        chonky_model = AutoModelForTokenClassification.from_pretrained(
            model_name, num_labels=2, id2label={0: "O", 1: "separator"},
            label2id={"O": 0, "separator": 1}
        )
        chonky_model.eval()
        return pipeline(
            task="ner",
            model=chonky_model,
            tokenizer=chonky_tokenizer,
            aggregation_strategy="simple",
            device=self.device,
            batch_size=8
        )

    def find_boundaries(self, texts, min_score=0.5):
        if isinstance(texts, str):
            texts = [texts]
        results = self.chonky(texts)  # one batched call; list-per-input
        if texts and isinstance(results[0], dict):  # single input -> pipeline returns a flat list
            results = [results]
        return [[(e["end"], e["score"]) for e in r if e["score"] > min_score] for r in results]

    def tokenize(self, text: str):
        enc = self.tokenizer(text, add_special_tokens=False, return_tensors="pt",
                             return_offsets_mapping=True)
        tokens = enc["input_ids"].to(self.device)
        offsets = enc["offset_mapping"][0].tolist()
        return tokens, offsets

    def get_token_offsets(self, document: str, offsets):
        token_to_char = [s for s, _ in offsets]
        token_to_char.append(len(document))
        return token_to_char

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
                pad_tensor = torch.full(
                    (self.max_context_window - chunk_ids.size(0),), self.pad_id, dtype=torch.long, device=self.device
                )
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

    def merge_segment_boundaries(self, segments_boundaries, segment_offsets, min_distance=10):
        refined = []
        for seps, base in zip(segments_boundaries, segment_offsets):
            refined.extend((base + char, conf) for char, conf in seps)
        refined = sorted(set(refined))

        deduped = {}
        for char, conf in refined:
            deduped[char] = conf

        merged = []
        for pos, conf in deduped.items():
            if merged and pos - merged[-1][0] < min_distance:
                merged[-1] = max(merged[-1], (pos, conf), key=lambda b: b[1])
            else:
                merged.append((pos, conf))

        return [pos for pos, _ in merged]

    def peak_adjacent_boundaries(self, peak_chars, recoded_text, radius=512, min_distance=16):
        doc_len = len(recoded_text)

        text_segments = []
        segment_offsets = []
        for peak in peak_chars:
            from_char = max(peak - radius, 0)
            to_char = min(peak + radius, doc_len)
            text_segments.append(recoded_text[from_char:to_char])
            segment_offsets.append(from_char)

        segments_boundaries = self.find_boundaries(text_segments)
        merged_boundaries = self.merge_segment_boundaries(segments_boundaries, segment_offsets, min_distance)

        return sorted([0] + merged_boundaries + [doc_len])

    def vectorize(self,
                  document: str,
                  internal_batch: int = 1,
                  stride: int = None,
                  f_sig: float = 1.0,
                  prominence: float = 0.5,
                  overlap: bool = False,
                  min_chunk_size: int = 100,
                  min_boundary_distance: int = 16,
                  boundary_radius: int = None
                  ):
        boundary_radius = boundary_radius if boundary_radius else self.boundary_radius
        stride = stride if stride else self.default_stride
        sigma = int(self.sigma * f_sig)

        tokens, offsets = self.tokenize(document)
        tokens = tokens.squeeze(0)
        tokens_len = tokens.size(0)

        if tokens_len == 0:
            raise ValueError("Input document resulted in zero tokens. Cannot process empty documents.")

        token_to_char = self.get_token_offsets(document, offsets)
        chunks = self.chunk_tokens(tokens, stride)
        chunk_vectors = self.process_chunks(chunks, internal_batch)

        window_size = chunk_vectors.size(1) - self.prefix_len - 2
        num_chunks = chunk_vectors.size(0)

        chunk_masks = self.generate_chunk_masks(num_chunks, window_size, tokens_len, stride)
        meshed_vectors = self.mesh_vectors(chunk_vectors, chunk_masks, tokens_len, stride)
        smoothed_vectors = self.apply_gaussian(meshed_vectors, sigma)
        semantic_velocity = self.calculate_velocity(smoothed_vectors)

        all_semantic_peaks = detect_peaks(semantic_velocity, sigma, 0)
        all_peak_chars = [token_to_char[p] for p in all_semantic_peaks]
        boundaries = self.peak_adjacent_boundaries(all_peak_chars, document, boundary_radius, min_boundary_distance)

        request = RagAtiniVectorizeRequest(
            sigma=sigma,
            document=document,
            boundaries=boundaries,
            velocity=semantic_velocity,
            vectors=smoothed_vectors,
            token_to_char=token_to_char
        )
        res = RagAtiniResponse(request=request, prominence=prominence, overlap=overlap, min_chunk_size=min_chunk_size)
        return res
