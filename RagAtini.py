import re
import bisect
import torch
import numpy as np
import copy
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def find_boundaries(text: str):
    boundaries = [m.start() for m in re.finditer(r'(?<=[.!?])\s+|\n+', text)]
    return boundaries


class RagSegment:
    def __init__(self,
                 vector: torch.Tensor,
                 bound_vector: torch.Tensor,
                 text: str = None,
                 text_coords: tuple[int, int] = None
                 ):
        self.vector = vector
        self.bound_vector = bound_vector
        self.text = text
        self.text_coords = text_coords


class RagAtiniResponse:
    def __init__(self,
                 velocity: torch.Tensor,
                 peaks: np.ndarray,
                 token_ids: torch.Tensor,
                 token_vectors: torch.Tensor,
                 recoded_text: str,
                 segments: list[RagSegment]
                 ):
        self.recoded_text = recoded_text
        self.velocity = velocity
        self.peaks = peaks
        self.token_ids = token_ids
        self.token_vectors = token_vectors
        self.segments = segments


class RagAtini:
    def __init__(self, model, tokenizer, model_max_length=None):
        self.model = model
        self.device = next(self.model.parameters()).device if hasattr(self.model, "parameters") else "cpu"

        self.max_context_window = model_max_length if model_max_length else getattr(tokenizer, "model_max_length", 8192)
        assert self.max_context_window, "Failed to resolve max_context_window"

        self.tokenizer = copy.deepcopy(tokenizer)
        self.tokenizer.model_max_length = int(1e9)
        self.pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        self.default_stride = int(self.max_context_window * 0.25)
        self.sigma = max(10, self.max_context_window // 100)

        self.find_boundaries = find_boundaries
        if hasattr(self.model, "eval"):
            self.model.eval()

    def tokenize(self, text: str):
        return self.tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.device)

    def get_token_offsets(self, tokens: torch.Tensor):
        text = self.tokenizer.decode(tokens[0:1], skip_special_tokens=True)
        token_to_char = [0]
        char_to_token = [0] * len(text)

        for i in range(0, len(tokens) - 1):
            token_to_char.append(len(text))
            pair = self.tokenizer.decode([tokens[i:i + 1], tokens[i:i + 2]], skip_special_tokens=True)
            next_token = pair[1][len(pair[0]):]
            char_to_token.extend([i + 1] * len(next_token))
            text += next_token

        return token_to_char, char_to_token, text

    def chunk_tokens(self, tokens, stride):
        window_size = self.max_context_window - 2
        tokens_len = tokens.size(0)

        cls_id = self.tokenizer.cls_token_id if self.tokenizer.cls_token_id is not None else 0
        sep_id = self.tokenizer.sep_token_id if self.tokenizer.sep_token_id is not None else 0

        cls_tensor = torch.tensor([cls_id], dtype=torch.long, device=self.device)
        sep_tensor = torch.tensor([sep_id], dtype=torch.long, device=self.device)

        chunks = []
        for i in range(0, max(1, tokens_len), stride):
            chunk = tokens[i:i + window_size]

            chunk_with_special = torch.cat([cls_tensor, chunk, sep_tensor])

            if chunk_with_special.size(0) < self.max_context_window:
                pad_tensor = torch.full((self.max_context_window - chunk_with_special.size(0),), self.pad_id,
                                        dtype=torch.long, device=self.device)
                chunk_with_special = torch.cat([chunk_with_special, pad_tensor])

            chunks.append(chunk_with_special)

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

        vectors_np = vectors.cpu().numpy()
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
            valid_vectors = chunk_vectors[chunk_idx, 1:chunk_len + 1, :]

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
        mad = min(1e-6, np.median(np.abs(vel_np - median_vel)))
        min_prominence = mad * prominence

        peaks, _ = find_peaks(vel_np, distance=distance, prominence=min_prominence)

        return peaks

    def snap_to_boundary(self, boundaries, text, segment_start, segment_end):
        b_start = bisect.bisect_left(boundaries, segment_start)
        b_end = bisect.bisect_right(boundaries, segment_end)

        first_char = boundaries[b_start]
        last_char = boundaries[b_end]

        return first_char, last_char, text[first_char:last_char]

    def vectorize(self,
                  document: str,
                  internal_batch: int = 1,
                  stride: int = None,
                  sigma: int = None,
                  prominence: float = 4.0,
                  ):
        stride = stride if stride else self.default_stride
        sigma = sigma if sigma else self.sigma

        tokens = self.tokenize(document).squeeze(0)
        tokens_len = tokens.size(0)

        if tokens_len == 0:
            raise ValueError("Input document resulted in zero tokens. Cannot process empty documents.")

        chunks = self.chunk_tokens(tokens, stride)
        chunk_vectors = self.process_chunks(chunks, internal_batch)

        window_size = chunk_vectors.size(1) - 2
        num_chunks = chunk_vectors.size(0)

        chunk_masks = self.generate_chunk_masks(num_chunks, window_size, tokens_len, stride)
        meshed_vectors = self.mesh_vectors(chunk_vectors, chunk_masks, tokens_len, stride)
        smoothed_vectors = self.apply_gaussian(meshed_vectors, sigma)
        semantic_velocity = self.calculate_velocity(smoothed_vectors)
        semantic_peaks = self.detect_peaks(semantic_velocity, sigma, prominence)

        token_to_char, char_to_token, recoded_text = self.get_token_offsets(tokens)
        boundaries = sorted([0] + self.find_boundaries(recoded_text) + [len(recoded_text)-1])

        return RagAtiniResponse(
            velocity=semantic_velocity,
            peaks=semantic_peaks,
            token_ids=tokens,
            token_vectors=meshed_vectors,
            recoded_text=recoded_text,
            segments=[]
        )
