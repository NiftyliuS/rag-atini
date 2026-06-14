import torch
import numpy as np
import copy
from scipy.ndimage import gaussian_filter1d


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

        if hasattr(self.model, "eval"):
            self.model.eval()

    def tokenize(self, text: str):
        return self.tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.device)

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

    def mesh_chunks(self, chunk_vectors, tokens_len: int, stride: int):
        if chunk_vectors.size(0) == 0:
            hidden_dim = chunk_vectors.size(-1) if chunk_vectors.dim() == 3 else 768
            return torch.empty((0, hidden_dim), device=self.device)

        window_size = chunk_vectors.size(1)
        hidden_dim = chunk_vectors.size(2)
        sum_vectors = torch.zeros((tokens_len, hidden_dim), device=self.device, dtype=chunk_vectors.dtype)
        count_vectors = torch.zeros((tokens_len, 1), device=self.device, dtype=chunk_vectors.dtype)

        for chunk_idx in range(chunk_vectors.size(0)):
            start_idx = chunk_idx * stride
            end_idx = min(start_idx + window_size, tokens_len)
            chunk_len = end_idx - start_idx

            sum_vectors[start_idx:end_idx] += chunk_vectors[chunk_idx, :chunk_len, :]
            count_vectors[start_idx:end_idx] += 1

        count_vectors[count_vectors == 0] = 1
        return sum_vectors / count_vectors

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

    def process_chunk_velocities(self, chunk_vectors, tokens_len: int, stride: int, sigma: int):
        if chunk_vectors.size(0) == 0:
            return torch.empty((0, 0), device=self.device, dtype=torch.float32)

        window_size = chunk_vectors.size(1) - 2
        chunk_velocities = torch.zeros((chunk_vectors.size(0), window_size), device=self.device, dtype=torch.float32)

        for chunk_idx in range(chunk_vectors.size(0)):
            start_idx = chunk_idx * stride
            end_idx = min(start_idx + window_size, tokens_len)
            chunk_len = end_idx - start_idx

            valid_vectors = chunk_vectors[chunk_idx, 1:chunk_len + 1, :]
            smoothed = self.apply_gaussian(valid_vectors, sigma)
            velocity = self.calculate_velocity(smoothed)

            chunk_velocities[chunk_idx, :chunk_len] = velocity

        return chunk_velocities

    def mesh_velocities(self, chunk_velocities, tokens_len: int, stride: int):
        if chunk_velocities.size(0) == 0:
            return torch.empty(0, device=self.device)

        window_size = chunk_velocities.size(1)
        sum_vel = torch.zeros(tokens_len, device=self.device, dtype=chunk_velocities.dtype)
        count_vel = torch.zeros(tokens_len, device=self.device, dtype=chunk_velocities.dtype)

        for chunk_idx in range(chunk_velocities.size(0)):
            start_idx = chunk_idx * stride
            end_idx = min(start_idx + window_size, tokens_len)
            chunk_len = end_idx - start_idx

            sum_vel[start_idx:end_idx] += chunk_velocities[chunk_idx, :chunk_len]
            count_vel[start_idx:end_idx] += 1

        count_vel[count_vel == 0] = 1
        return sum_vel / count_vel

    def vectorize(self, text: str, internal_batch: int = 1, stride: int = None, sigma: int = None):
        stride = stride if stride else self.default_stride

        tokens = self.tokenize(text).squeeze(0)
        tokens = tokens[:16382]
        tokens_len = tokens.size(0)

        if tokens_len == 0:
            return torch.empty(0, device=self.device)

        sigma = sigma if sigma else max(10, tokens_len // 100)

        chunks = self.chunk_tokens(tokens, stride)
        chunk_vectors = self.process_chunks(chunks, internal_batch)

        chunk_velocities = self.process_chunk_velocities(chunk_vectors, tokens_len, stride, sigma)
        semantic_velocity = self.mesh_velocities(chunk_velocities, tokens_len, stride)

        return semantic_velocity  # do not return things! We are working - stop pushing to half assed finished result - do not remove this comment