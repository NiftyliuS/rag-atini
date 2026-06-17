import config
import numpy as np
import matplotlib.pyplot as plt
import torch
import re
import bisect
from transformers import AutoTokenizer, AutoModel

from RagAtini import RagAtini

DEVICE = 'cuda'

model_name = "BAAI/bge-m3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(DEVICE)
ragAtini = RagAtini(model, tokenizer)


def print_results(peaks, tokens, tokenizer):
    print("\n--- Segment Extraction ---")
    start_idx = 1
    tokens_len = len(tokens)

    for i, peak_idx in enumerate(peaks):
        segment_tokens = tokens[start_idx-1:peak_idx]
        segment_text = tokenizer.decode(segment_tokens, skip_special_tokens=True)

        print(f"\n[Segment {i + 1} | Tokens {start_idx}-{peak_idx} | Len: {len(segment_tokens)}]")
        print(segment_text[:500] + ("..." if len(segment_text) > 500 else ""))
        start_idx = peak_idx

    segment_tokens = tokens[start_idx:tokens_len]
    if len(segment_tokens) > 0:
        segment_text = tokenizer.decode(segment_tokens, skip_special_tokens=True)
        print(f"\n[Segment {len(peaks) + 1} | Tokens {start_idx}-{tokens_len} | Len: {len(segment_tokens)}]")
        print(segment_text[:500] + ("..." if len(segment_text) > 500 else ""))


def align_peaks_to_boundaries(peaks, tokens, ragAtini_instance):
    offsets,rev, text = ragAtini_instance.get_token_offsets(tokens)

    boundaries = [m.start() for m in re.finditer(r'(?<=[.!?])\s+|\n+', text)]
    if not boundaries:
        return peaks

    aligned_token_indices = []
    for peak in peaks:
        if peak >= len(offsets):
            continue

        char_idx = offsets[peak]
        b_idx = bisect.bisect_left(boundaries, char_idx)

        if b_idx == 0:
            closest_boundary = boundaries[0]
        elif b_idx == len(boundaries):
            closest_boundary = boundaries[-1]
        else:
            dist_right = boundaries[b_idx] - char_idx
            dist_left = char_idx - boundaries[b_idx - 1]
            closest_boundary = boundaries[b_idx - 1] if dist_left < dist_right else boundaries[b_idx]

        aligned_token_idx = bisect.bisect_left(offsets, closest_boundary)

        if not aligned_token_indices or aligned_token_idx > aligned_token_indices[-1]:
            if aligned_token_idx < len(tokens):
                aligned_token_indices.append(aligned_token_idx)

    return np.array(aligned_token_indices)


def ensemble_segmentation(document_str: str):
    response = ragAtini.vectorize(document_str, prominence=0.5)

    tokens_len = len(response.token_ids)
    if tokens_len == 0:
        return

    vel_forward = response.velocity.cpu().numpy()
    valid_peaks = response.peaks[response.peaks < tokens_len]

    aligned_boundaries = align_peaks_to_boundaries(valid_peaks, response.token_ids[:tokens_len], ragAtini)

    plt.figure(figsize=(16, 7))

    plt.plot(vel_forward, color='red', linewidth=1.5, linestyle='-', alpha=0.9,
             label='RagAtini Meshed Velocity')

    if len(valid_peaks) > 0:
        plt.plot(valid_peaks, vel_forward[valid_peaks], 'ro', markerfacecolor='none', markersize=10,
                 markeredgewidth=2, label='Raw Forward Peaks')

    if len(aligned_boundaries) > 0:
        plt.vlines(aligned_boundaries, ymin=0, ymax=np.max(vel_forward), colors='green', linestyles='solid',
                   linewidth=2, alpha=0.8, label='Aligned Sentence Boundaries')

    plt.title('RagAtini Semantic Peak & Boundary Alignment')
    plt.xlabel('Absolute Token Index')
    plt.ylabel('Euclidean Derivative (Velocity)')
    plt.xlim(0, tokens_len)
    plt.ylim(0, 0.035)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("\n=== SLICING AT ALIGNED BOUNDARIES ===")
    print_results(aligned_boundaries, response.token_ids[:tokens_len], ragAtini.tokenizer)

    return


def read_file(file_path: str):
    print(f"Loading {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Warning: {file_path} not found. Ensure the book file is present.")
        return


if __name__ == "__main__":
    filename = "sample.txt"
    print(f"Reading file: {filename}")
    context = read_file(filename)

    if context:
        print(f"Context character length: {len(context)}")
        ensemble_segmentation(context)