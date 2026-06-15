import config
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel

from RagAtini import RagAtini

DEVICE = 'cuda'

model_name = "BAAI/bge-m3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(DEVICE)
ragAtini = RagAtini(model, tokenizer)


def print_results(peaks, tokens, tokenizer):
    print("\n--- Segment Extraction ---")
    start_idx = 0
    tokens_len = len(tokens)

    for i, peak_idx in enumerate(peaks):
        segment_tokens = tokens[start_idx:peak_idx]
        segment_text = tokenizer.decode(segment_tokens, skip_special_tokens=True)

        print(f"\n[Segment {i + 1} | Tokens {start_idx}-{peak_idx} | Len: {len(segment_tokens)}]")
        print(segment_text[:500] + ("..." if len(segment_text) > 500 else ""))
        start_idx = peak_idx

    segment_tokens = tokens[start_idx:tokens_len]
    if len(segment_tokens) > 0:
        segment_text = tokenizer.decode(segment_tokens, skip_special_tokens=True)
        print(f"\n[Segment {len(peaks) + 1} | Tokens {start_idx}-{tokens_len} | Len: {len(segment_tokens)}]")
        print(segment_text[:500] + ("..." if len(segment_text) > 500 else ""))


def ensemble_segmentation(document_str: str):
    response = ragAtini.vectorize(document_str, prominence=0.5)

    vel_np = response.velocity.cpu().numpy()
    tokens_len = len(vel_np)

    if tokens_len == 0:
        return

    peaks = response.peaks[response.peaks < tokens_len]

    plt.figure(figsize=(16, 7))

    plt.plot(vel_np, color='orange', linewidth=1.5, linestyle='-', alpha=0.9,
             label='RagAtini Meshed Velocity')

    if len(peaks) > 0:
        plt.plot(peaks, vel_np[peaks], 'ro', markerfacecolor='none', markersize=10,
                 markeredgewidth=2, label='Detected Peaks')

    plt.title('RagAtini Final Vector Mesh Velocity Profile')
    plt.xlabel('Absolute Token Index')
    plt.ylabel('Euclidean Derivative (Velocity)')
    plt.xlim(0, tokens_len)
    plt.ylim(0, 0.035)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print_results(peaks, response.tokens[:tokens_len], ragAtini.tokenizer)

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
    # filename = "Frankenstein_or_the_modern_prometheus.txt"
    filename = "sample.txt"
    print(f"Reading file: {filename}")
    context = read_file(filename)

    if context:
        print(f"Context character length: {len(context)}")
        ensemble_segmentation(context)