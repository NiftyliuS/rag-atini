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

def ensemble_segmentation(document_str: str):
    velocities = ragAtini.vectorize(document_str)

    vel_np = velocities.cpu().numpy()
    tokens_len = len(vel_np)

    if tokens_len == 0:
        return

    plt.figure(figsize=(16, 7))

    plt.plot(vel_np, color='blue', linewidth=1.5, linestyle='-', alpha=1.0,
             label='RagAtini Meshed & Smoothed Velocity')

    plt.title('RagAtini Segment Velocity Profile')
    plt.xlabel('Absolute Token Index')
    plt.ylabel('Euclidean Derivative (Velocity)')
    plt.xlim(0, tokens_len)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

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
    print(f"Context character length: {len(context)}")

    ensemble_segmentation(context)
