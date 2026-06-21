import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import numpy as np
from RagAtini import RagAtini

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

model_name = "BAAI/bge-m3"
model_name = "nomic-ai/modernbert-embed-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(DEVICE)
ragAtini = RagAtini(model, tokenizer)

EVALUATION_QUESTIONS = [
    ("What weight-norm intervention on decoder layers accelerates grokking, and across how many algebraic tasks?", 0),
    ("Which four established frameworks does the analysis connect weight norm clipping to?", 1),
    ("When did Power et al. first document grokking and on what kind of datasets?", 2),
    ("What speedup does clipping achieve on modular multiplication at 2-layer versus 8-layer?", 3),
    ("Per the contributions, what does the only hyperparameter max_norm correlate with?", 4),
    ("In equation (1), what does each weight row get multiplied by to project onto the l2 ball of radius c?", 5),
    ("In project_to_sphere, the parameter is multiplied by c and divided by what?", 6),
    ("For edge initialization, which three components are projected to norm c before training?", 7),
    ("What is the convergence criterion and train/validation split for the modular multiplication setup?", 8),
    ("How many steps does the AdamW baseline require to generalize on 2-layer modular multiplication?", 9),
    ("In the 2-layer seed stability table, what is the P95 step count for SignSGD with no init_norm?", 10),
    ("At init_norm=2.0, which optimizer achieves the fastest median and what is its IQR?", 11),
    ("Under the 25/75 data-scarce split, what happens to Adam+Clip and SignSGD?", 12),
    ("In the 8-layer AdamW baseline, what value do individual seeds' validation loss spike to?", 13),
    ("What IQR reduction does edge initialization give for Lion, Adam, and SignSGD?", 14),
    ("In the 8-layer table, what is the median step count for AdamW with no clipping?", 15),
    ("On the 9.8k-parameter model, what learning rate produces unstable grokking around 3,000 steps?", 16),
    ("In Table 3, what is SignSGD's median step count at max_norm 1.0?", 17),
    ("Why is max_norm below 2.0 too tight for modular multiplication?", 18),
    ("In Table 4, what is the add-p97 median step count at max_norm 1.75?", 19),
    ("Which operations favor max_norm 1.5-1.75 because they require modular inversion?", 20),
    ("In Table 5, what is the all-mod median step count at max_norm 1.25?", 21),
    ("How is the all-mod batch size of 2048 derived?", 22),
    ("In the cross-task speedups table, what are the AdamW steps and speedup for mul-p97?", 23),
    ("What does Lion without clipping do on the S5 task?", 24),
    ("What max_norm and speedup are reported for the S5 permutation panel?", 25),
    ("In the Lion learning-rate stability setup, how many LRs and seeds per LR are used?", 26),
    ("In section 4.1, what does the grokking timescale collapse to when c approaches wc?", 27),
    ("In the homogeneous motivating calculation, what is alpha^L at L=2 versus L=8?", 28),
    ("What l-infinity constrained optimization does Lion with weight decay solve, per Chen et al. [2024]?", 29),
    ("In the Grokfast-to-sign-based derivation, what does Adam's second-moment normalization drive updates toward?",
     30),
    ("In the Softmax Collapse explanation, what floating-point overflow causes absorption errors?", 31),
    ("Who identified Fourier multiplication circuits and three training phases?", 32),
    ("What does nGPT normalize all vectors to, and for what LM speedup?", 33),
    ("Which optimizer did Tveit et al. [2025] show accelerates grokking via spectral norm control?", 34),
    ("What is named as the most direct future-work extension for norm clipping?", 35),
    ("What is the title of the Chen et al. [2023] paper in the references?", 36),
    ("What is the arXiv number for Sfyraki & Wang's 'Lions and muons' paper?", 37),
]


def embed_query(query: str, tokenizer, model, device) -> torch.Tensor:
    inputs = tokenizer(query, return_tensors="pt",
                       truncation=True, max_length=8192).to(device)
    with torch.no_grad():
        outputs = model(**inputs).last_hidden_state[0]  # (T, H)
        return outputs.mean(dim=0)


def compare(query: str, peak_vec: torch.Tensor, bound_vec: torch.Tensor, tokenizer, model, device):
    q_vec = embed_query("search_query: "+query, tokenizer, model, device)

    q_vec = F.normalize(q_vec, p=2, dim=0)
    p_vec = F.normalize(peak_vec, p=2, dim=0)
    b_vec = F.normalize(bound_vec, p=2, dim=0)

    p_score = F.cosine_similarity(q_vec.unsqueeze(0), p_vec.unsqueeze(0)).item()
    b_score = F.cosine_similarity(q_vec.unsqueeze(0), b_vec.unsqueeze(0)).item()

    return p_score, b_score


def evaluate_retrieval(response, questions, tokenizer, model, device):
    if not response.segments:
        print("No segments generated. Aborting evaluation.")
        return

    n_seg = len(response.segments)
    print(f"\n{'=' * 80}\nEXTRACTED SEGMENTS\n{'=' * 80}")
    for i, seg in enumerate(response.segments):
        preview = seg.text.replace('\n', ' ')
        preview = preview if len(preview) <= 120 else preview[:300] + "..."
        print(f"Segment {i:02d} | Coords: {seg.text_coords} | Text: {preview}")

    p_vecs = F.normalize(torch.stack([s.vector for s in response.segments]), p=2, dim=1)
    b_vecs = F.normalize(torch.stack([s.bound_vector for s in response.segments]), p=2, dim=1)
    t_vecs = F.normalize(torch.stack([
        embed_query("search_document: " + s.text, tokenizer, model, device) if s.text
        else torch.zeros(p_vecs.size(1), device=device)
        for s in response.segments
    ]), p=2, dim=1)

    def get_scores(query: str):
        q = F.normalize(embed_query("search_query: "+query, tokenizer, model, device), p=2, dim=0).unsqueeze(0)
        return (F.cosine_similarity(q, p_vecs).cpu().numpy(),
                F.cosine_similarity(q, b_vecs).cpu().numpy(),
                F.cosine_similarity(q, t_vecs).cpu().numpy())

    def fmt(t):
        c = str(t).replace('\n', ' ').strip()
        return c if len(c) <= 100 else c[:97] + "..."

    print(f"\n{'=' * 80}\nTop-1 retrieval vs target segment\n{'=' * 80}")
    peak_hits = bound_hits = text_hits = total = 0

    for question, target in questions:
        if target >= n_seg:  # target chunk doesn't exist in this run
            print(f"\nQ: {question}\n  SKIP (target seg {target} >= {n_seg} segments)")
            continue
        total += 1
        p, b, t = get_scores(question)
        bp, bb, bt = int(np.argmax(p)), int(np.argmax(b)), int(np.argmax(t))
        ph, bh, th = bp == target, bb == target, bt == target
        peak_hits += ph;
        bound_hits += bh;
        text_hits += th

        print(f"\nQ: {question}\n  target=seg{target}")
        print(
            f"  PEAK  hit={ph!s:5} score@target={p[target]:.4f}  top1=seg{bp}[{p[bp]:.4f}] {fmt(response.segments[bp].text)}")
        print(
            f"  BOUND hit={bh!s:5} score@target={b[target]:.4f}  top1=seg{bb}[{b[bb]:.4f}] {fmt(response.segments[bb].text)}")
        print(
            f"  TEXT  hit={th!s:5} score@target={t[target]:.4f}  top1=seg{bt}[{t[bt]:.4f}] {fmt(response.segments[bt].text)}")

    print(f"\n{'=' * 80}")
    print(f"TOP-1 ACCURACY over {total} scored questions  ->  "
          f"PEAK: {peak_hits}/{total} ({peak_hits / total:.0%}) | "
          f"BOUND: {bound_hits}/{total} ({bound_hits / total:.0%}) | "
          f"TEXT: {text_hits}/{total} ({text_hits / total:.0%})")
    print(f"{'=' * 80}\n")


def read_file(file_path: str):
    print(f"Loading {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return None


if __name__ == "__main__":
    filename = "sample.txt"
    context = read_file(filename)

    if context:
        print(f"Context character length: {len(context)}")
        response = ragAtini.vectorize(context, prominence=0.5, overlap=0)

        tokens_len = len(response.token_ids)
        if tokens_len > 0:
            vel_forward = response.velocity.cpu().numpy()
            valid_peaks = response.peaks[response.peaks < tokens_len]

            plt.figure(figsize=(16, 7))
            plt.plot(vel_forward, color='red', linewidth=2.0, linestyle='-', alpha=0.9,
                     label='Smoothed Semantic Velocity')

            if len(valid_peaks) > 0:
                plt.plot(valid_peaks, vel_forward[valid_peaks], 'ro', markerfacecolor='none', markersize=10,
                         markeredgewidth=2, label='Smoothed Peaks')

            plt.title('RagAtini Semantic Peak Alignment')
            plt.xlabel('Absolute Token Index')
            plt.ylabel('Euclidean Derivative (Smoothed Velocity)')
            plt.xlim(0, tokens_len)
            plt.ylim(0, np.max(vel_forward) * 1.05)
            plt.legend(loc='upper right')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

        evaluate_retrieval(response, EVALUATION_QUESTIONS, tokenizer, model, DEVICE)
