import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import numpy as np
from RagAtini import RagAtini

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

model_name = "BAAI/bge-m3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(DEVICE)
ragAtini = RagAtini(model, tokenizer)

EVALUATION_QUESTIONS = [
    "Who are the independent researchers that authored Clip to Grok?",
    "In Figure 1's single-seed demonstration on the 2-layer architecture, what is the exact beta2 value used for Lion?",
    "In the listed contributions, what specific task correlates with the max_norm hyperparameter?",
    "In the Method section for weight norm clipping, what mathematical formula is used to project every weight row onto the l2 ball of radius c?",
    "In Algorithm 1's project_to_sphere function, what variable is divided by the clamped norm?",
    "In the asymmetric design rationale table, what layers are initialized to c but not clipped for shallow networks?",
    "In the experimental setup, what is the exact convergence criterion used for the modular multiplication task?",
    "In Figure 2's single-run training dynamics, exactly how many steps does it take the AdamW baseline to reach 95% validation?",
    "When decomposing the 66x speedup, what is the exact 2-layer median step count for Adam+Clip?",
    "In the 2-layer seed stability table, what is the 95th percentile (P95) step count for SignSGD with no init_norm?",
    "In Figure 3's multi-seed accuracy caption, what are the exact learning rates listed for Adam, Lion, and SignSGD?",
    "In the data-scarce regime, what specifically provides the directional consistency needed when fewer training examples are available per step?",
    "In the 8-layer edge initialization experiments, what specific depth-dependent learning rate is assigned to Lion?",
    "In Figure 5's Softmax Collapse illustration, what is the exact value that individual seeds spike to for validation loss?",
    "According to the main result for edge initialization, what is the exact IQR reduction percentage for SignSGD?",
    "In the 8-layer architecture table, what is the exact median step count for the AdamW baseline without clipping?",
    "At the lower extreme scale robustness test, what exact learning rate produces unstable grokking around 3,000 steps for the 9.8k parameter model?",
    "In the max_norm selection table for modular multiplication, what is the median step count for SignSGD at max_norm 1.0?",
    "According to the U-shaped trade-off visualization text, which optimizer degrades gracefully but slows 4x at a max_norm of 2.5?",
    "In the single modular arithmetic tasks table, what is the exact median step count for the add-p97 task at a max_norm of 1.75?",
    "Which specific operations favor a max_norm of 1.5 to 1.75 because they require modular inversion?",
    "In the multi-task and non-abelian tasks table, what is the median step count for all-mod at max_norm 1.25?",
    "For the all-mod task's batch size adjustment, what is the resulting concatenated training set batch size calculation?",
    "In the cross-task speedups table, what is the exact AdamW step count for the all-mod task?",
    "What specific failure mode occurs when Lion without clipping is applied to the S5 task?",
    "In the Lion Learning Rate Stability setup, how many seeds are used per learning rate?",
    "What loose bound value is suggested for optionally clipping the output head to suppress post-convergence oscillation?",
    "If the clipping threshold c is approximately equal to wc, what does the natural log of w0 divided by wc approach?",
    "In the motivating calculation for the homogeneous case, what is the approximate value of alpha to the power of L when L equals 8?",
    "In residual networks, what exact mathematical perturbation formula do skip connections convert the exponential alpha L risk into?",
    "Under the Grokfast EMA filter equations, what does v-hat sub t approximately equal when driving updates toward plus or minus 1?",
    "According to the Softmax Collapse section, what specific floating-point precision format overflows and causes absorption errors?",
    "Which authors provided mechanistic interpretability by identifying Fourier multiplication circuits and three training phases?",
    "What specific gradient modification method did Prieto et al. [2025] use to address Softmax Collapse?",
    "In the comparison with Karras et al. [2024], what specific target norm value does the EDM2 approach force each weight row to?",
    "Which paper showed that the Muon optimizer accelerates grokking via spectral norm control?",
    "What specific representation-space normalization method is suggested for combination with norm clipping in language model pretraining?",
    "What exact version of PyTorch and CUDA were used for all experiments?",
    "In the references, who are the authors of the 2024 paper 'Deep grokking: Would deep neural networks generalize better?'",
    "In the references, what is the exact title of the 2024 paper by Lyu, Jin, Li, Du, Lee, and Hu?",
    "In the references, what is the arXiv preprint number for the paper 'Lions and muons: Optimization via stochastic FrankWolfe'?",
    "In the references, what is the exact title of the 2025 paper by Xu et al. published in ICLR?"
]


def embed_query(query: str, tokenizer, model, device) -> torch.Tensor:
    inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=8192).to(device)
    with torch.no_grad():
        outputs = model(**inputs).last_hidden_state
        cls_vector = outputs[0, 0, :]

        return  outputs[0, 1:-1, :].mean(dim=0)


def compare(query: str, peak_vec: torch.Tensor, bound_vec: torch.Tensor, tokenizer, model, device):
    q_vec = embed_query(query, tokenizer, model, device)

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

    print(f"\n{'=' * 80}")
    print("EXTRACTED SEGMENTS")
    print(f"{'=' * 80}")
    for i, seg in enumerate(response.segments):
        preview = seg.text.replace('\n', ' ')
        preview = preview if len(preview) <= 120 else preview[:300] + "..."
        print(f"Segment {i:02d} | Tokens: {seg.text_coords} | Text: {preview}")

    eval_count = min(len(questions), len(response.segments))

    print(f"\n{'=' * 80}\nEvaluating explicit targets vs actual winners...\n{'=' * 80}")

    p_vecs = F.normalize(torch.stack([s.vector for s in response.segments]), p=2, dim=1)
    b_vecs = F.normalize(torch.stack([s.bound_vector for s in response.segments]), p=2, dim=1)

    def get_scores(query: str):
        q_vec = embed_query(query, tokenizer, model, device)
        q_vec = F.normalize(q_vec, p=2, dim=0).unsqueeze(0)

        p_scores = F.cosine_similarity(q_vec, p_vecs).cpu().numpy()
        b_scores = F.cosine_similarity(q_vec, b_vecs).cpu().numpy()
        return p_scores, b_scores

    def format_text(text):
        cleaned = str(text).replace('\n', ' ').strip()
        return cleaned if len(cleaned) <= 100 else cleaned[:97] + "..."

    q_minus1 = "Why did the chicken cross the road?"
    p_scores, b_scores = get_scores(q_minus1)

    best_p_idx = int(np.argmax(p_scores))
    best_b_idx = int(np.argmax(b_scores))

    print(f"\nQ[-1]: {q_minus1}")
    print(f"PEAK : {p_scores[0]:.4f}")
    print(
        f"PEAK_WINNER: seg {best_p_idx}[{p_scores[best_p_idx]:.4f}]: {format_text(response.segments[best_p_idx].text)}")
    print(f"BOUND: {b_scores[0]:.4f}")
    print(
        f"BOUND_WINNER: seg {best_b_idx}[{b_scores[best_b_idx]:.4f}]: {format_text(response.segments[best_b_idx].text)}")

    peak_wins = 0
    bound_wins = 0

    for i in range(eval_count):
        question = questions[i]
        p_scores, b_scores = get_scores(question)

        best_p_idx = int(np.argmax(p_scores))
        best_b_idx = int(np.argmax(b_scores))

        exp_p = p_scores[i]
        exp_b = b_scores[i]

        diff = exp_p - exp_b

        if exp_p > exp_b:
            winner = "PEAK"
            peak_wins += 1
        elif exp_b > exp_p:
            winner = "BOUND"
            bound_wins += 1
        else:
            winner = "TIE"

        print(f"\nQ[{i}]: {question}")
        print(f"PEAK : {exp_p:.4f}")
        print(
            f"PEAK_WINNER: seg {best_p_idx}[{p_scores[best_p_idx]:.4f}]: {format_text(response.segments[best_p_idx].text)}")
        print(f"BOUND: {exp_b:.4f}")
        print(
            f"BOUND_WINNER: seg {best_b_idx}[{b_scores[best_b_idx]:.4f}]: {format_text(response.segments[best_b_idx].text)}")
        print(f"Winner: {winner} (Δ {abs(diff):.4f})")

    print(f"\n{'=' * 80}")
    print(f"DIRECT MATCH WINS -> PEAK: {peak_wins} | BOUND: {bound_wins}")
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