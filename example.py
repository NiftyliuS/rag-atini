import torch
from ragatini.RagAtini import RagAtini
from charts import peak_velocity_chart

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

ragAtini = RagAtini(
    vectorizer_model="nomic-ai/modernbert-embed-base",
    boundary_model="mirth/chonky_modernbert_base_1",
    doc_prefix="search_document: ",  # prefix classification required by nomic-ai modernbert
    device=DEVICE
)


# multilanguage setup
# ragAtini = RagAtini(
#     vectorizer_model="nomic-ai/nomic-embed-text-v2-moe",
#     boundary_model="mirth/chonky_mmbert_small_multilingual_1",
#     doc_prefix="search_document: ",
#     device=DEVICE
# )

def print_segments(response, full: bool = False):
    print(f"\n{'=' * 80}\nEXTRACTED SEGMENTS (prominence={response.prominence})\n{'=' * 80}")
    for i, seg in enumerate(response.segments):
        preview = seg.text.replace('\n', ' ')
        preview = preview if len(preview) <= 120 or full else preview[:300] + "..."
        print(f"Segment {i:02d} | Coords: {seg.text_coords} | Text: {preview}")


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
        response_lg = ragAtini.vectorize(context, prominence=2.0, overlap=False)
        print_segments(response_lg)

        response_sm = response_lg.to(prominence=0.5)
        print_segments(response_sm)

        # velocity is calculated once - not per prominence
        velocity = response_lg._request.velocity.cpu().numpy()  # 1D semantic velocity measurement

        peaks_sm = response_sm.peaks  # detected peaks in semantic velocity
        peaks_lg = response_lg.peaks  # detected peaks in semantic velocity

        # optional setup to view the smoothed vectors directly
        # vectors = response_sm._request.vectors.cpu().numpy()  # gaussian smoothed vectors
        # umap_chart_2d(vectors, peaks_sm)
        # umap_chart_2d(vectors)

        # visual representation of semantic velocity
        peak_velocity_chart(velocity, peaks_sm)
        peak_velocity_chart(velocity, peaks_lg)
