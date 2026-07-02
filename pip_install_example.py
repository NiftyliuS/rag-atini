########################
# pip install ragatini #
########################

from ragatini import RagAtini

ragAtini = RagAtini(
    vectorizer_model="nomic-ai/modernbert-embed-base",
    boundary_model="mirth/chonky_modernbert_base_1",
    doc_prefix="search_document: ",  # prefix classification required by nomic-ai modernbert
    device='cuda'
)


# multilanguage setup
# ragAtini = RagAtini(
#     vectorizer_model="nomic-ai/nomic-embed-text-v2-moe",
#     boundary_model="mirth/chonky_mmbert_small_multilingual_1",
#     doc_prefix="search_document: ",
#     device='cuda',
#     trust_remote_code=True,
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
        response_lg = ragAtini.vectorize(context, prominence=4.0, overlap=False)
        print_segments(response_lg)

        response_sm = response_lg.to(prominence=0.5)
        print_segments(response_sm)
