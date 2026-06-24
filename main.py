import torch
from transformers import AutoTokenizer, AutoModel
from RagAtini import RagAtini

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'



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

    model_name = "nomic-ai/modernbert-embed-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(DEVICE)
    ragAtini = RagAtini(model, tokenizer)

    print(f"Context character length: {len(context)}")
    response = ragAtini.vectorize(context, prominence=0.5, overlap=0)
