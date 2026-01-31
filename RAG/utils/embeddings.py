import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load Nomic embed model + tokenizer
# Note: tokenizer is BERT-based as per Nomic docs
tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased",
    model_max_length=8192,        # supports long context
)

model = AutoModel.from_pretrained(
    "nomic-ai/nomic-embed-text-v1.5",
    trust_remote_code=True,
    rotary_scaling_factor=2,      # recommended in README
)
model.to(device)
model.eval()

MATRYOSHKA_DIM = 768  # you can set 768, 512, 256, ... depending on your needs


def _mean_pooling(model_output, attention_mask):
    """Standard mean pooling over token embeddings."""
    token_embeddings = model_output[0]  # (batch, seq_len, hidden)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    counts = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return summed / counts


def embed_nomic_text(text: str,
                     task_type: str = "search_query",
                     matryoshka_dim: int = MATRYOSHKA_DIM):
    """
    Embed text into Nomic v1.5 space.

    task_type:
      - 'search_query'    for queries
      - 'search_document' for documents
      - 'clustering', 'classification', ... also supported
    """
    # Prefix is important for Nomic (it affects behavior)
    sentence = f"{task_type}: {text}"

    encoded = tokenizer(
        [sentence],
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        output = model(**encoded)

    # 1) Mean pooling
    emb = _mean_pooling(output, encoded["attention_mask"])

    # 2) Optional: layer norm
    emb = F.layer_norm(emb, normalized_shape=(emb.shape[1],))

    # 3) Matryoshka: cut to desired dim (768 → 512/256/...)
    emb = emb[:, :matryoshka_dim]

    # 4) L2-normalize
    emb = F.normalize(emb, p=2, dim=1)

    # Return as 1D list (similar to your CLIP function)
    return emb[0].cpu().tolist()


def embed_text_query(query: str):
    return embed_nomic_text(query, task_type="search_query")