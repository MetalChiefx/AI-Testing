import torch
import torch.nn as nn
from torch.nn import functional as F
from pathlib import Path
import json


# --------------------------------------------------
# Model Folder Configuration
# --------------------------------------------------

MODEL_DIR = "model"

MODEL_FILE = f"{MODEL_DIR}/pytorch_model.bin"
VOCAB_FILE = f"{MODEL_DIR}/vocab.json"
CONFIG_FILE = f"{MODEL_DIR}/config.json"


# --------------------------------------------------
# Device Setup
# --------------------------------------------------

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


device = get_device()


# --------------------------------------------------
# Load Model Metadata
# --------------------------------------------------

def load_json(file_path):
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Could not find required file: {file_path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


config = load_json(CONFIG_FILE)
char_to_id = load_json(VOCAB_FILE)

id_to_char = {int(v): k for k, v in char_to_id.items()}

vocab_size = int(config["vocab_size"])
BLOCK_SIZE = int(config["block_size"])
EMBEDDING_DIM = int(config["embedding_dim"])
NUM_HEADS = int(config["num_heads"])
NUM_LAYERS = int(config["num_layers"])

# Handles both the corrected name and your current typo.
DROPOUT = float(config.get("dropout", config.get("dropomp", 0.2)))


print(f"Using device: {device}")
print(f"Loaded vocabulary size: {vocab_size}")
print(f"Block size: {BLOCK_SIZE}")
print(f"Embedding dim: {EMBEDDING_DIM}")
print(f"Attention heads: {NUM_HEADS}")
print(f"Layers: {NUM_LAYERS}")


# --------------------------------------------------
# Character-Level Tokenizer
# --------------------------------------------------

def encode(input_text):
    """
    Converts text into character IDs.

    Since this is a character-level model, every character in the prompt
    must exist in the training vocabulary. Unknown characters are skipped.
    """

    encoded = []

    for ch in input_text:
        if ch in char_to_id:
            encoded.append(char_to_id[ch])
        else:
            print(f"Warning: character not in vocabulary and will be skipped: {repr(ch)}")

    return encoded


def decode(token_ids):
    """
    Converts token IDs back into text.
    """

    return "".join([id_to_char[int(i)] for i in token_ids if int(i) in id_to_char])


# --------------------------------------------------
# Transformer Components
# --------------------------------------------------

class AttentionHead(nn.Module):
    """
    One self-attention head.
    """

    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(EMBEDDING_DIM, head_size, bias=False)
        self.query = nn.Linear(EMBEDDING_DIM, head_size, bias=False)
        self.value = nn.Linear(EMBEDDING_DIM, head_size, bias=False)

        self.register_buffer(
            "tril",
            torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))
        )

        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        batch_size, time_steps, channels = x.shape

        k = self.key(x)
        q = self.query(x)

        attention_scores = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)

        attention_scores = attention_scores.masked_fill(
            self.tril[:time_steps, :time_steps] == 0,
            float("-inf")
        )

        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        v = self.value(x)
        output = attention_weights @ v

        return output


class MultiHeadAttention(nn.Module):
    """
    Multiple self-attention heads in parallel.
    """

    def __init__(self, num_heads, head_size):
        super().__init__()

        self.heads = nn.ModuleList([
            AttentionHead(head_size) for _ in range(num_heads)
        ])

        self.projection = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        output = torch.cat([head(x) for head in self.heads], dim=-1)
        output = self.projection(output)
        output = self.dropout(output)
        return output


class FeedForward(nn.Module):
    """
    Feed-forward layer inside each transformer block.
    """

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(EMBEDDING_DIM, 4 * EMBEDDING_DIM),
            nn.ReLU(),
            nn.Linear(4 * EMBEDDING_DIM, EMBEDDING_DIM),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.network(x)


class TransformerBlock(nn.Module):
    """
    Transformer block:
    - Multi-head self-attention
    - Feed-forward network
    - Residual connections
    - Layer normalization
    """

    def __init__(self):
        super().__init__()

        head_size = EMBEDDING_DIM // NUM_HEADS

        self.self_attention = MultiHeadAttention(NUM_HEADS, head_size)
        self.feed_forward = FeedForward()

        self.layer_norm_1 = nn.LayerNorm(EMBEDDING_DIM)
        self.layer_norm_2 = nn.LayerNorm(EMBEDDING_DIM)

    def forward(self, x):
        x = x + self.self_attention(self.layer_norm_1(x))
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x


# --------------------------------------------------
# GPT-Style Language Model
# --------------------------------------------------

class TinyTransformerLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, EMBEDDING_DIM)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, EMBEDDING_DIM)

        self.blocks = nn.Sequential(*[
            TransformerBlock() for _ in range(NUM_LAYERS)
        ])

        self.final_layer_norm = nn.LayerNorm(EMBEDDING_DIM)
        self.language_model_head = nn.Linear(EMBEDDING_DIM, vocab_size)

    def forward(self, index, targets=None):
        batch_size, time_steps = index.shape

        token_embeddings = self.token_embedding_table(index)

        position_indices = torch.arange(time_steps, device=index.device)
        position_embeddings = self.position_embedding_table(position_indices)

        x = token_embeddings + position_embeddings
        x = self.blocks(x)
        x = self.final_layer_norm(x)

        logits = self.language_model_head(x)

        if targets is None:
            loss = None
        else:
            batch_size, time_steps, channels = logits.shape

            logits = logits.view(batch_size * time_steps, channels)
            targets = targets.view(batch_size * time_steps)

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        index,
        max_new_tokens=300,
        temperature=0.8,
        top_k=20
    ):
        """
        Generates new text.

        temperature:
            Lower values make output more predictable.
            Higher values make output more random.

        top_k:
            Limits sampling to the top K most likely next characters.
        """

        self.eval()

        for _ in range(max_new_tokens):
            index_context = index[:, -BLOCK_SIZE:]

            logits, loss = self(index_context)

            logits = logits[:, -1, :]

            if temperature <= 0:
                temperature = 1.0

            logits = logits / temperature

            if top_k is not None and top_k > 0:
                values, indices = torch.topk(logits, min(top_k, logits.size(-1)))
                filtered_logits = torch.full_like(logits, float("-inf"))
                filtered_logits.scatter_(1, indices, values)
                logits = filtered_logits

            probabilities = F.softmax(logits, dim=-1)

            next_index = torch.multinomial(probabilities, num_samples=1)

            index = torch.cat((index, next_index), dim=1)

        return index


# --------------------------------------------------
# Load Trained Model
# --------------------------------------------------

def load_model():
    model_path = Path(MODEL_FILE)

    if not model_path.exists():
        raise FileNotFoundError(f"Could not find trained model file: {MODEL_FILE}")

    model = TinyTransformerLanguageModel().to(device)

    state_dict = torch.load(MODEL_FILE, map_location=device)
    model.load_state_dict(state_dict)

    model.eval()

    return model


# --------------------------------------------------
# Chat Functions
# --------------------------------------------------

def generate_response(
    model,
    prompt,
    max_new_tokens=300,
    temperature=0.8,
    top_k=20
):
    token_ids = encode(prompt)

    if len(token_ids) == 0:
        return "I could not encode the prompt because none of the characters exist in the model vocabulary."

    input_tensor = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=device
    )

    generated_ids = model.generate(
        input_tensor,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k
    )

    generated_text = decode(generated_ids[0].tolist())

    return generated_text


def chat_loop():
    model = load_model()

    print("\nTiny Transformer Chat Interface")
    print("--------------------------------")
    print("Type a prompt and press Enter.")
    print("Commands:")
    print("  /exit       Quit")
    print("  /settings   Show generation settings")
    print()

    max_new_tokens = 300
    temperature = 0.8
    top_k = 20

    while True:
        prompt = input("You: ").strip()

        if prompt.lower() in ["/exit", "exit", "quit", "/quit"]:
            print("Exiting chat.")
            break

        if prompt.lower() == "/settings":
            print(f"max_new_tokens: {max_new_tokens}")
            print(f"temperature: {temperature}")
            print(f"top_k: {top_k}")
            continue

        if not prompt:
            continue

        try:
            response = generate_response(
                model=model,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k
            )

            print("\nModel:")
            print(response)
            print()

        except Exception as e:
            print(f"\nError during generation: {e}\n")


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":
    chat_loop()