import torch
import torch.nn as nn
from torch.nn import functional as F
from pathlib import Path
import json
import time


# --------------------------------------------------
# Configuration
# --------------------------------------------------

TEXT_FILE = "training_text.txt"
MODEL_FILE = "tiny_transformer_model.pt"
VOCAB_FILE = "tiny_transformer_vocab.json"

BATCH_SIZE = 32
BLOCK_SIZE = 64          # Context window size
MAX_ITERS = 2000
EVAL_INTERVAL = 200
LEARNING_RATE = 3e-4
EVAL_ITERS = 100

EMBEDDING_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 4
DROPOUT = 0.2

GENERATE_TOKENS = 300


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
print(f"Using device: {device}")


# --------------------------------------------------
# Load Text
# --------------------------------------------------

def load_text(file_path):
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {file_path}. Create a text file named {file_path} in this folder."
        )

    return path.read_text(encoding="utf-8")


text = load_text(TEXT_FILE)

if len(text) < 100:
    print("Warning: Your text file is very small. The model may not learn useful patterns.")


# --------------------------------------------------
# Character-Level Tokenizer
# --------------------------------------------------

chars = sorted(list(set(text)))
vocab_size = len(chars)

char_to_id = {ch: i for i, ch in enumerate(chars)}
id_to_char = {i: ch for ch, i in char_to_id.items()}


def encode(input_text):
    return [char_to_id[ch] for ch in input_text]


def decode(token_ids):
    return "".join([id_to_char[i] for i in token_ids])


print(f"Text length: {len(text)} characters")
print(f"Vocabulary size: {vocab_size}")


# --------------------------------------------------
# Prepare Dataset
# --------------------------------------------------

data = torch.tensor(encode(text), dtype=torch.long)

split_index = int(0.9 * len(data))
train_data = data[:split_index]
val_data = data[split_index:]


def get_batch(split):
    source_data = train_data if split == "train" else val_data

    if len(source_data) <= BLOCK_SIZE:
        raise ValueError(
            f"Text is too short for BLOCK_SIZE={BLOCK_SIZE}. "
            f"Add more text or reduce BLOCK_SIZE."
        )

    ix = torch.randint(len(source_data) - BLOCK_SIZE, (BATCH_SIZE,))

    x = torch.stack([source_data[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([source_data[i + 1:i + BLOCK_SIZE + 1] for i in ix])

    x = x.to(device)
    y = y.to(device)

    return x, y


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
    Simple feed-forward layer used inside each transformer block.
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

        position_indices = torch.arange(time_steps, device=device)
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

    def generate(self, index, max_new_tokens):
        for _ in range(max_new_tokens):
            index_context = index[:, -BLOCK_SIZE:]

            logits, loss = self(index_context)

            logits = logits[:, -1, :]

            probabilities = F.softmax(logits, dim=-1)

            next_index = torch.multinomial(probabilities, num_samples=1)

            index = torch.cat((index, next_index), dim=1)

        return index


# --------------------------------------------------
# Loss Evaluation
# --------------------------------------------------

@torch.no_grad()
def estimate_loss(model):
    output = {}

    model.eval()

    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_ITERS)

        for k in range(EVAL_ITERS):
            x_batch, y_batch = get_batch(split)
            logits, loss = model(x_batch, y_batch)
            losses[k] = loss.item()

        output[split] = losses.mean()

    model.train()

    return output


# --------------------------------------------------
# Save Model and Vocabulary
# --------------------------------------------------

def save_artifacts(model):
    torch.save(model.state_dict(), MODEL_FILE)

    vocab_data = {
        "char_to_id": char_to_id,
        "id_to_char": {str(k): v for k, v in id_to_char.items()},
        "vocab_size": vocab_size,
        "block_size": BLOCK_SIZE,
        "embedding_dim": EMBEDDING_DIM,
        "num_heads": NUM_HEADS,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
    }

    with open(VOCAB_FILE, "w", encoding="utf-8") as file:
        json.dump(vocab_data, file, indent=4)

    print(f"Saved model to: {MODEL_FILE}")
    print(f"Saved vocabulary to: {VOCAB_FILE}")


# --------------------------------------------------
# Main Training Loop
# --------------------------------------------------

def main():
    torch.manual_seed(1337)

    model = TinyTransformerLanguageModel().to(device)

    total_parameters = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_parameters:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    start_time = time.time()

    for iteration in range(MAX_ITERS):
        if iteration % EVAL_INTERVAL == 0 or iteration == MAX_ITERS - 1:
            losses = estimate_loss(model)
            elapsed = time.time() - start_time

            print(
                f"Step {iteration:5d} | "
                f"Train loss: {losses['train']:.4f} | "
                f"Val loss: {losses['val']:.4f} | "
                f"Elapsed: {elapsed:.1f}s"
            )

        x_batch, y_batch = get_batch("train")

        logits, loss = model(x_batch, y_batch)

        optimizer.zero_grad(set_to_none=True)

loss.backward()

optimizer.step()

print("\nGenerating sample text...\n")

start_token = torch.zeros((1, 1), dtype=torch.long, device=device)

generated_ids = model.generate(start_token, max_new_tokens=GENERATE_TOKENS)

generated_text = decode(generated_ids[0].tolist())


print(generated_text)

save_artifacts(model)


if __name__ == "__main__":
main()