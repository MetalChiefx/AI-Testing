import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import re
import json
import random
from pathlib import Path


# -----------------------------
# 1. Configuration
# -----------------------------

TEXT_FILE = "training_text.txt"
MODEL_FILE = "simple_language_model.pt"
VOCAB_FILE = "vocab.json"

EMBEDDING_DIM = 64
HIDDEN_DIM = 128
CONTEXT_SIZE = 5
EPOCHS = 200
LEARNING_RATE = 0.01


# -----------------------------
# 2. Device Setup
# -----------------------------

def get_device():
    """
    Uses Apple Silicon GPU acceleration if available,
    otherwise falls back to CPU.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


device = get_device()
print(f"Using device: {device}")


# -----------------------------
# 3. Tokenization
# -----------------------------

def tokenize(text):
    """
    Very simple tokenizer:
    - lowercases text
    - separates words and punctuation
    """
    text = text.lower()
    tokens = re.findall(r"\b\w+\b|[^\w\s]", text)
    return tokens


# -----------------------------
# 4. Load Text File
# -----------------------------

def load_text_file(file_path):
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {file_path}. Create a text file named {file_path} in this folder."
        )

    with open(path, "r", encoding="utf-8") as file:
        return file.read()


# -----------------------------
# 5. Build Vocabulary
# -----------------------------

def build_vocab(tokens):
    """
    Creates mappings:
    token_to_id: word -> number
    id_to_token: number -> word
    """
    token_counts = Counter(tokens)

    vocab = ["<PAD>", "<UNK>"] + sorted(token_counts.keys())

    token_to_id = {token: idx for idx, token in enumerate(vocab)}
    id_to_token = {idx: token for token, idx in token_to_id.items()}

    return token_to_id, id_to_token


def tokens_to_ids(tokens, token_to_id):
    return [token_to_id.get(token, token_to_id["<UNK>"]) for token in tokens]


# -----------------------------
# 6. Create Training Examples
# -----------------------------

def create_training_data(token_ids, context_size):
    """
    Creates examples like:

    Input:
        the server runs virtual machines

    If context_size = 3:
        [the, server, runs] -> virtual
        [server, runs, virtual] -> machines

    This is next-token prediction.
    """
    X = []
    y = []

    for i in range(len(token_ids) - context_size):
        context = token_ids[i:i + context_size]
        target = token_ids[i + context_size]

        X.append(context)
        y.append(target)

    return torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long)


# -----------------------------
# 7. Simple Language Model
# -----------------------------

class SimpleLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, context_size):
        super(SimpleLanguageModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.network = nn.Sequential(
            nn.Linear(context_size * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size)
        )

    def forward(self, x):
        """
        x shape:
            batch_size x context_size

        embedding output shape:
            batch_size x context_size x embedding_dim

        flattened shape:
            batch_size x context_size * embedding_dim
        """
        embedded = self.embedding(x)
        flattened = embedded.view(embedded.size(0), -1)
        output = self.network(flattened)
        return output


# -----------------------------
# 8. Training Loop
# -----------------------------

def train_model(model, X, y, epochs, learning_rate):
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    for epoch in range(1, epochs + 1):
        X = X.to(device)
        y = y.to(device)

        predictions = model(X)
        loss = loss_function(predictions, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.4f}")


# -----------------------------
# 9. Generate Text
# -----------------------------

def generate_text(model, starting_text, token_to_id, id_to_token, max_tokens=25):
    model.eval()

    tokens = tokenize(starting_text)

    if len(tokens) < CONTEXT_SIZE:
        tokens = ["<PAD>"] * (CONTEXT_SIZE - len(tokens)) + tokens

    generated_tokens = tokens[:]

    for _ in range(max_tokens):
        context_tokens = generated_tokens[-CONTEXT_SIZE:]
        context_ids = [
            token_to_id.get(token, token_to_id["<UNK>"])
            for token in context_tokens
        ]

        input_tensor = torch.tensor([context_ids], dtype=torch.long).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)

            next_token_id = torch.multinomial(probabilities[0], num_samples=1).item()
            next_token = id_to_token[next_token_id]

        generated_tokens.append(next_token)

    return " ".join(generated_tokens)


# -----------------------------
# 10. Save Model and Vocabulary
# -----------------------------

def save_model(model, token_to_id, id_to_token):
    torch.save(model.state_dict(), MODEL_FILE)

    vocab_data = {
        "token_to_id": token_to_id,
        "id_to_token": {str(k): v for k, v in id_to_token.items()},
        "context_size": CONTEXT_SIZE,
        "embedding_dim": EMBEDDING_DIM,
        "hidden_dim": HIDDEN_DIM
    }

    with open(VOCAB_FILE, "w", encoding="utf-8") as file:
        json.dump(vocab_data, file, indent=4)

    print(f"Saved model to {MODEL_FILE}")
    print(f"Saved vocabulary to {VOCAB_FILE}")


# -----------------------------
# 11. Main Program
# -----------------------------

def main():
    print("Loading text...")
    text = load_text_file(TEXT_FILE)

    print("Tokenizing text...")
    tokens = tokenize(text)

    if len(tokens) <= CONTEXT_SIZE:
        raise ValueError(
            f"Text file is too small. Add more text. Need more than {CONTEXT_SIZE} tokens."
        )

    print(f"Total tokens: {len(tokens)}")

    print("Building vocabulary...")
    token_to_id, id_to_token = build_vocab(tokens)

    print(f"Vocabulary size: {len(token_to_id)}")

    print("Converting tokens to IDs...")
    token_ids = tokens_to_ids(tokens, token_to_id)

    print("Creating training examples...")
    X, y = create_training_data(token_ids, CONTEXT_SIZE)

    print(f"Training examples: {len(X)}")

    print("Creating model...")
    model = SimpleLanguageModel(
        vocab_size=len(token_to_id),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        context_size=CONTEXT_SIZE
    ).to(device)

    print("Starting pre-training...")
    train_model(model, X, y, EPOCHS, LEARNING_RATE)

    print("\nGenerating sample text...")
    seed_text = "the server"
    generated = generate_text(model, seed_text, token_to_id, id_to_token)

    print("\nGenerated text:")
    print(generated)

    save_model(model, token_to_id, id_to_token)


if __name__ == "__main__":
    main()