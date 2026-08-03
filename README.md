# Tiny Transformer AI Training Project

## Overview

This project implements a custom GPT-style Transformer Language Model using PyTorch. The model learns from a text file and is trained using next-token prediction, which is the same fundamental technique used by modern Large Language Models (LLMs).

The project demonstrates:

- Text tokenization
- Vocabulary creation
- Character embeddings
- Self-attention
- Multi-head attention
- Transformer blocks
- Language model pre-training
- Text generation
- Model persistence
- Interactive chatbot inference

This implementation is intended as an educational and experimental framework for understanding how transformer-based language models work internally.

---

## How It Works

### Training Process

The training pipeline follows these steps:

```text
Text File
    ↓
Tokenizer
    ↓
Character Vocabulary
    ↓
Embedding Layer
    ↓
Transformer Blocks
    ↓
Next Token Prediction
    ↓
Loss Calculation
    ↓
Backpropagation
    ↓
Updated Model Weights
```

The model repeatedly learns how to predict the next character in a sequence.

Example:

```text
Input:
"The server run"

Expected Output:
"s"
```

After training on thousands of examples, the model learns language patterns contained within the training dataset.

---

## Project Components

### Training Data

The training corpus is stored in:

```text
training_text.txt
```

This file can contain:

- Technical documentation
- IT architecture descriptions
- Operational procedures
- Question and answer pairs
- General text

The quality of the generated responses depends heavily on the quality and volume of this file.

---

### Character-Level Tokenization

Unlike modern LLMs that use sub-word tokenization, this project uses character-level tokenization.

Example:

```text
Server
```

Becomes:

```python
['S', 'e', 'r', 'v', 'e', 'r']
```

Each unique character receives an integer ID:

```python
{
    "S": 0,
    "e": 1,
    "r": 2,
    ...
}
```

---

### Embedding Layer

Characters are converted into dense vectors.

Example:

```text
"S"
```

May become:

```python
[0.41, -0.18, 0.77, ...]
```

The embedding dimension is configured using:

```python
EMBEDDING_DIM = 256
```

These vectors are learned during training.

---

## Transformer Architecture

### Positional Embeddings

Transformers must understand token order.

Example:

```text
Server Storage
```

is different from

```text
Storage Server
```

Positional embeddings encode token position information.

---

### Self-Attention

Self-attention allows the model to determine which previous characters are important when predicting the next character.

Example:

```text
Virtual machines run on ______
```

The model learns to attend to:

```text
Virtual
machines
run
```

when predicting the next word.

---

### Multi-Head Attention

The model uses multiple attention heads simultaneously:

```python
NUM_HEADS = 8
```

Different heads learn different relationships within the text.

Example:

Head 1 may learn:

```text
Grammar
```

Head 2 may learn:

```text
Technical terms
```

Head 3 may learn:

```text
Sentence structure
```

---

### Transformer Blocks

The model contains:

```python
NUM_LAYERS = 6
```

Transformer blocks.

Each block contains:

- Multi-head attention
- Feed-forward network
- Layer normalization
- Residual connections

Structure:

```text
Input
   ↓
Layer Norm
   ↓
Self Attention
   ↓
Residual Connection
   ↓
Layer Norm
   ↓
Feed Forward
   ↓
Residual Connection
   ↓
Output
```

---

## Training Process

### Dataset Creation

Training samples are created automatically.

Example:

```text
Input:
"The server"

Target:
" "
```

Then:

```text
Input:
"The server "

Target:
"r"
```

This allows the model to learn sequential language patterns.

---

### Loss Calculation

The model uses:

```python
CrossEntropyLoss
```

to measure prediction error.

Lower values indicate better predictions.

Example:

```text
Train Loss: 3.8
```

Eventually becomes:

```text
Train Loss: 1.2
```

as training progresses.

---

### Backpropagation

The model learns using:

```python
loss.backward()
optimizer.step()
```

This updates millions of parameters to minimize prediction error.

---

## Model Training

Key training parameters:

```python
BATCH_SIZE = 32
BLOCK_SIZE = 128

EMBEDDING_DIM = 256

NUM_HEADS = 8
NUM_LAYERS = 6

MAX_ITERS = 5000

LEARNING_RATE = 3e-4
```

### Context Window

```python
BLOCK_SIZE = 128
```

The model can view up to 128 characters of context when making predictions.

---

## Text Generation

After training, text generation occurs using:

```python
model.generate(...)
```

The model predicts one character at a time.

Example:

Prompt:

```text
The server
```

Generated Result:

```text
The server runs virtual machines and provides application hosting services.
```

The text is generated autoregressively:

```text
Predict next character
    ↓
Append character
    ↓
Predict next character
    ↓
Repeat
```

---

## Model Output Files

After training, the following files are created:

### Model Weights

```text
model/pytorch_model.bin
```

Contains all learned neural network weights.

---

### Vocabulary

```text
model/vocab.json
```

Maps characters to token IDs.

Example:

```json
{
  "A": 0,
  "B": 1,
  "C": 2
}
```

---

### Tokenizer Definition

```text
model/tokenizer.json
```

Contains tokenizer metadata.

---

### Tokenizer Configuration

```text
model/tokenizer_config.json
```

Stores tokenizer settings.

---

### Special Tokens

```text
model/special_tokens_map.json
```

Defines:

```text
[UNK]
[BOS]
[EOS]
[PAD]
```

special tokens.

---

### Model Configuration

```text
model/config.json
```

Stores hyperparameters such as:

```json
{
  "vocab_size": 80,
  "block_size": 128,
  "embedding_dim": 256,
  "num_heads": 8,
  "num_layers": 6
}
```

---

## Chat Interface

A separate chat application can load:

```text
model/pytorch_model.bin
```

and interactively generate responses.

Example:

```text
You: What is virtualization?

Model:
Virtualization is the process of running multiple virtual machines on a physical server.
```

Note that the quality of answers is directly dependent on the training data provided.

---

## Limitations

This project is intended for learning and experimentation.

Limitations include:

- Character-level tokenization
- Small context window
- No instruction tuning
- No reinforcement learning
- No retrieval augmentation
- No distributed training
- Limited training corpus

As a result, performance will be significantly below production models such as:

- GPT-4
- GPT-5
- Llama
- Claude
- Gemini
- Qwen

---

## Suggested Improvements

Future enhancements could include:

### Better Tokenization

Replace character tokens with:

- BPE
- SentencePiece
- WordPiece

---

### Instruction Tuning

Train using:

```text
User:
Question

Assistant:
Answer
```

pairs for chatbot behavior.

---

### Larger Training Corpus

Add:

- Technical documentation
- Architecture standards
- Operational procedures
- Knowledge base articles

---

### Retrieval Augmented Generation (RAG)

Add:

- Embeddings
- Vector database
- Similarity search

to answer questions from real documents.

---

## Educational Goals

This project provides hands-on experience with:

- Neural networks
- PyTorch
- Tokenization
- Embeddings
- Self-attention
- Transformer architecture
- Language model training
- Text generation
- Model serialization
- AI chatbot development

It serves as a practical demonstration of the core technologies that power modern Large Language Models.


