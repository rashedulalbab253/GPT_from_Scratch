# Building a GPT from Scratch

> Companion notebook to Andrej Karpathy's [Zero To Hero](https://karpathy.ai/zero-to-hero.html) series — a ground-up implementation of a GPT-style language model in pure PyTorch.

---

## Overview

This notebook walks through the complete construction of a character-level GPT, starting from a simple bigram model and progressively adding every component that makes the Transformer architecture work: tokenization, self-attention, multi-head attention, feedforward layers, residual connections, and layer normalization. By the end, you have a fully functional autoregressive language model trained on the Tiny Shakespeare dataset.

The goal is understanding — every line of code is written to be readable and instructive, not optimized for production.

---

## What You'll Learn

- **Tokenization** — character-level encoding and decoding; building vocabulary mappings from raw text
- **Data preparation** — train/val splits, batched sequence sampling, context windows
- **Bigram Language Model** — a simple baseline that predicts the next token from only the current one
- **The mathematical trick in self-attention** — weighted aggregation via lower-triangular matrix multiplication, evolving from loops → `torch.tril` → Softmax masking → full self-attention
- **Single-head self-attention** — keys, queries, values; scaled dot-product attention; causal masking
- **Multi-head attention** — running multiple attention heads in parallel and projecting back
- **Feedforward sublayers** — position-wise MLPs with ReLU non-linearity
- **Transformer blocks** — combining attention + feedforward with residual connections
- **Layer normalization** — implemented from scratch, with a clear contrast against BatchNorm
- **Full GPT assembly** — token embeddings, positional embeddings, stacked Transformer blocks, and a language model head
- **Training loop** — AdamW optimizer, loss estimation on train and validation sets, text generation

---

## Model Architecture (Final)

| Hyperparameter | Value |
|---|---|
| Embedding dimension (`n_embd`) | 64 |
| Attention heads (`n_head`) | 4 |
| Transformer layers (`n_layer`) | 4 |
| Block size (context length) | 32 |
| Batch size | 16 |
| Training steps | 5,000 |
| Learning rate | 1e-3 |
| Dropout | 0.0 |

The final model contains roughly **0.2M parameters** and is trained on the [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) dataset (~1MB of text).

---

## Key Concepts Explained

### Self-Attention as a Communication Mechanism
Attention is framed as nodes in a directed graph aggregating information from other nodes with data-dependent weights. Crucially, there is no inherent notion of position — which is why positional embeddings are added explicitly.

### Decoder vs. Encoder Attention
The notebook implements **decoder-style** (causal) attention: a triangular mask prevents any token from attending to future tokens, making the model autoregressive. Removing this mask gives an encoder block useful for tasks like translation.

### Scaled Dot-Product Attention
Raw `Q @ K^T` produces high-variance outputs that push Softmax into saturation. Dividing by `sqrt(head_size)` keeps the weights diffuse and gradients healthy.

### Layer Norm vs. Batch Norm
A custom `LayerNorm1d` is built from scratch to illustrate the difference: LayerNorm normalizes across the feature dimension of each individual example, making it well-suited for variable-length sequences in language modeling.

---

## Requirements

```bash
pip install torch
```

A GPU is recommended for the full training run but not required — the model is small enough to train on CPU.

---

## Getting Started

```bash
# 1. Download the dataset
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# 2. Open the notebook
jupyter notebook Gpt_from_scratch.ipynb

# 3. Run all cells top to bottom
```

---

## Credit

This notebook is the companion resource to **Andrej Karpathy's** [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) — part of his [Zero To Hero](https://karpathy.ai/zero-to-hero.html) deep learning series. Karpathy is a former Director of AI at Tesla and co-founder of OpenAI, and this series is widely regarded as one of the best free resources for deeply understanding modern neural networks.

The original repository lives at [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT).

---

## License

This notebook follows the licensing of the original material. See Andrej Karpathy's repository for details.
