# MyLLM — Mini Transformer Spec (Java, decoder-only, from scratch)

Goal: build tiny GPT-style transformer. Understand math, own code, no ML libs (no DL4J/ND4J/etc). Plain Java arrays/double[][] only.

Scope kept small on purpose: char-level (or tiny word) tokenizer, small vocab, small `d_model`, 1-4 layers, 1-4 heads, train on small text corpus, forward + backward pass by hand (manual backprop, chain rule).

---

## 1. Overall pipeline

```
text -> tokens (int ids) -> token embedding + positional embedding
     -> N x DecoderBlock (masked self-attention + FFN, each with residual + layernorm)
     -> final layernorm
     -> output projection to vocab logits
     -> softmax -> next-token probabilities
```

Training: cross-entropy loss on next-token prediction, backprop through everything, gradient descent (plain SGD or Adam) update weights.

---

## 2. Notation

- `V` = vocab size
- `T` = sequence length (context window, e.g. 32-64)
- `d_model` = embedding dimension (e.g. 32-64)
- `h` = number heads
- `d_k = d_model / h` = per-head dimension
- `d_ff` = hidden size of feedforward (usually 4 * d_model)
- `B` = batch size (can start with B=1 to keep it simple)
- Matrices: row = position/token, column = feature. `X` shape `(T, d_model)`.

---

## 3. Tokenizer

Simplest: character-level.

- Build vocab = sorted set of unique chars in training corpus.
- `stoi: char -> int`, `itos: int -> char`.
- Encode string -> `int[]` ids. Decode reverse.

No BPE needed for spec-level understanding (can add later as stretch goal).

---

## 4. Embeddings

### 4.1 Token embedding
Matrix `W_te` shape `(V, d_model)`. Row `i` = embedding vector for token id `i`. Lookup only (no matmul), gradient = scatter-add into rows used.

### 4.2 Positional embedding
Two options, pick ONE:

**Option A (learned, simplest to implement/backprop):**
Matrix `W_pe` shape `(T_max, d_model)`. Row `p` = vector for position `p`. Lookup like token embedding.

**Option B (sinusoidal, no extra learned params):**
For position `pos` (0-indexed) and dimension index `i` (0..d_model-1):

```
PE(pos, 2i)   = sin( pos / 10000^(2i/d_model) )
PE(pos, 2i+1) = cos( pos / 10000^(2i/d_model) )
```

Recommend Option A first — easier to reason about gradients while learning.

### 4.3 Combine
```
X[t] = TokenEmbedding[token_id_t] + PositionalEmbedding[t]     for t = 0..T-1
```
`X` shape `(T, d_model)`.

---

## 5. Scaled dot-product attention (single head)

Given input `X` shape `(T, d_model)`, per-head weight matrices:

- `W_q, W_k, W_v` each shape `(d_model, d_k)`

Compute:
```
Q = X · W_q     shape (T, d_k)
K = X · W_k     shape (T, d_k)
V = X · W_v     shape (T, d_k)

scores = Q · K^T / sqrt(d_k)     shape (T, T)
```

**Causal mask** (decoder-only — token t can't see future tokens > t):
```
for row i, col j:
    if j > i: scores[i][j] = -infinity   (use large negative, e.g. -1e9)
```

```
attn_weights = row_softmax(scores)       shape (T, T)   -- softmax over each row
output = attn_weights · V                shape (T, d_k)
```

Softmax per row `i`:
```
softmax(z)_j = exp(z_j - max(z)) / sum_k exp(z_k - max(z))
```
(subtract max for numerical stability — do this always.)

---

## 6. Multi-head attention

- Run `h` independent heads in parallel, each with its own `W_q, W_k, W_v` (shape `(d_model, d_k)`).
- Concatenate head outputs along feature axis -> shape `(T, h*d_k)` = `(T, d_model)`.
- Project through output matrix `W_o` shape `(d_model, d_model)`:
```
MHA(X) = Concat(head_1, ..., head_h) · W_o
```

---

## 7. Residual connection + LayerNorm

Applied around both attention and feedforward sublayers. Use **pre-norm** (easier gradients, modern standard):

```
X1 = X + MultiHeadAttention(LayerNorm(X))
X2 = X1 + FeedForward(LayerNorm(X1))
```

### LayerNorm (per row, across features)
For a row vector `x` of length `d_model`, with learned scale `gamma` and shift `beta` (vectors length `d_model`):

```
mean = (1/d_model) * sum(x)
var  = (1/d_model) * sum((x - mean)^2)
x_hat = (x - mean) / sqrt(var + eps)     eps ~ 1e-5
y = gamma * x_hat + beta
```

---

## 8. Feedforward network (position-wise)

Applied independently to each row (token position):

```
FFN(x) = W2 · activation(W1 · x + b1) + b2
```

- `W1` shape `(d_model, d_ff)`, `b1` shape `(d_ff)`
- `W2` shape `(d_ff, d_model)`, `b2` shape `(d_model)`
- activation: ReLU is simplest — `relu(z) = max(0, z)`. (GELU optional stretch goal.)

---

## 9. Decoder block summary

One block = steps 6+7+8 combined:

```
DecoderBlock(X):
    A = MultiHeadAttention(LayerNorm1(X))
    X = X + A
    F = FeedForward(LayerNorm2(X))
    X = X + F
    return X
```

Stack `N` blocks sequentially (each with own weights).

---

## 10. Output head

After final block, apply one more LayerNorm, then project to vocab logits:

```
X_final = LayerNorm_final(X)          shape (T, d_model)
logits = X_final · W_unembed          shape (T, V)
```

`W_unembed` shape `(d_model, V)`. Optional: tie weights so `W_unembed = W_te^T` (saves params, common trick) — stretch goal, start with separate matrix.

For generation: take last row `logits[T-1]`, softmax -> probability distribution over vocab, sample or argmax next token.

---

## 11. Loss (training)

Cross-entropy over next-token prediction. For each position `t` (0..T-2), target is token at `t+1`:

```
p_t = softmax(logits[t])
loss_t = -log(p_t[target_id])
total_loss = (1/T) * sum(loss_t)
```

Gradient of softmax+cross-entropy combined (standard, avoids re-deriving softmax grad separately):
```
dL/dlogits[t][j] = p_t[j] - 1{j == target_id}
```
This is the seed gradient you backprop from.

---

## 12. Backpropagation — what to derive yourself

This is the hard/valuable part. Work backward through:

1. Output projection (`W_unembed`) — standard matmul gradient.
2. Final LayerNorm — layernorm backward (the trickiest formula; look up "layernorm backward derivation" once you understand forward, or derive via chain rule on mean/var).
3. Each decoder block in reverse order:
   - FeedForward backward (through W2, activation, W1) — standard matmul + ReLU gradient (`dReLU = 1 if z>0 else 0`).
   - Residual add — gradient just flows through both branches (add gradients).
   - LayerNorm backward (again).
   - Multi-head attention backward — hardest part: backprop through softmax, through masking (masked positions get zero gradient), through Q/K/V projections.
4. Embedding backward — scatter gradient into rows of `W_te`/`W_pe` that were looked up (accumulate, since same token may appear multiple times).

Suggested approach: implement forward pass first fully, verify with **numerical gradient checking** (perturb one weight by epsilon, compare finite-difference gradient to your analytic backward gradient) before trusting your backward code. This catches nearly all backprop bugs.

---

## 13. Matrix op primitives needed (build once, reuse everywhere)

- `matmul(A, B)`
- `transpose(A)`
- `add(A, B)`, `addRowVector(A, v)` (broadcast bias)
- `scale(A, scalar)`
- `elementwiseMul(A, B)`
- `rowSoftmax(A)` (+ its backward)
- `relu(A)` (+ its backward)
- `layerNormForward(x, gamma, beta)` (+ its backward)
- Random init: small random values, e.g. uniform in `[-0.1, 0.1]` or Xavier/He init (`scale = sqrt(1/d_model)` roughly) — matters for training stability but not for understanding forward math.

---

## 14. Suggested Java class layout (structure only, no impl)

```
tokenizer/
    CharTokenizer.java        encode/decode, vocab build

math/
    Matrix.java               double[][] wrapper + matmul/add/transpose/etc
    Ops.java                  softmax, relu, layernorm (static helpers), each with forward+backward

model/
    Embedding.java            token + positional embedding lookup, backward = scatter-add
    AttentionHead.java        single head forward/backward
    MultiHeadAttention.java   owns h AttentionHeads + W_o
    FeedForward.java          W1/b1/W2/b2 forward/backward
    LayerNorm.java            gamma/beta forward/backward
    DecoderBlock.java         wires attention+ffn+norms+residual
    TransformerModel.java     stacks N DecoderBlocks + embedding + output head

train/
    Optimizer.java            SGD or Adam step
    Trainer.java               training loop: forward, loss, backward, step
    GradCheck.java             numerical gradient check utility

Main.java                     wire it up: load text, build vocab, train, generate sample
```

Each forward-capable class should cache whatever intermediate values it needs (inputs, softmax outputs, normalized values, masks) so its own `backward()` can use them — no autodiff engine, so you manage the tape manually per class.

---

## 15. Recommended build order (incremental, testable at each step)

1. `Matrix` + `Ops` (matmul, softmax, relu) + unit-test them by hand on tiny 2x2/3x3 examples.
2. `CharTokenizer` on a tiny string ("hello world" style corpus).
3. `Embedding` forward only — check shapes.
4. Single `AttentionHead` forward only, no mask, verify against hand-computed tiny example.
5. Add causal mask.
6. `MultiHeadAttention` (combine heads).
7. `LayerNorm` forward.
8. `FeedForward` forward.
9. Assemble one `DecoderBlock` forward, then full `TransformerModel` forward -> logits. Verify output shape `(T, V)` and that softmax rows sum to 1.
10. Loss forward (cross-entropy) on random-init model — sanity check: loss should be roughly `ln(V)` at init (uniform-ish predictions).
11. Backward pass, class by class, innermost (output layer) outward. Use `GradCheck` after each class to confirm correctness before moving to the next.
12. Full training loop on tiny corpus (e.g. a few sentences repeated) — confirm loss decreases.
13. Greedy/sampled generation from trained model — qualitative sanity check (memorizes tiny corpus).

---

## 16. Suggested tiny hyperparameters to start

```
V        ~ 30-70   (char vocab of small corpus)
T        = 16-32
d_model  = 32
h        = 2 or 4
d_k      = d_model / h
d_ff     = 128
N layers = 2
learning rate ~ 1e-2 to 1e-3 (SGD) or 1e-3 (Adam)
```

Small enough to run pure-Java, no GPU, seconds per epoch on tiny corpus.

---

## 17. Stretch goals (after core works)

- BPE tokenizer instead of char-level.
- Weight tying (`W_unembed = W_te^T`).
- Adam optimizer instead of SGD.
- Dropout (forward: zero random units + scale; backward: mask same units).
- Sinusoidal positional encoding instead of learned.
- Batch training (B > 1) instead of single sequence.
- Top-k / temperature sampling for generation instead of greedy.

---

## References for self-study (concepts only, implement yourself)

- "Attention Is All You Need" (Vaswani et al., 2017) — original transformer paper, defines scaled dot-product attention, multi-head attention, positional encoding formulas above.
- Any GPT architecture description (decoder-only, causal mask, pre-norm) — this spec follows that variant since it's simpler than encoder-decoder.
- LayerNorm backward: derive via chain rule treating mean and variance as intermediate functions of `x` — worth working out on paper before coding.
