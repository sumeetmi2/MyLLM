# MyLLM

A small GPT-style decoder-only transformer written from scratch in plain Java: no ML libraries, manual backprop.
See [SPEC.md](SPEC.md) for the architecture and math.

## List all supported commands

```
./gradlew run -q --console=plain --args="help"
```

Any unknown command prints the same usage text.

<!-- BEGIN GENERATED: usage (./gradlew updateDocs, from Main help) -->
```
usage:
  (no args)                                   demo: train on built-in text, no save
  train <corpus file|dir> <model.bin> [steps] [bpeVocab] [batch=B] [accum=K] [ckpt=<file>] [ckptEvery=N]
                                              clean, train and save (dir = all .md files under it)
                                              bpeVocab e.g. 1024 = BPE tokens, omitted = one token per char
                                              batch=B sequences per micro-batch, accum=K micro-batches
                                              per optimizer step (default 1 and 1)
                                              ckpt=<file> saves training state every ckptEvery steps
                                              (default 500) and at the end, for resume
  resume <file.ckpt> [ckptEvery=N]            continue an interrupted train/finetune run exactly
  serve <model.bin> [port]                    chat in the browser at http://localhost:8080 (default port);
                                              the page can switch to any .bin next to the model
  generate <model.bin> <prompt> [length]      load a saved model and generate text
  chat <model.bin> [length] [temperature] [topK] [template]
                                              interactive: type prompts in the terminal
                                              (defaults 0.8, 10; temperature 0 = greedy)
                                              template = send "User: <line>\nAssistant:", print the reply only
  eval <model.bin> <chat corpus> [samples]    validation loss on answers + plot-word hits on fixed prompts
                                              (default 10 samples per prompt)
  finetune <base.bin> <corpus> <out.bin> [steps] [peakLr] [lora=r] [lora-all] [sft]
                                              continue training a saved model on new text (same tokenizer)
                                              defaults 3000 steps, lr 0.0004 (0.002 with lora=r)
                                              lora=r trains only rank-r adapters on W_q/W_v, merged before saving
                                              lora-all also adapts W_o and the FFN (W_1, W_2)
                                              sft = corpus is User:/Assistant: examples, one per step,
                                              loss on the answer only
                                              batch=B / accum=K / ckpt=<file> / ckptEvery=N as for train
```
<!-- END GENERATED: usage -->

## Commands

All commands run from the project root. `-q --console=plain` hides Gradle's progress output; `chat` needs it so
the prompt reads cleanly.

| command | what it does |
|---|---|
| `./gradlew run` | demo: trains on a built-in sentence corpus, saves nothing |
| `train <corpus file\|dir> <model.bin> [steps] [bpeVocab]` | cleans the corpus, builds a tokenizer, trains a new model, saves it |
| `finetune <base.bin> <corpus> <out.bin> [steps] [peakLr] [lora=r] [lora-all] [sft]` | continues training a saved model on new text, with the base model's tokenizer and T_max |
| `generate <model.bin> <prompt> [length]` | loads a model and generates one continuation of the prompt |
| `chat <model.bin> [length] [temperature] [topK] [template]` | interactive prompt loop in the terminal |
| `eval <model.bin> <chat corpus> [samples]` | validation loss on answers + plot-word hit rates on fixed prompts |
| `resume <file.ckpt> [ckptEvery=N]` | continues a `train` / `finetune` run that was started with `ckpt=<file>` |
| `serve <model.bin> [port]` | chat page at http://localhost:8080 (localhost only); any `.bin` next to the model can be picked in the page |

`train` and `finetune` also take `batch=B`, `accum=K` (sequences per micro-batch, micro-batches per step),
`ckpt=<file>` and `ckptEvery=N` anywhere after the required arguments.

Pass the arguments with `--args="..."`, for example `./gradlew run -q --console=plain --args="chat models/out-chat-full.bin"`.

### train

```
./gradlew run -q --console=plain --args="train data/tinystories-train-100mb.txt models/out-pretrain-256.bin 20000 512"
```

- `corpus`: a text file, or a directory, in which case every `.md` file under it is used
- `steps`: default 2000; warmup is 5% of the steps, then the learning rate follows a cosine decay from 0.002 to 0.0002
- `bpeVocab`: BPE vocabulary size (e.g. 512); leave it out for one token per character
- model shape comes from the constants at the top of `src/main/java/com/myllm/Main.java` (T_MAX, D_MODEL, HEADS, D_FF, BLOCKS)

### finetune

```
# full fine-tune: every weight trains
./gradlew run -q --console=plain --args="finetune models/out-pretrain-256.bin data/instruct/chat-train.txt models/out-chat-full.bin 3000" | tee logs/ft-full.log

# LoRA: only rank-4 adapters on W_q / W_v train (~2% of weights), merged into the weights before saving
./gradlew run -q --console=plain --args="finetune models/out-pretrain-256.bin data/instruct/chat-train.txt models/out-chat-lora.bin 3000 lora=4" | tee logs/ft-lora.log

# supervised fine-tuning (sft): one whole chat example per step, loss on the Assistant answer only
./gradlew run -q --console=plain --args="finetune models/out-pretrain-256.bin data/instruct/chat-train.txt models/out-chat-lora-sft.bin 3000 lora=4 sft" | tee logs/ft-lora-sft.log
```

| arg | default | meaning |
|---|---|---|
| `base.bin` | required | trained model to start from; its tokenizer and T_max are reused |
| `corpus` | required | text to fine-tune on (file, or directory of `.md` files) |
| `out.bin` | required | where to save the result; must differ from `base.bin` |
| `steps` | 3000 | training steps (5% warmup, cosine decay to peakLr / 10) |
| `peakLr` | 0.0004, or 0.002 with LoRA | peak learning rate; needs `steps` before it |
| `lora=r` | off | train only rank-r LoRA adapters on W_q / W_v (alpha = 2r); can go anywhere after `out.bin` |
| `lora-all` | off | with `lora=r`: also put adapters on W_o and the FFN's W_1 / W_2 (11,264 × r trainable vs 5,120 × r) |
| `sft` | off | corpus is `User: ...\nAssistant: ...<\|endoftext\|>` examples; each step trains on one whole example (dropped if longer than T_max + 1 tokens) and only the answer counts in the loss; can go anywhere after `out.bin` |

Prints a sample from the base model before training, then a sample every 500 steps, so the change is visible.
The saved model is a normal model file, so `chat` and `generate` work on it unchanged.

### generate

```
./gradlew run -q --console=plain --args="generate models/out-pretrain-256.bin \"Once upon a time\" 300"
```

- `length`: tokens to generate, default 200; stops early at `<|endoftext|>`

### chat

```
./gradlew run -q --console=plain --args="chat models/out-pretrain-256.bin 300 0.8 10"

# fine-tuned chat model: each line is sent as "User: <line>\nAssistant:" and only the reply is printed
./gradlew run -q --console=plain --args="chat models/out-chat-lora-sft.bin 300 template"
```

- `length`: tokens per reply, default 200
- `temperature`: default 0.8; 0 = always pick the most likely token
- `topK`: default 10; sample only from the k most likely tokens
- `template`: wrap each line in the fine-tuning chat format (can go anywhere after `model.bin`); the reply stops
  at `<|endoftext|>` or if the model starts a new `User:` turn
- type `/quit` or press Ctrl+D to exit

With the template, phrase requests like the training data, e.g.
`Write a story with this plot: Tom finds a lost puppy in the park and takes it home.`

### eval

```
./gradlew run -q --console=plain --args="eval models/out-chat-lora-sft.bin data/instruct/chat-valid.txt"
```

- validation loss: average loss on Assistant answer tokens of every example in the corpus (same mask as `sft`),
  forward pass only; lower is better
- plot prompts: 5 fixed requests, `samples` replies each (default 10, temperature 0.8, topK 10, fixed seed);
  counts replies containing each plot word (whole word, case-insensitive) and how many stopped on their own
- last line: `summary <model>: val_loss ... plot_score ...` (plot_score = average fraction of plot words per reply)
- run it on the base model too: common words (mom, happy, park) show up in many stories anyway, so compare
  against the base, not against zero

## Data preparation

Convert TinyStories-Instruct into single-turn chat examples (`User: ... / Assistant: ... / <|endoftext|>`) for `finetune`:

```
python3 scripts/convert_instruct.py data/instruct/TinyStories-Instruct-train-150mb.txt data/instruct/chat-train.txt
python3 scripts/convert_instruct.py data/instruct/TinyStories-Instruct-valid.txt data/instruct/chat-valid.txt
```

- `--max-chars` (default 700): drops examples longer than this, so prompt + story fit in 256 tokens
- `--seed` (default 42): fixes which request phrasing each example gets

## Project layout

| path | contents |
|---|---|
| `src/main/java/com/myllm` | the code, one package per component (see the code map below); `Main` is the command line |
| `src/main/resources/web` | the chat page served by `serve` |
| `src/test/java` | unit tests |
| `config/checkstyle` | Checkstyle rules and suppressions |
| `scripts` | data preparation |
| `data/` | corpora (git-ignored) |
| `models/` | trained `.bin` models (git-ignored) |
| `logs/` | training logs saved with `tee` (git-ignored) |

### Code map

<!-- BEGIN GENERATED: packages (./gradlew updateDocs, from each package-info.java) -->
| package | what it holds |
|---|---|
| `com.myllm` | A GPT-style decoder-only transformer written from scratch in plain Java; `Main` is the command line. |
| `com.myllm.attention` | Causal self-attention: single heads and the multi-head layer that concatenates them, with manual backward passes. |
| `com.myllm.batched` | Batched training path: stacks several sequences of different lengths as rows and runs the existing layers on them without padding. |
| `com.myllm.corpus` | Cleans raw training text: normalises characters, drops non-ASCII noise and removes duplicate paragraphs. |
| `com.myllm.crossentropy` | Softmax cross-entropy loss over the vocabulary, with an optional mask that leaves positions out of the loss. |
| `com.myllm.dao` | The flat row-major float `Matrix` and the SIMD kernel every matrix multiply goes through. |
| `com.myllm.decoder` | One transformer decoder block: pre-norm attention and feed-forward sublayers with residual connections. |
| `com.myllm.embedding` | Token and learned position embeddings. |
| `com.myllm.eval` | Evaluation of chat models: validation loss on answers and plot-word hit rates on fixed prompts. |
| `com.myllm.feedforward` | The position-wise two-layer feed-forward network (ReLU) inside each decoder block. |
| `com.myllm.generator` | Text generation with temperature and top-k sampling, and the interactive terminal chat loop. |
| `com.myllm.layernorm` | Layer normalisation with learned gain and bias. |
| `com.myllm.lora` | Low-rank adapters (LoRA) that fine-tune a frozen weight matrix through two small trainable matrices. |
| `com.myllm.optimizer` | Optimizers (SGD, Adam) and the warmup plus cosine learning-rate schedule. |
| `com.myllm.persist` | Binary file formats: saved models (`.bin`) and resumable training checkpoints (`.ckpt`). |
| `com.myllm.server` | Local HTTP server that serves the browser chat page and runs the model behind it. |
| `com.myllm.tokenize` | Tokenizers that turn text into ids and back: one token per character, or learned byte-pair encoding (BPE). |
| `com.myllm.train` | Numerical gradient checking, used by the tests to verify the hand-written backward passes. |
| `com.myllm.trainer` | Training loops: random windows for pretraining, whole chat examples for supervised fine-tuning, single or batched. |
| `com.myllm.transformer` | The full decoder-only model: embeddings, the stack of decoder blocks, final norm and output projection. |
| `com.myllm.utils` | Small matrix helpers shared by the layers: column concatenation and in-place addition. |
<!-- END GENERATED: packages -->

## Tests and static checks

```
./gradlew test            # unit tests
./gradlew lint            # static checks only, no tests
./gradlew spotlessApply   # reformat every Java file
./gradlew build           # compile + lint + tests: what has to pass before a change is done
```

The setup follows Apache Kafka's build (Checkstyle with a curated rule set, plus a bug-pattern checker) and
Google's Error Prone as used by Guava and Caffeine:

- **javac `-Xlint:all`**: every compiler warning fails the build (the unavoidable "incubating module" warning
  for the Vector API is the one exception)
- **Error Prone**: compile-time bug patterns (missing `@Override`, default charset, `Enum.ordinal()` in file
  formats, boxing comparators, ...). Two checks are off, with the reason in `build.gradle`
- **Checkstyle**: naming, imports, braces, whitespace, complexity limits; rules in
  `config/checkstyle/checkstyle.xml`. Math names like `d_Model` and `T_max` are allowed on purpose
- **Security**: SpotBugs with Find Security Bugs (OWASP) on the compiled classes, reporting only the SECURITY
  category (injection, path traversal, unsafe deserialization, weak crypto, ...), plus Error Prone's
  `BanSerializableRead`, `BanJNDI` and `BanClassLoader`. Accepted findings, each with its reason, are in
  `config/spotbugs/exclude.xml`; the report is `build/reports/spotbugs/main.html`
- **Comments and Javadoc**: rules in `CONTRIBUTING.md`, enforced by Checkstyle and javadoc doclint
- **Spotless + palantir-java-format**: one layout for every file (4 spaces, 120 columns, sorted imports);
  `spotlessApply` fixes it, `lint` only checks

A deliberate exception is suppressed where it happens, with a comment that says why, e.g.
`@SuppressWarnings("checkstyle:MethodLength")` on `Main.runTraining`.
