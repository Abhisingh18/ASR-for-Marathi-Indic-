# Glossary

- **WER (Word Error Rate)** — (substitutions + deletions + insertions) / reference word count.
  Lower is better. The primary metric used throughout `results/`.
- **CER (Character Error Rate)** — same idea, at the character level. Reported alongside WER in
  each `wer_*` file (useful for Marathi, where word-boundary conventions vary).
- **SER (Sentence Error Rate)** — fraction of utterances with at least one error. A model can
  have a low WER but a high SER if errors are spread across many short utterances.
- **LoRA (Low-Rank Adaptation)** — the only trainable part of each frozen LLM here: two small
  rank-`r` matrices per target projection, added to (not replacing) the frozen weight, r=8/alpha=32
  throughout this repo.
- **ZeRO (Zero Redundancy Optimizer)** — DeepSpeed's optimizer-state sharding; stage 2 here shards
  optimizer states + gradients (not parameters) across the GPUs used in a run.
- **SLAM-ASR** — the architecture family this repo's scripts follow: frozen speech encoder ->
  small trainable projector -> frozen LLM, adapted only through the projector and LoRA.
- **Code-switching** — an utterance mixing two languages (here, Marathi and English) within
  a single sentence; tagged `mr-en` in this dataset and given an explicit prompt note.
