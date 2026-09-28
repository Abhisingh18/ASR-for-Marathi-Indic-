# Changelog

## Encoder comparison study, Marathi+English SLAM-ASR

- Started with the data2vec-AQC (Marathi CTC fine-tuned) + Gemma-3-4B-IT baseline: 1 epoch,
  effective batch 32, LoRA r=8/alpha=32.
- Added the multilingual-SSL (pretrained-only) data2vec-AQC variant as a controlled ablation --
  same everything, encoder fine-tuning status is the only difference.
- Added XEUS and Whisper large-v3 as pretrained-only encoders from different families
  (ESPnet-SSL and Whisper's supervised multitask training, respectively).
- Added a Sarvam-1 LLM run on the best (Marathi-FT data2vec-AQC) encoder, to separate
  "encoder matters" from "LLM matters".
- Added ccc-wav2vec2.0 (a second Marathi-fine-tuned encoder family, fairseq wav2vec2 architecture)
  for a same-domain-fine-tuning comparison against data2vec-AQC.
- Fixed a training-resume bug: resuming only restored LoRA/projector weights, not the DeepSpeed
  optimizer/LR-scheduler state, causing resumed runs to silently re-warm LR from 0 (see
  `docs/RESUME.md`).
- Fixed a decode-script bug: the first version of the Marathi inference script omitted the
  `model_config.file` override, silently falling back to the framework's default (pre-"_new")
  model factory instead of the one actually used to train the checkpoint being decoded.
- Ran all 6 comparison models' epoch-1 (step 32,000) checkpoints through the same 7-test-set
  decode + WER pipeline for a controlled comparison (`results/SUMMARY.md`).
