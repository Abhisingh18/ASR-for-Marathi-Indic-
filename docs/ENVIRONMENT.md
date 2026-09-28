# Environment

- Conda env with PyTorch 2.4.1+cu124, DeepSpeed 0.19.0, `transformers` (Gemma-3/Sarvam-1 support),
  `fairseq` (for the data2vec-AQC / ccc-wav2vec2 encoders), `openai-whisper`, ESPnet (for XEUS).
- CUDA 12.4 toolkit (DeepSpeed's fused AdamW needs `nvcc` at build time).
- 4x NVIDIA RTX 6000 Ada Generation (49GB) per training/decode run.
- `CUDA_DEVICE_ORDER=PCI_BUS_ID` set everywhere, so CUDA indices match `nvidia-smi` indices --
  important on multi-GPU hosts shared with other users' jobs.
