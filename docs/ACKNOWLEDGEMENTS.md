# Acknowledgements

This work builds on, and would not exist without:

- **[SLAM-LLM](https://github.com/ddlBoJack/SLAM-LLM)** (Ziyang Ma et al., MIT license) -- the
  SLAM-ASR training/inference framework these scripts drive. This repo does not vendor the
  framework itself, only the Marathi-specific scripts and small patches layered on top of it.
- **SPRING-INX** (asr.iitm.ac.in/models) -- the data2vec-AQC and ccc-wav2vec2.0 Marathi
  fine-tuned/SSL encoder checkpoints.
- **OpenAI Whisper**, **CMU/ESPnet XEUS** -- the two pretrained-only multilingual encoders
  compared here.
- Colleagues at the lab who integrated the ccc-wav2vec2.0 encoder into the shared codebase and
  prepared the CDAC-side Marathi test-set data used for decoding.
