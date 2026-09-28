# mucs_marathi_test

MUCS (Multilingual and Code-Switching ASR Challenge), Marathi. 636 utterances. The one test set
where the ranking flips: Whisper+Gemma3 (26.99%) beats data2vec-FT+Gemma3 (29.24%), and
Sarvam-1+FT does noticeably worse than every other model (43.62%) -- worth investigating further
(possibly a domain mismatch between MUCS audio and Sarvam-1's LLM training distribution).
