# Dataset

**Training/dev data:** Marathi+English bilingual code-switch, one JSONL row per utterance:

```json
{"key": "...", "source": "/path/to/audio.wav", "target": "transcript",
 "prev_context": "previous utterance's transcript (or empty)", "language": "mr|en|mr-en"}
```

| Split | Utterances | Approx. hours | mr / en / mr-en |
|---|---|---|---|
| train | 526,134 | ~1,180 | 69% / 22% / 9% |
| dev | 26,413 | -- | similar mix |

Audio: 16kHz mono wav, 0.3s-27s (median ~7.8s). Rows tagged `mr-en` get an extra prompt note
("This utterance code-switches between Marathi and English.") -- see
`src_patches/speech_dataset_marathi.py`.

## Test sets (decoding)

Seven held-out benchmark test sets, pulled from a separate CDAC-side data prep run:

| Test set | Utterances | Source |
|---|---|---|
| commonvoice | 1,205 | Mozilla Common Voice, Marathi |
| fleurs | 1,015 | Google FLEURS, Marathi |
| indictts | 100 | IndicTTS, Marathi |
| kathbath | 1,631 | AI4Bharat Kathbath, Marathi |
| kathbath_noisy | 1,631 | Kathbath, noise-augmented |
| mucs | 636 | MUCS (Multilingual and Code-Switching ASR Challenge), Marathi |
| R12_eval_filtered | 6,282 | in-house eval set (filtered) |
