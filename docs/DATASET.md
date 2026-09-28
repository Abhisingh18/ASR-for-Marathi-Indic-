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