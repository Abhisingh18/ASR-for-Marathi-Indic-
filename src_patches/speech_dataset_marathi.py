import os.path as osp
import random
import json, yaml
import copy

import numpy as np
from scipy import signal
import soundfile as sf

import torch
import torchaudio
from torch.utils.data import Dataset
try:
    import whisper as _whisper_mod
except ImportError:
    _whisper_mod = None  # openai-whisper not installed; raw input_type still works
from slam_llm.utils.compute_utils import calculate_output_length_1d

# as Assamese - অসমীয়া, bn Bangla - বাংলা, brx Boro - बड़ो, gu Gujarati - ગુજરાતી, hi Hindi - हिंदी,
# kn Kannada - ಕನ್ನಡ, ks Kashmiri - كٲشُر, gom Konkani Goan - कोंकणी, mai Maithili - मैथिली,
# ml Malayalam - മലയാളം, mni Manipuri - ꯃꯤꯇꯩꯂꯣꯟ, mr Marathi - मराठी, ne Nepali - नेपाली,
# or Oriya - ଓଡ଼ିଆ, pa Panjabi - ਪੰਜਾਬੀ, sa Sanskrit - संस्कृतम्, sd Sindhi - سنڌي, si Sinhala - සිංහල,
# ta Tamil - தமிழ், te Telugu - తెలుగు, ur Urdu - اُردُو
_LANG_DISPLAY = {
    "as": "Assamese",
    "bn": "Bangla",
    "brx": "Boro",
    "gu": "Gujarati",
    "hi": "Hindi",
    "kn": "Kannada",
    "ks": "Kashmiri",
    "gom": "Konkani",
    "mai": "Maithili",
    "ml": "Malayalam",
    "mni": "Manipuri",
    "mr": "Marathi",
    "ne": "Nepali",
    "or": "Oriya",
    "pa": "Panjabi",
    "sa": "Sanskrit",
    "sd": "Sindhi",
    "si": "Sinhala",
    "ta": "Tamil",
    "te": "Telugu",
    "ur": "Urdu",
    "en": "English",
    "hi-en": "Hindi-English",
    "kn-en": "Kannada-English",
    "mai-en": "Maithili-English",
    "mni-en": "Manipuri-English",
    "mr-en": "Marathi-English",
}


def _lang_display(code):
    if not code:
        return ""
    return _LANG_DISPLAY.get(code, code)


def _lang_note(code):
    if code == "hi-en":
        return " This utterance code-switches between Hindi and English."
    if code == "kn-en":
        return " This utterance code-switches between Kannada and English."
    if code == "mai-en":
        return " This utterance code-switches between Maithili and English."
    if code == "mni-en":
        return " This utterance code-switches between Manipuri and English."
    if code == "mr-en":
        return " This utterance code-switches between Marathi and English."
    return ""


class SpeechDatasetJsonl(torch.utils.data.Dataset):
    
    def __init__(self,
                 dataset_config,
                 tokenizer=None,
                 split='train',
                 ):
        super().__init__()
        self.dataset_config = dataset_config
        self.tokenizer = tokenizer
        # data_parallel_size = dist.get_world_size()
        data_parallel_size = 1
        
        # self.data_list = contents
        self.IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        self.prompt = dataset_config.get("prompt", None)
        # MaLa-ASR historical context (see examples/asr_librispeech): when
        # use_history_context is set, EVERY utterance uses the same prompt; its
        # {prev_context} slot is filled per-sample from prev_context (empty string
        # when the sample has no prior context). No per-sample prompt selection.
        self.use_history_context = dataset_config.get("use_history_context", False)
        self.mel_size = dataset_config.get("mel_size", 80) # 80 for whisper large v1 and v2, 128 for large v3
        # self.prompt_library = [
        #     "Begin by converting the spoken words into written text. ",
        #     "Can you transcribe the speech into a written format? ",
        #     "Focus on translating the audible content into text. ",
        #     "Transcribe the speech by carefully listening to it. ",
        #     "Would you kindly write down the content of the speech? ",
        #     "Analyze the speech and create a written transcription. ",
        #     "Engage with the speech to produce a text-based version. ",
        #     "Can you document the speech in written form? ",
        #     "Transform the spoken words into text accurately. ",
        #     "How about putting the speech's content into writing? "
        # ]
        _prompt_style = dataset_config.get("prompt_style", "vicuna")
        if _prompt_style == "gemma2":
            self.prompt_template = "<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n"
            if self.tokenizer is not None and "<end_of_turn>" in self.tokenizer.all_special_tokens:
                self.eot_token_id = self.tokenizer.convert_tokens_to_ids("<end_of_turn>")
            else:
                self.eot_token_id = None
        elif _prompt_style == "qwen2":
            self.prompt_template = (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n{}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            self.eot_token_id = None
        elif _prompt_style == "qwen3_5":
            # Copied from speech_dataset_qwen.py (see there for the full rationale):
            # ChatML with a generic system turn and an empty <think> block (non-
            # thinking mode). Training targets end with tokenizer.eos_token_id,
            # which for this tokenizer IS <|im_end|>, so eot_token_id mirrors that
            # (only ever assigned, never read, same as the other branches here).
            self.prompt_template = (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n{}<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            )
            self.eot_token_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        elif _prompt_style == "airavata":
            # ai4bharat/Airavata: tulu-style chat template. Tokenizer prepends <s> BOS
            # automatically; answer is terminated with </s> via tokenizer.eos_token_id,
            # so no custom eot is needed here.
            self.prompt_template = "<|user|>\n{}\n<|assistant|>\n"
            self.eot_token_id = None
        else:
            self.prompt_template = "USER: {}\n ASSISTANT:"
            self.eot_token_id = None
        self.answer_template = "{}"
        self.fix_length_audio = dataset_config.get("fix_length_audio", -1)
        # Frame-budget mode: when audio_frames_per_sec > 0 (raw input only), reserve
        # audio token slots from DURATION as round(fps * sec) // ds_rate instead of
        # the full encoder frame rate (samples//320 = 50 frames/s). The scatter in
        # slam_model then fills only these slots with the LEFTMOST encoder frames,
        # so the <fill>-heavy tail beyond the budget never reaches the LLM. ds_rate
        # here must match model_config.encoder_projector_ds_rate.
        self.audio_frames_per_sec = dataset_config.get("audio_frames_per_sec", -1.0)
        self.encoder_projector_ds_rate = dataset_config.get("encoder_projector_ds_rate", 5)
        # Some encoders' real frame count differs from the samples//320 formula
        # below (e.g. wav2vec2_bert's log-mel + stride-2 feature stacking is
        # consistently 1 frame SHORTER than samples//320, verified empirically
        # across multiple durations: 16000 samples -> 49 frames not 50, 59200
        # samples -> 184 not 185, 116800 samples -> 364 not 365). Without this,
        # the dataset reserves one MORE audio-token slot than the encoder
        # actually produces; slam_model.py's scatter clamps to the real length
        # but leaves that extra reserved slot as an all-zero embedding (neither
        # real audio nor the original text token) injected into every single
        # training example. Set via ++dataset_config.encoder_frame_offset=1 in
        # the wav2vec2_bert finetune script.
        self.encoder_frame_offset = dataset_config.get("encoder_frame_offset", 0)
        self.inference_mode = dataset_config.get("inference_mode", False)
        self.normalize = dataset_config.get("normalize", False)
        self.input_type = dataset_config.get("input_type", None)
        assert self.input_type in ["raw", "mel"], "input_type must be one of [raw, mel]" 