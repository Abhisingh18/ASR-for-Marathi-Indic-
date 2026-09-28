#!/usr/bin/env python3
"""Rewrite the 7 Marathi decode-testset jsonls' "source" field from CDAC's
absolute path prefix to the local (gpu18) audio path, after the audio has
been rsync'd over (see the two-hop "speech" machine bridge steps).

Input:  /speech/abhishek/marathi_data/decode_test/data_ENMR/<name>_output_prev.jsonl
Output: /speech/abhishek/marathi_data/decode_test/data_ENMR/<name>_output_prev.local.jsonl
(original files are left untouched)

Run after the audio rsync (Step 2 into .../decode_test/audio/) finishes:
    /speech/abhishek/miniconda3/envs/slam_llm/bin/python3 rewrite_marathi_testset_paths.py
"""
import json
import os

CDAC_PREFIX = "/nlsasfs/home/dibd/dibd-speech/iitm/nithyar/CLEAN/SLAM_ASR/BILING/dump_slam30/raw/"
LOCAL_PREFIX = "/speech/abhishek/marathi_data/decode_test/audio/"
TEST_DIR = "/speech/abhishek/marathi_data/decode_test/data_ENMR"

TESTSETS = [
    "commonvoice_marathi_test",
    "fleurs_marathi_test",
    "indictts_marathi_test",
    "kathbath_marathi_test",
    "kathbath_noisy_marathi_test",
    "mucs_marathi_test",
    "R12_marathi_eval_filtered",
]

for name in TESTSETS:
    src_path = os.path.join(TEST_DIR, f"{name}_output_prev.jsonl")
    dst_path = os.path.join(TEST_DIR, f"{name}_output_prev.local.jsonl")
    if not os.path.isfile(src_path):
        print(f"[SKIP] {src_path} not found")
        continue

    n, missing, first_missing = 0, 0, None
    with open(src_path) as fin, open(dst_path, "w") as fout:
        for line in fin:
            row = json.loads(line)
            if not row["source"].startswith(CDAC_PREFIX):
                raise ValueError(f"{name}: unexpected source prefix: {row['source']}")
            row["source"] = LOCAL_PREFIX + row["source"][len(CDAC_PREFIX):]
            if not os.path.isfile(row["source"]):
                missing += 1
                first_missing = first_missing or row["source"]
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1

    status = "OK" if missing == 0 else f"MISSING {missing}/{n} audio files (e.g. {first_missing})"
    print(f"{name}: {n} rows -> {dst_path}  [{status}]")
