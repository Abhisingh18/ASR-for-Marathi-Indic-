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