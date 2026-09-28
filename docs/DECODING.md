# Decoding pipeline

Each of the 6 decode scripts (`scripts/decoding/inference_*.sh`) runs one model's checkpoint over
all 7 test sets on a single GPU, with a 5-attempt retry loop per test set (the shared cluster
occasionally throws a transient CUDA fault mid-`generate()`).

Steps to reproduce: