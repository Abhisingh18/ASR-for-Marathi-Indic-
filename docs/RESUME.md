# Checkpoint resume mechanism

Training on shared, occasionally-unstable GPU hardware means runs get killed (reboots, OOM,
driver hangs) and need to resume mid-epoch without restarting the LR warmup or losing optimizer
state. Two things are restored on resume, both necessary: