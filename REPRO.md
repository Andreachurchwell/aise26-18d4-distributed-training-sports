# REPRO.md

## Overview
This project demonstrates a scale-ready training loop with:
- single-process training
- Distributed Data Parallel (DDP) support
- gradient accumulation
- metric logging for scaling evidence

The focus is on **distributed training mechanics**, not model accuracy.

---

## Environment

### Local Development
- OS: Windows
- Python: 3.10
- PyTorch: >= 2.x
- Launch mode: single-process only

### Distributed Training Evidence
- Platform: Google Colab (Linux)
- Python: 3.10
- PyTorch: >= 2.x
- Launch mode: torchrun (multi-process)

---

## Install
```bash
pip install -r requirements.txt
```
Data
The training script expects:
- data/nba_team_games.csv

### This dataset is synthetically generated using a fixed NumPy random seed
- (np.random.default_rng(42)) to ensure reproducibility across runs.
Runs
Single-Process (Windows or Colab)
```
python train.py --cpu
```
### Expected behavior:
- Training runs successfully
- Step / loss values are printed
- A temporary metrics.csv is written to the repo root

Note: This root-level metrics.csv is an intermediate artifact and is not
committed. It is overwritten on each run.
Distributed Data Parallel (Linux / Colab)
```
torchrun --standalone --nproc_per_node=2 train.py --cpu
```
Expected behavior:
- Training runs with multiple processes
- Logged metrics show world_size = 2
- Effective batch size reflects gradient accumulation

Scaling Evidence
The committed scaling evidence includes:
- data/metrics.csv (final collected metrics)
- Screenshots under screenshots/ showing successful DDP execution
- These artifacts were generated on Linux (Google Colab).

Windows DDP Limitation
On my Windows environment, multi-process DDP failed with errors including:
- Gloo networking error: makeDeviceForInterface(): unsupported gloo device
- rendezvous / libuv initialization failures
- unexpected host reference: kubernetes.docker.internal
- These are known PyTorch + Windows limitations.

Mitigation
- The same DDP command was run successfully on Linux (Google Colab)
- Resulting metrics and screenshots are included as evidence
- The Colab notebook (aise26_18d4.ipynb) is included for transparency and auditability