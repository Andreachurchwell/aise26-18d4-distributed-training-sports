# W18D4 – Scale-Ready Distributed Training (Sports Example)

## Goal
The goal of this assignment was to build a **scale-ready training loop** that:
- starts as a single-process script
- can be launched with **Distributed Data Parallel (DDP)**
- supports **gradient accumulation**
- produces **reviewable scaling evidence**, even when run locally

This repo uses a **sports-inspired tabular dataset** to predict a win outcome and focuses on
distributed training mechanics rather than model complexity.

---

## What This Project Does
- Trains a small MLP on tabular “NBA-style” game features
- Supports:
  - single-process CPU training
  - multi-process DDP (`world_size=2`)
  - gradient accumulation (`accum_steps`)
- Logs training evidence to `metrics.csv`

The model and data are intentionally simple so the focus stays on **distributed training correctness**.


---

## Files Overview
- `train.py` – Main training script (single-process + DDP-ready)
- `ddp_spawn.py` – Alternate spawn launcher (used during Windows testing)
- `metrics.csv` – Scaling evidence (includes `world_size` and effective batch size)
- `data/nba_team_games.csv` – Synthetic sports dataset used for training
- `SCALE_PLAN.md` – DDP vs FSDP vs TP/PP decision reasoning
- `REPRO.md` – Reproducibility instructions
- `KNOWN_ISSUES.md` – Distributed training failure modes and mitigations

## Repo Structure
```
aise26-18d4-distributed-training-sports/
├── train.py
├── ddp_spawn.py
├── metrics.csv
├── requirements.txt
├── README.md
├── REPRO.md
├── SCALE_PLAN.md
├── KNOWN_ISSUES.md
├── MY_NOTES.md 
├── data/
│   └── nba_team_games.csv 
│   └── metrics.csv         
└── screenshots/           
   └── colab_train.png
   └── torchrun_node2.png
```
---

## Key Results
- Single-process run logs `world_size = 1`
- DDP run logs `world_size = 2`
- Effective batch size increases correctly with gradient accumulation
- Evidence is captured in `metrics.csv`

---

## Notes on Environment
Multi-process DDP failed on my Windows environment due to known PyTorch
libuv / Gloo networking limitations.  
The same code ran successfully on **Linux (Google Colab)**, and the resulting
metrics are committed as the scaling evidence.

This reflects real-world distributed training constraints and is documented in
`KNOWN_ISSUES.md`.