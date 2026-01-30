# REPRO.md

## Environment
- Python 3.10
- PyTorch

## Reproducibility
- Fixed random seeds are used
- Training is run via a single-process loop first, then extended to DDP


## DDP Note (Windows)
Multi-process DDP launch fails on my Windows environment due to:
- torchrun rendezvous/libuv issues
- Gloo networking error: `makeDeviceForInterface(): unsupported gloo device`
- environment interference referencing `kubernetes.docker.internal`

Mitigation: run the same DDP command on Linux (Colab or WSL) and commit the
resulting `metrics.csv` as the scaling evidence artifact.

## Linux / Colab Repro

```bash
pip install -r requirements.txt
python train.py --cpu
torchrun --standalone --nproc_per_node=2 train.py --cpu
``