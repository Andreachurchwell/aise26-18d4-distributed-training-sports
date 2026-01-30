# REPRO.md

## Environment
- Python 3.10
- PyTorch

## Reproducibility
- Fixed random seeds are used
- Training is run via a single-process loop first, then extended to DDP

## Notes
Commands and configuration details will be added as the training loop
is finalized.

## DDP Note (Windows)
Multi-process DDP launch fails on my Windows environment due to:
- torchrun rendezvous/libuv issues
- Gloo networking error: `makeDeviceForInterface(): unsupported gloo device`
- environment interference referencing `kubernetes.docker.internal`

Mitigation: run the same DDP command on Linux (Colab or WSL) and commit the
resulting `metrics.csv` as the scaling evidence artifact.