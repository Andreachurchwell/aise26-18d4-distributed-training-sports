## Issue: Windows torch.distributed launch fails (libuv + gloo)

**Symptoms**
- `torchrun` / `python -m torch.distributed.run` fails with:
  `use_libuv was requested but PyTorch was built without libuv support`
- Spawned DDP init fails with:
  `RuntimeError: makeDeviceForInterface(): unsupported gloo device`
- Attempts may try connecting to `kubernetes.docker.internal:29501`

**Mitigations**
- Run DDP on Linux (Colab or WSL) where rendezvous + gloo are supported.
- Keep the code DDP-ready (process group init, DistributedSampler, DDP wrap),
  and log metrics (`world_size`, `effective_batch_size`) to validate scaling.