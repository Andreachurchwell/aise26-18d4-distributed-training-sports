## Issue #1: Windows torch.distributed launch fails (libuv + gloo)

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

## Issue #2: DDP init can hang if ranks/ports/env are misconfigured (common failure mode)

**What it looks like**
- Run appears to "freeze" (no training output)
- One process prints, the other does not
- Stuck before training starts

**Why it happens**
- `MASTER_ADDR/MASTER_PORT/RANK/WORLD_SIZE` mismatch
- Port already in use from a previous run
- One worker crashes, the other waits forever

**Mitigation**
- Prefer `torchrun --standalone` (auto-sets env correctly).
- If re-running multiple times, change the port or restart runtime.
- Add early logging of `rank` and `world_size` to confirm processes started.

---

## Issue #3: Incorrect gradient accumulation changes effective batch behavior (common failure mode)

**What it looks like**
- Effective batch size doesn’t match the intended calculation
- Loss behavior differs unexpectedly between runs (single vs DDP)

**Why it happens**
- Loss not scaled by `accum_steps`
- Optimizer step happens every batch instead of every `accum_steps`

**Mitigation**
- Use the standard pattern:
  - divide loss by `accum_steps` before backward
  - call `optimizer.step()` only every `accum_steps`
- Log `effective_batch_size = micro_batch_size * accum_steps * world_size`
  (this project records that in metrics.csv)