## DDP (Distributed Data Parallel)
- You make multiple copies of the same model
- Each copy sees different data
- After each step, they share gradients so weights stay identical

## Gradient Accumulation
- Your computer can only fit small batches in memory.
***Solution:***
- Instead of:
- update after every batch
- You:
- wait N batches
- add the gradients together
- update once
***That pretends you had a bigger batch.***
```
effective_batch_size = batch_size × accum_steps × world_size
```

## Scale-Ready
- This does NOT mean:
***“It scales to 100 GPUs”***
- It means:
***“If I did add GPUs, the structure wouldn’t be wrong.”***
- That’s why toy data is allowed.

## Why metrics exists
- step number
- loss
- world size
- accumulation steps
- effective batch size
***That’s evidence, not performance.***

## Why the Windows → Colab thing matters
In real life:
- distributed training breaks on Windows
- cloud/Linux is standard
- networking issues are common

By documenting:
- “DDP failed locally”
- “DDP succeeded on Linux”
- “Here are the metrics”

**I learned the structure of a scale-ready training loop, how gradient accumulation simulates larger batches, and why distributed training depends heavily on environment and networking.**