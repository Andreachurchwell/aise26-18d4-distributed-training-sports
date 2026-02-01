# SCALE_PLAN.md

## Project Overview
This project trains a binary classifier to predict whether an NBA team
wins or loses a game. Each training example represents one team’s
performance in a single game.

The goal is not model accuracy, but to demonstrate a training loop
that is **scale-ready** using data parallelism and gradient accumulation.

---

## Scaling Strategy

### Why Data Parallelism (DDP)
As additional seasons are added, the number of games grows linearly.
This increases dataset size but does not significantly increase
model size.

Distributed Data Parallel (DDP) is the natural first scaling strategy
because each worker can process different games independently while
synchronizing gradients.

---

### Why Gradient Accumulation
Memory constraints limit the number of team-game examples that can be
processed in a single batch.

Gradient accumulation is used to simulate a larger effective batch size
by accumulating gradients over multiple smaller micro-batches before
updating model weights.

---

## Effective Batch Size
Effective batch size is defined as:

micro_batch_size × accumulation_steps × world_size

Logging effective batch size allows comparison across different scaling
configurations and ensures consistent training behavior.

---

## When DDP Is No Longer Enough (Decision Triggers)

### Move to FSDP if memory is the blocker
Use FSDP if the model + optimizer state cannot fit comfortably on a single
device (OOM errors), because FSDP shards parameters/gradients/optimizer state.

### Consider TP/PP only if a single model layer cannot fit on one device
Tensor parallelism (TP) or pipeline parallelism (PP) are usually for very large
models (e.g., huge transformers). This project’s model is small, so TP/PP would
only be relevant if the model grows dramatically beyond single-device limits.
