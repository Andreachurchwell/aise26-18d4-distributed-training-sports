import os
import pandas as pd
import matplotlib.pyplot as plt

# Prefer data/metrics.csv (your real one), fallback to root metrics.csv
path = "data/metrics.csv" if os.path.exists("data/metrics.csv") else "metrics.csv"

df = pd.read_csv(path)

# Basic sanity check
required = {"step", "loss", "effective_batch_size", "world_size", "accum_steps"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns in {path}: {missing}")

df = df.sort_values("step")

plt.figure()
plt.plot(df["step"], df["loss"])
plt.xlabel("Step")
plt.ylabel("Loss")
ws = df["world_size"].iloc[0]
acc = df["accum_steps"].iloc[0]
ebs = df["effective_batch_size"].iloc[0]
plt.title(f"Training Loss vs Step (world_size={ws}, accum_steps={acc}, eff_batch={ebs})")

os.makedirs("screenshots", exist_ok=True)
out_path = "screenshots/loss_curve.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"Saved: {out_path}")