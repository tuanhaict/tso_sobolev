import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

LOG_DIR = Path("logs")           # adjust if needed

# ---- If you changed PRINT_STEPS in the training script, update the list below ----
CHECKPOINT_STEPS = {0, 19, 39, 59, 79, 99}
      # ordered exactly as they were recorded
# ----------------------------------------------------------------------------------

def parse_tag(fname: str):
    """Extract loss_type, lr, seed from a filename."""
    m = re.search(r"(?P<loss>[a-z_]+)_lr(?P<lr>[0-9.eE+-]+)_seed(?P<seed>\d+)", fname)
    if m is None:
        raise ValueError(f"Cannot parse run tag from {fname}")
    return m["loss"], float(m["lr"]), int(m["seed"])

records = []
for dist_path in glob.glob(str(LOG_DIR / "*_distances.txt")):
    loss, lr, seed = parse_tag(dist_path)
    distances      = np.loadtxt(dist_path, delimiter=",", ndmin=1)

    if len(distances) != len(CHECKPOINT_STEPS):
        raise ValueError(
            f"{dist_path} has {len(distances)} checkpoints, but "
            f"CHECKPOINT_STEPS has {len(CHECKPOINT_STEPS)} entries."
        )

    for step, d in zip(CHECKPOINT_STEPS, distances):
        records.append(
            {"loss_type": loss, "lr": lr, "seed": seed,
             "step": step, "distance": d}
        )

df = pd.DataFrame(records)

# ----- aggregate over seeds, pivot so each step becomes a pair of columns ---------------
agg = (df.groupby(["loss_type", "lr", "step"])["distance"]
         .agg(["mean"])
         .reset_index())

# build a wide table with columns: dist_step<k>_mean
wide = agg.pivot_table(index=["loss_type", "lr"],
                       columns="step",
                       values=["mean"])
# flatten the MultiIndex columns
wide.columns = [f"dist_step{step}_{stat}"
                for stat, step in wide.columns]

wide = wide.reset_index().sort_values(["lr", "loss_type"])

# pretty-print
# pd.set_option("display.float_format", lambda x: f"{x:8.4f}")
pd.set_option("display.float_format", lambda x: f"{x:.4e}")
print("\n=== DISTANCE - MEAN ± STD AT EACH CHECKPOINT ===")
for lr_val, sub in wide.groupby("lr", sort=True):
    print(f"\n=== LEARNING RATE {lr_val:g} ===")
    # drop the redundant 'lr' column before printing the block
    print(sub.drop(columns="lr").to_string(index=False))


# optional: write to CSV
# wide.to_csv("checkpoint_summary.csv", index=False)
