# main_point.py
"""
Quick usage
-----------
```bash
# run on every visible GPU
python main_point.py --parallel

# select GPUs 0 and 2 only
python main_point.py --parallel --gpus 0 2
```
"""

from __future__ import annotations

import argparse
import itertools
import os
import random
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.multiprocessing as mp

from utils import SW, compute_true_Wasserstein  # utils.py
from tsw import TWConcurrentLines, generate_trees_frames  # Treed SW implementation
from n_tsw import NTWConcurrentLines  # n-TSW implementation
from osb_tsw import OSb_TSConcurrentLines  # GST implementation
# ------------------------------ default hyper‑parameters -----------------------------------------
DEFAULT_LOSS_TYPES: List[str] = [
    "sw",        # sliced Wasserstein
    "twd",       # Treed Wasserstein (Gaussian directions)
    "fw_twd",    # FW‑TWD (geometric‑median intercept)
    "fw_twd_rp", # FW‑TWD with random‑path dirs
]
DEFAULT_LRS: List[float] = [1e-2]
DEFAULT_SEEDS: List[int] = [0,1,2,3,4]

# Treed Wasserstein structure
NLINES   = 4
L_TOTAL  = 100                       # directions for SW; also NTREES * NLINES for TWD
NTREES   = L_TOTAL // NLINES
DIM      = 3
STD      = 0.1                       # Gaussian sampling std for directions

# Optimisation schedule
NSTEPS       = 500
PRINT_STEPS  = {0, 99, 199, 299, 399, 499}#, 349, 399, 449, 499}
# PRINT_STEPS  = {0, 49, 99, 149, 199}

# NSTEPS       = 500
# PRINT_STEPS  = {0, 99, 199, 299, 399, 499}

# ShapeNet indices
IND_TARGET = 21
IND_SOURCE = 8

# --------------------------------------------------------------------------------------------------

def ensure_dirs(*dirs: str) -> None:
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def build_twd_obj(device: torch.device) -> TWConcurrentLines:
    """Return a *plain* TWConcurrentLines object (no `torch.compile`).

    `torch.compile` spawns its own background compilation pool, which is forbidden
    inside multiprocessing *daemon* workers and triggers the
    "daemonic processes are not allowed to have children" assertion you saw.
    Using the raw module avoids that issue while having negligible impact on
    these small point‑cloud models.
    """
    obj = TWConcurrentLines(
        ntrees=NTREES,
        nlines=NLINES,
        mass_division="distance_based",
        device=device,
        p=1,
        delta=10,
    )
    return obj

def build_gst_obj(device: torch.device, n_function: str = "exp", p_agg: int =2) -> OSb_TSConcurrentLines:
    """Return a *plain* OSb_TSConcurrentLines object (no `torch.compile`).

    `torch.compile` spawns its own background compilation pool, which is forbidden
    inside multiprocessing *daemon* workers and triggers the
    "daemonic processes are not allowed to have children" assertion you saw.
    Using the raw module avoids that issue while having negligible impact on
    these small point‑cloud models.
    """
    obj = OSb_TSConcurrentLines(
        ntrees=NTREES,
        nlines=NLINES,
        mass_division="distance_based",
        device=device,
        p=2,
        n_function=n_function,
        p_agg=p_agg,
        delta=10,
    )
    return obj

def build_ntwd_obj(device: torch.device, noisy_mode=None, lambda_=0.0, p_noise=2) -> NTWConcurrentLines:
    """Return a *plain* n‑TSW object (no `torch.compile`).

    `torch.compile` spawns its own background compilation pool, which is forbidden
    inside multiprocessing *daemon* workers and triggers the
    "daemonic processes are not allowed to have children" assertion you saw.
    Using the raw module avoids that issue while having negligible impact on
    these small point‑cloud models.
    """
    obj = NTWConcurrentLines(
        p=1,
        delta=10,
        mass_division="distance_based",
        device=device,
        noisy_mode=noisy_mode,
        lambda_=lambda_,
        p_noise=p_noise
    )
    return obj
def loss_fn(
    loss_type: str,
    X: torch.Tensor,
    Y: torch.Tensor,
    twd_obj: TWConcurrentLines | None | NTWConcurrentLines,
    step: int,
) -> torch.Tensor:
    if loss_type == "sw":
        return SW(X=X, Y=Y, L=L_TOTAL, p=2, device=Y.device)

    assert twd_obj is not None, "TWD object required for TWD‑based losses"

    progress = step / NSTEPS
    kappa = 30

    mean_local = Y.mean(dim=0)
    rp = loss_type.endswith("_rp")
    gen_mode = "random_path" if rp else "gaussian_raw"

    common = dict(
        ntrees=NTREES, nlines=NLINES, d=DIM, std=STD, device=Y.device,
        kappa=kappa if rp else None, X=X.detach(), Y=Y.detach()
    )

    if loss_type.startswith("fw_"):
        theta, intercept = generate_trees_frames(
            intercept_mode="geometric_median", gen_mode=gen_mode, **common)
    elif loss_type.startswith("n_tsw"):
        theta, intercept = generate_trees_frames(
            mean=mean_local, gen_mode=gen_mode, **common)        
    else:
        theta, intercept = generate_trees_frames(
            mean=mean_local, gen_mode=gen_mode, **common)

    return twd_obj(X, Y, theta, intercept)

# --------------------------------------------------------------------------------------------------

def run_one(args, loss_type: str, lr: float, gpu_id: int, data_path: str) -> None:
    """Worker: run one (loss, lr, seed) combo on the chosen GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)

    w2_dist = []
    for seed in DEFAULT_SEEDS:
        # deterministic seed per run
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed)

        # data
        arr = np.load(data_path)
        Y = torch.tensor(arr[IND_TARGET], device=device)
        X = torch.tensor(arr[IND_SOURCE], requires_grad=True, device=device)
        N = Y.shape[0]
        twd_obj = None if loss_type == "sw" else build_ntwd_obj(device, noisy_mode=args.noisy_mode, lambda_=args.lambda_, p_noise=args.p_noise) if loss_type.startswith("n_tsw") else build_gst_obj(device, n_function=args.n_function, p_agg=args.p_agg) if loss_type.startswith("gst") else build_twd_obj(device)
        opt = torch.optim.Adam([X], lr=lr)

        traj, dists, times = [], [], []
        t0 = time.time()
        for step in range(NSTEPS):
            if step in PRINT_STEPS:
                dist = compute_true_Wasserstein(X, Y)
                elapsed = time.time() - t0
                print(
                    f"GPU{gpu_id} | {loss_type.upper():8s} lr={lr:<6g} seed={seed} "
                    f"step {step+1:3d}/{NSTEPS} dist {dist:.4e} time {elapsed:.1f}s"
                )
                traj.append(X.detach().cpu().numpy())
                dists.append(dist)
                times.append(elapsed)

            opt.zero_grad()
            (loss_fn(loss_type, X, Y, twd_obj, step)).backward()
            opt.step()

        traj.append(Y.detach().cpu().numpy())

        tag = f"{loss_type}_lr{lr:g}_src{IND_SOURCE}_tgt{IND_TARGET}"
        np.save(f"saved/{tag}_seed{seed}_points.npy", np.stack(traj))
        np.savetxt(f"logs/{tag}_seed{seed}_distances.txt", np.array(dists), delimiter=",")
        w2_dist.append(dists)
    
    mean_val = np.mean(w2_dist, axis=0)
    std_val = np.std(w2_dist, axis=0)
    out = np.column_stack((mean_val, std_val))
    np.savetxt(f"logs/mean_std_{tag}.txt", out, delimiter=",", header="mean,std", comments='')

    # for Overleaf
    with open(f"logs/mean_std_{tag}.txt", "a") as f:
        line = " & ".join(f"{x:.2e}" for x in mean_val)
        f.write("\n" + line)

# --------------------------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Parallel Wasserstein launcher")
    p.add_argument("--data", default="reconstruct_random_50_shapenetcore55.npy",
                   help="Path to the .npy point‑cloud file")
    p.add_argument("--losses", nargs="*", default=DEFAULT_LOSS_TYPES,
                   help="Loss types to include")
    p.add_argument("--lrs", type=float, nargs="*", default=DEFAULT_LRS,
                   help="Learning rates to sweep")
    p.add_argument("--parallel", action="store_true",
                   help="Enable multi‑GPU execution (sequential if omitted)")
    p.add_argument("--gpus", type=int, nargs="*",
                   help="GPUs to use (default: all visible GPUs)")
    p.add_argument("--runs_per_gpu", type=int, default=1,
                   help="Number of runs per GPU")
    p.add_argument("--noisy_mode", type=str, default=None,
                   help="Type of noise regularization for n-TSW: None | interval | ball")
    p.add_argument("--lambda_", type=float, default=0.0,
                   help="Regularization strength for n-TSW")
    p.add_argument("--p_noise", type=int, default=2,
                   help="Dual norm exponent for noise regularization in n-TSW")
    p.add_argument("--p_agg", type=float, default=2.0,
                   help="Aggregation exponent for n-TSW loss")
    p.add_argument("--n_function", type=str, default="exp",
                   help="Choice of n-function for GST: exp | identity")
    return p.parse_args()

# --------------------------------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    ensure_dirs("saved", "logs")

    combos = list(itertools.product(args.losses, args.lrs))

    # ---------- sequential (default) ------------------------------------------------------
    if not args.parallel:
        devs = args.gpus if args.gpus is not None else [0]
        for idx, (loss, lr) in enumerate(combos):
            run_one(args, loss, lr, devs[idx % len(devs)], args.data)
        return

    # ---------- parallel with spawn -------------------------------------------------------
    if not torch.cuda.is_available():
        raise RuntimeError("--parallel requires at least one CUDA device")

    all_gpus = list(range(torch.cuda.device_count()))
    devs = args.gpus if args.gpus is not None else all_gpus
    if not devs:
        raise ValueError("No GPUs specified and no CUDA devices detected")

    devs = devs * args.runs_per_gpu

    tasks = [(args, *combo, devs[i % len(devs)], args.data)
         for i, combo in enumerate(combos)]


    print(f"Launching {len(tasks)} runs on GPUs {devs} (spawn context)")
    ctx = mp.get_context("spawn")
    with ctx.Pool(len(devs)) as pool:
        pool.starmap(run_one, tasks)

# --------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # safe even if already 'spawn'
    main()
