import glob, json, os, torch, numpy as np
from policy import *; from enc import Enc
"""Held-out test: 3 fresh seeds x 512 tracks, never used for selection. best.pt was selected on seed 999, so last.pt is reported as well."""
R = "/data1/j50057870/world-model/runs"; enc = Enc().cuda(); SE = (70001, 70002, 70003); out = {}
def run(name, ag, **k):
    rs = [evaluate(ag, enc, n=512, seed=s, **k) for s in SE]
    out[name] = {m: (float(np.mean([r[m] for r in rs])), float(np.std([r[m] for r in rs]))) for m in rs[0]}
    print(f"{name:26s}", {m: f"{a:.3f}+-{b:.3f}" for m, (a, b) in out[name].items()}, flush=True)
run("random", None, kind="random")
for v in (4., 6., 7., 8.): run(f"expert_{v:g}mps", None, kind="expert", v=v)
run("bc", load_agent(f"{R}/bc0/bc.pt"))
for d in sorted(glob.glob(f"{R}/ppo_*") + glob.glob(f"{R}/img_*")):
    if os.path.isdir(d):
        for c in ("best", "last"):
            if os.path.exists(f"{d}/{c}.pt"): run(f"{os.path.basename(d)}/{c}", load_agent(f"{d}/{c}.pt"))
json.dump(out, open(f"{R}/final_test.json", "w"), indent=1)
