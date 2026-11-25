import sys, glob, json, time, os, numpy as np, torch, torch.nn.functional as F
from policy import *; from enc import Enc
"""bc.py OUT DATAGLOB STEPS [init]: regress expert action at every visited valid state. Also linear probe of gate pose."""
out, gl, steps = sys.argv[1], sys.argv[2], int(sys.argv[3]); D = "/data1/j50057870/world-model/data"
S = [{k: np.load(f"{d}/{k}.npy", mmap_mode="r" if k == "feat" else None) for k in ("feat", "prop", "priv", "exp", "valid")} for d in sorted(glob.glob(f"{D}/{gl}"))]
V = {k: np.load(f"{D}/val/{k}.npy", mmap_mode="r" if k == "feat" else None) for k in ("feat", "prop", "priv", "exp", "valid")}
I = np.concatenate([np.concatenate([np.full((int(x["valid"].sum()), 1), j), np.argwhere(x["valid"])], 1) for j, x in enumerate(S)]); rng = np.random.default_rng(0)
def bat(n):
    i = I[rng.integers(len(I), size=n)]; return [torch.from_numpy(np.stack([S[j][k][a, b] for j, a, b in i])) for k in ("feat", "prop", "priv", "exp")]
import threading, queue
q = queue.Queue(16)
def w():
    while True: q.put(bat(512))
for _ in range(6): threading.Thread(target=w, daemon=True).start()
ag = Agent().cuda(); f = np.concatenate([np.asarray(x["feat"][:64, ::8], np.float32).reshape(-1, 384) for x in S]); ag.mu.copy_(torch.tensor(f.mean(0))); ag.sd.copy_(torch.tensor(f.std(0)) + 1e-3)
pr = Net(6).cuda(); opt = torch.optim.AdamW([*ag.pi.parameters(), *pr.parameters()], 5e-4, weight_decay=.01); sc = torch.amp.GradScaler("cuda"); t0 = time.time()
for st in range(steps):
    for g in opt.param_groups: g["lr"] = 5e-4 * min(1, (st + 1) / 500) * (.5 + .5 * np.cos(np.pi * st / steps))
    f, p, pv, e = [x.cuda(non_blocking=True) for x in q.get()]
    with torch.autocast("cuda", dtype=torch.float16):
        z = ag.norm(f); l = F.smooth_l1_loss(ag.pi(z, p).float(), e, beta=.1); lp = F.smooth_l1_loss(pr(z, p).float()[:, :3], pv[:, :3] / 5)
    opt.zero_grad(); sc.scale(l + .3 * lp).backward(); sc.unscale_(opt); torch.nn.utils.clip_grad_norm_(ag.pi.parameters(), 1.); sc.step(opt); sc.update()
    if st % 500 == 0: print(st, l.item(), lp.item(), time.time() - t0, flush=True)
ag.eval(); pr.eval(); m = V["valid"]; ii = np.argwhere(m)[::11]; E = []; X = []
with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
    for k in range(0, len(ii), 512):
        j = ii[k:k + 512]; g = lambda n: torch.from_numpy(V[n][j[:, 0], j[:, 1]]).cuda()
        z = ag.norm(g("feat")); E.append((ag.pi(z, g("prop")).float() - g("exp")).abs().mean(1)); X.append((pr(z, g("prop")).float()[:, :3] * 5 - g("priv")[:, :3]).norm(dim=1) / g("priv")[:, :3].norm(dim=1))
os.makedirs(out, exist_ok=True); torch.save(dict(agent=ag.state_dict()), f"{out}/bc.pt")
enc = Enc().cuda(); R = dict(act_mae=torch.cat(E).mean().item(), gate_relerr_median=torch.cat(X).median().item(), policy=evaluate(ag, enc), expert6=evaluate(None, enc, kind="expert", n=512), expert8=evaluate(None, enc, kind="expert", v=8.), random=evaluate(None, enc, kind="random"))
json.dump(R, open(f"{out}/bc.json", "w"), indent=1); print(json.dumps(R, indent=1)); os._exit(0)
