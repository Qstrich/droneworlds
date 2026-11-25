import sys, torch, numpy as np, time, os
from sim import *; from enc import Enc
"""collect.py OUT SEED N [policy.pt]  -> fixed 64-step segments; arrays padded after done, valid mask marks real steps"""
def priv(s):
    ar = torch.arange(s.n, device=s.d); R = q2R(s.q).transpose(1, 2)
    rel = (R @ (s.gp[ar, s.next] - s.p)[..., None])[..., 0]; n = (R @ s.normals()[ar, s.next][..., None])[..., 0]
    return torch.cat([rel, n], -1)
if __name__ == "__main__":
    out, seed, N = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); T = 64; B = 256
    os.makedirs(out, exist_ok=True); enc = Enc().cuda()
    pol = None
    if len(sys.argv) > 4:
        from policy import load_agent; pol = load_agent(sys.argv[4])
    mm = lambda n, sh, dt: np.lib.format.open_memmap(f"{out}/{n}.npy", "w+", dt, (N, *sh))
    A = dict(feat=mm("feat", (T + 1, 64, 384), np.float16), prop=mm("prop", (T + 1, 9), np.float32), priv=mm("priv", (T + 1, 6), np.float32),
             act=mm("act", (T, 4), np.float32), exp=mm("exp", (T, 4), np.float32), rew=mm("rew", (T,), np.float32), gate=mm("gate", (T,), bool),
             done=mm("done", (T,), bool), valid=mm("valid", (T,), bool))
    t0 = time.time()
    for b in range(0, N, B):
        s = Sim(B, seed=seed * 100003 + b); alive = torch.ones(B, dtype=torch.bool, device="cuda"); rec = {k: [] for k in A}
        rnd = torch.rand(B, device="cuda") < (.1 if pol is None else 0.)
        if pol is not None: pol.reset(B); beta = torch.rand(B, device="cuda") * .3
        for t in range(T + 1):
            f = enc(s.render()); rec["feat"].append(f); rec["prop"].append(s.proprio()); rec["priv"].append(priv(s))
            if t == T: break
            e = s.expert(False, 6.)
            if pol is None:
                s.si_ = None; a = s.expert(True)
                a[rnd] = (s.ou[rnd] * .5).clamp(-1, 1)
                a[rnd, 0] = (a[rnd, 0] - .33).clamp(-1, 1)
            else:
                a = (pol.act(f, rec["prop"][-1]) + beta[:, None] * torch.randn(B, 4, device="cuda")).clamp(-1, 1)
            r, g, d = s.step(a)
            for k, x in (("act", a), ("exp", e), ("rew", r), ("gate", g & alive), ("done", d & alive), ("valid", alive)): rec[k].append(x.clone())
            alive = alive & ~d
        for k in A: A[k][b:b + B] = torch.stack(rec[k], 1).cpu().numpy()
        if b % 2048 == 0: print(b, time.time() - t0, "valid", A["valid"][b:b + B].mean(), "gates/seg", A["gate"][b:b + B].sum(1).mean(), flush=True)
    for k in A: A[k].flush()
    open(f"{out}/ok", "w").write("ok")
