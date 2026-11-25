import os, sys, json, time, argparse, numpy as np, torch, torch.nn.functional as F, torch.distributed as dist, glob
from model import *
ap = argparse.ArgumentParser(); A = ap.add_argument
A("--size", default="s"); A("--data", default="tr[0-9]"); A("--out", required=True); A("--steps", type=int, default=30000); A("--bs", type=int, default=16)
A("--lr", type=float, default=3e-4); A("--T", type=int, default=16); A("--roll", type=int, default=0); A("--init", default=""); A("--overfit", type=int, default=0); A("--eval", type=int, default=2000); A("--nact", type=int, default=0); A("--hours", type=float, default=99)
a = ap.parse_args()
rank, ws = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1)); lr_ = int(os.environ.get("LOCAL_RANK", 0)); torch.cuda.set_device(lr_)
if ws > 1: dist.init_process_group("nccl")
D = "/data1/j50057870/world-model/data"; os.makedirs(a.out, exist_ok=True); torch.manual_seed(rank); rng = np.random.default_rng(rank + int(time.time()) % 100000)
class DS:
    def __init__(s, dirs, limit=0):
        s.A = [{k: np.load(f"{d}/{k}.npy", mmap_mode="r" if k == "feat" else None) for k in ("feat", "prop", "act", "rew", "gate", "done", "valid")} for d in dirs]
        s.W = []
        for x in s.A:
            v = x["valid"][:limit or None]; n = v.sum(1)            # usable window starts: n valid transitions
            s.W.append(n)
        s.idx = {}
    def index(s, T, full=False):
        """full: every step valid (clean rollout metrics). else: any window starting on a valid step; losses are masked after a crash"""
        key = (T, full)
        if key not in s.idx:
            L = []
            for j, n in enumerate(s.W):
                c = np.clip(n - T + 1, 0, None) if full else np.minimum(n, 64 - T + 1); seg = np.repeat(np.arange(len(n)), c); off = np.concatenate([np.arange(k) for k in c]) if c.sum() else np.zeros(0, int)
                L.append(np.stack([np.full(len(seg), j), seg, off], 1))
            s.idx[key] = np.concatenate(L)
        return s.idx[key]
    def batch(s, B, T, sel=None):
        I = s.index(T); I = I[rng.integers(len(I), size=B)] if sel is None else sel
        g = lambda k, e: np.stack([s.A[j][k][n, o:o + T + e] for j, n, o in I])
        o = dict(feat=g("feat", 1), prop=g("prop", 1), act=g("act", 0), rew=g("rew", 0), gate=g("gate", 0), done=g("done", 0), valid=g("valid", 0))
        return {k: torch.from_numpy(v) for k, v in o.items()}
def loader(ds, B, T):
    import threading, queue
    q = queue.Queue(8)
    def w():
        while True: q.put({k: v.pin_memory() for k, v in ds.batch(B, T).items()})
    for _ in range(6): threading.Thread(target=w, daemon=True).start()
    return q
cu = lambda b: {k: v.cuda(non_blocking=True) for k, v in b.items()}
net = WM(a.size).cuda(); net.ckpt = a.size in ("xxl",)
tr = DS(sorted(glob.glob(f"{D}/{a.data}")), a.overfit); va = DS([f"{D}/val"]) if not a.overfit else tr
if a.init: net.load_state_dict(torch.load(a.init, map_location="cuda")["net"])
else:
    f = np.concatenate([np.asarray(x["feat"][:64, ::8], np.float32).reshape(-1, 384) for x in tr.A]); net.mu.copy_(torch.tensor(f.mean(0))); net.sd.copy_(torch.tensor(f.std(0)) + 1e-3)
    x = tr.A[0]; dp = (x["prop"][:, 1:] - x["prop"][:, :-1])[x["valid"]]; net.dsd.copy_(torch.tensor(dp.std(0)))
npar = sum(p.numel() for p in net.parameters())
def losses(o, z1, p1, b, w=None):
    z, p, r, g, d = o; w = b["valid"].float(); n = w.sum().clamp(min=1); wm = lambda x: (x * w).sum() / n
    # no latent/proprio loss on the step that crashes (its next frame is inside geometry / post-crash)
    wz = w * (1 - b["done"].float()); nz = wz.sum().clamp(min=1); wd = lambda x: (x * wz).sum() / nz
    lz = wd(F.smooth_l1_loss(z, z1, beta=.5, reduction="none").mean((2, 3))); e = (p - b["prop"][:, 1:]) / net.dsd
    lp = wd(F.smooth_l1_loss(e, torch.zeros_like(e), reduction="none").mean(2)); lr = wm(F.smooth_l1_loss(r, b["rew"] * 4, reduction="none"))
    lg = wm(F.binary_cross_entropy_with_logits(g, b["gate"].float(), reduction="none")); ld = wm(F.binary_cross_entropy_with_logits(d, b["done"].float(), reduction="none"))
    return lz + .2 * lp + .5 * lr + .5 * lg + .5 * ld, dict(z=lz, p=lp, r=lr, g=lg, d=ld)
def fwd(m, b, roll):
    z = net.norm(b["feat"]); o = m(z[:, :-1], b["prop"][:, :-1], b["act"]); l, d = losses(o, z[:, 1:], None, b)
    if roll:                                     # feed predictions back, k extra closed-loop passes; gradient through last two
        zi, pi = z[:, :-1], b["prop"][:, :-1]
        for _ in range(roll):
            zz = torch.cat([z[:, :1], o[0][:, :-1]], 1); pp = torch.cat([b["prop"][:, :1], o[1][:, :-1]], 1)
            o = m(zz, pp, b["act"]); l2, d2 = losses(o, z[:, 1:], None, b); l = l + l2
        d = {**d, "z_roll": d2["z"]}
    return l, d
@torch.no_grad()
def rollout(b, ctx, H, act=None):
    """open loop: ctx real frames then H predicted; T window slides by restart (T<=net.T)"""
    z = net.norm(b["feat"][:, :ctx]); p = b["prop"][:, :ctx]; ac = b["act"] if act is None else act; out = []
    for h in range(H):
        o = net(z, p, ac[:, :ctx + h]); z = torch.cat([z, o[0][:, -1:]], 1); p = torch.cat([p, o[1][:, -1:]], 1); out.append([x[:, -1] for x in o[2:]])
    return z[:, ctx:], p[:, ctx:], [torch.stack(x, 1) for x in zip(*out)]
@torch.no_grad()
def evaluate():
    net.eval(); R = {}; n = 8; r2 = np.random.default_rng(0); I = va.index(16, True); J = va.index(16); S2 = r2.integers(len(J), size=n * 32); S = r2.integers(len(I), size=n * 32)
    for k in range(n):
        b = cu(va.batch(32, 16, I[S[k * 32:k * 32 + 32]])); b2 = cu(va.batch(32, 16, J[S2[k * 32:k * 32 + 32]]))
        with torch.autocast("cuda", dtype=torch.float16):
            z = net.norm(b["feat"]); o = net(z[:, :-1][:, :a.T], b["prop"][:, :-1][:, :a.T], b["act"][:, :a.T])
            R.setdefault("mse1", []).append(F.mse_loss(o[0], z[:, 1:a.T + 1])); R.setdefault("copy1", []).append(F.mse_loss(z[:, :a.T], z[:, 1:a.T + 1]))
            zz, pp, (r, g, d) = rollout(b, 4, 12); R.setdefault("mse_k", []).append(((zz - z[:, 4:16]) ** 2).mean((0, 2, 3))); R.setdefault("copy_k", []).append(((z[:, 3:4] - z[:, 4:16]) ** 2).mean((0, 2, 3)))
            R.setdefault("prop_k", []).append(((pp - b["prop"][:, 4:16]) ** 2).mean((0, 2)))
            perm = torch.randperm(32, device="cuda"); s2 = b["act"].clone(); s2[:, 4:] = b["act"][perm, 4:]; zs, _, _ = rollout(b, 4, 12, s2); R.setdefault("shuf_k", []).append(((zs - z[:, 4:16]) ** 2).mean((0, 2, 3)))
            R.setdefault("rp", []).append(r.sum(1) / 4); R.setdefault("rt", []).append(b["rew"][:, 3:15].sum(1))
            R.setdefault("gp", []).append(g.sigmoid().amax(1)); R.setdefault("gt", []).append(b["gate"][:, 3:15].any(1).float()); 
            _, _, (_, _, d) = rollout(b2, 4, 12); k2 = b2["valid"][:, 3:15]; R.setdefault("dp", []).append(d[k2].float()); R.setdefault("dt", []).append(b2["done"][:, 3:15][k2].float())
    net.train(); c = lambda k: torch.cat(R[k]).float(); m = lambda k: torch.stack(R[k]).float().mean(0)
    def auc(p, t): o = p.argsort(); rk = torch.empty_like(p); rk[o] = torch.arange(1, len(p) + 1, device=p.device).float(); n1 = t.sum(); return ((rk * t).sum() - n1 * (n1 + 1) / 2) / (n1 * (len(p) - n1) + 1e-9)
    return dict(mse1=m("mse1").item(), copy1=m("copy1").item(), **{f"{k}@{h}": m(k)[h - 1].item() for k in ("mse_k", "copy_k", "shuf_k", "prop_k") for h in (1, 4, 8, 12)},
                rew_corr=torch.corrcoef(torch.stack([c("rp"), c("rt")]))[0, 1].item(), gate_auc=auc(c("gp"), c("gt")).item(), done_auc=auc(c("dp"), c("dt")).item() if 0 < c("dt").sum() < len(c("dt")) else float("nan"), n_crash=int(c("dt").sum()))
m = torch.nn.parallel.DistributedDataParallel(net, device_ids=[lr_], gradient_as_bucket_view=True) if ws > 1 else net
wd = [p for p in net.parameters() if p.dim() > 1]; nw = [p for p in net.parameters() if p.dim() <= 1]
opt = torch.optim.AdamW([dict(params=wd, weight_decay=.02), dict(params=nw, weight_decay=0)], lr=a.lr, betas=(.9, .95)); sc = torch.amp.GradScaler("cuda", init_scale=2. ** 14)
q = loader(tr, a.bs, a.T + (0)); net.train(); t0 = time.time(); skips = 0; gns = []; lf = open(f"{a.out}/log.jsonl", "a"); best = 9e9; lrm = 1.; step = 0; last_good = None; retries = 0
def save(n): torch.save(dict(net=net.state_dict(), size=a.size, step=step, args=vars(a)), f"{a.out}/{n}.tmp"); os.replace(f"{a.out}/{n}.tmp", f"{a.out}/{n}.pt")
while step < a.steps:
    if (time.time() - t0) / 3600 > a.hours: a.steps = step; break
    lr = a.lr * lrm * min(1, step / 1000) * (.5 * (1 + np.cos(np.pi * min(1, step / a.steps))) * .95 + .05)
    for g in opt.param_groups: g["lr"] = lr
    b = cu(q.get())
    with torch.autocast("cuda", dtype=torch.float16): l, d = fwd(m, b, a.roll)
    opt.zero_grad(set_to_none=True); sc.scale(l).backward(); sc.unscale_(opt); gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
    s0 = sc.get_scale(); sc.step(opt); sc.update(); skips += sc.get_scale() < s0; step += 1; gns.append(gn.item())
    # watchdog (deterministic): overflow storm, scale collapse or NaN -> restore last good weights, halve LR, at most 3 times
    if step % 200 == 0:
        bad = torch.tensor([float(skips > 10 or sc.get_scale() < 128 or not np.isfinite(l.item()))], device="cuda")
        if ws > 1: dist.all_reduce(bad)
        if bad > 0:
            retries += 1; print(f"WATCHDOG step {step} skips {skips} scale {sc.get_scale()} loss {l.item()}", flush=True)
            if retries > 3 or last_good is None: sys.exit("watchdog: giving up")
            net.load_state_dict(last_good); lrm *= .5; opt.state.clear(); sc = torch.amp.GradScaler("cuda", init_scale=2. ** 12)
        elif step % 1000 == 0: last_good = {k: v.clone() for k, v in net.state_dict().items()}
        if rank == 0:
            r = dict(step=step, lr=lr, loss=l.item(), **{k: v.item() for k, v in d.items()}, gn=float(np.median(gns[-200:])), gnmax=float(np.nanmax(gns[-200:])), skips=int(skips), scale=sc.get_scale(), sps=step / (time.time() - t0), mem=torch.cuda.max_memory_allocated() / 1e9)
            lf.write(json.dumps(r) + "\n"); lf.flush(); print({k: round(v, 5) for k, v in r.items()}, flush=True)
        skips = 0
    if (step % a.eval == 0 or step == a.steps) and rank == 0:
        e = evaluate(); lf.write(json.dumps(dict(step=step, params=npar, eval=e)) + "\n"); lf.flush(); print("EVAL", step, {k: round(v, 4) for k, v in e.items()}, flush=True)
        save("last")
        if e["mse_k@12"] < best: best = e["mse_k@12"]; save("best")
if rank == 0: save("last"); print("done", npar, time.time() - t0)
if ws > 1: dist.destroy_process_group()
os._exit(0)
