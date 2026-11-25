import sys, glob, json, time, os, copy, argparse, numpy as np, torch, torch.nn.functional as F, threading, queue
from model import *; from policy import *; from enc import Enc; from sim import Sim
"""PPO on a behaviour-cloned agent. --wm CKPT: rollouts happen inside the world model. --real: same PPO in the simulator (control)."""
ap = argparse.ArgumentParser(); A = ap.add_argument
A("--wm", default=""); A("--real", type=int, default=0); A("--bc", required=True); A("--out", required=True); A("--data", default="tr[0-9]"); A("--iters", type=int, default=1500); A("--B", type=int, default=192); A("--H", type=int, default=12)
A("--lr", type=float, default=1e-4); A("--kl", type=float, default=.02); A("--crash", type=float, default=10.); A("--gamma", type=float, default=.985); A("--hours", type=float, default=3.); A("--seed", type=int, default=0)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); D = "/data1/j50057870/world-model/data"; CT = 4
ag = load_agent(a.bc).train(); bc = copy.deepcopy(ag).eval().requires_grad_(False); enc = Enc().cuda(); amp = lambda: torch.autocast("cuda", dtype=torch.float16)
if a.real:
    env = Sim(a.B, seed=5000 + a.seed); obs = (enc(env.render()), env.proprio()); inter = 0
else:
    ck = torch.load(a.wm, map_location="cuda"); wm = WM(ck["size"]).cuda().eval().requires_grad_(False); wm.load_state_dict(ck["net"])
    S = [{k: np.load(f"{d}/{k}.npy", mmap_mode="r" if k == "feat" else None) for k in ("feat", "prop", "act", "valid")} for d in sorted(glob.glob(f"{D}/{a.data}"))]
    I = []
    for j, x in enumerate(S):
        n = x["valid"].sum(1); c = np.clip(n - CT + 1, 0, None); I.append(np.stack([np.full(c.sum(), j), np.repeat(np.arange(len(n)), c), np.concatenate([np.arange(k) for k in c])], 1))
    I = np.concatenate(I); q = queue.Queue(8)
    def w():
        while True:
            i = I[rng.integers(len(I), size=a.B)]; q.put([torch.from_numpy(np.stack([S[j][k][n, o:o + CT] for j, n, o in i])).pin_memory() for k in ("feat", "prop", "act")])
    for _ in range(4): threading.Thread(target=w, daemon=True).start()
@torch.no_grad()
def imagine():
    f, p, ac = [x.cuda(non_blocking=True) for x in q.get()]; z = wm.norm(f); c = [[] for _ in wm.blocks]
    with amp():
        for t in range(CT - 1): wm(z[:, t:t + 1], p[:, t:t + 1], ac[:, t:t + 1], c, t)
        zt, pt = z[:, CT - 1], p[:, CT - 1]; o = {k: [] for k in ("f", "p", "a", "lp", "r", "c")}
        for h in range(a.H):
            raw = zt * wm.sd + wm.mu; za = ag.norm(raw); d = ag.dist(za, pt); x = d.sample()
            zn, pn, r, g, dn = [y[:, 0] for y in wm(zt[:, None], pt[:, None], x.clamp(-1, 1)[:, None], c, CT - 1 + h)]
            dn = dn.sigmoid(); rew = r / 4 + g.sigmoid() - a.crash * dn
            for k, v in (("f", za.half()), ("p", pt), ("a", x), ("lp", d.log_prob(x).sum(-1)), ("r", rew), ("c", 1 - dn)): o[k].append(v)
            zt, pt = zn, pn
        o["f"].append(ag.norm(zt * wm.sd + wm.mu).half()); o["p"].append(pt)
    return {k: torch.stack(v) for k, v in o.items()}
@torch.no_grad()
def real():
    global obs, inter
    o = {k: [] for k in ("f", "p", "a", "lp", "r", "c")}
    for h in range(a.H):
        with amp(): za = ag.norm(obs[0]); d = ag.dist(za, obs[1])
        x = d.sample(); r, g, dn = env.step(x); rew = r + g.float() - a.crash * dn.float()
        for k, v in (("f", za.half()), ("p", obs[1]), ("a", x), ("lp", d.log_prob(x).sum(-1)), ("r", rew), ("c", 1 - dn.float())): o[k].append(v)
        env.reset((dn | (env.t > 300)).nonzero()[:, 0]); obs = (enc(env.render()), env.proprio()); inter += a.B
    o["f"].append(ag.norm(obs[0]).half()); o["p"].append(obs[1])
    return {k: torch.stack(v) for k, v in o.items()}
opt = torch.optim.Adam([dict(params=[*ag.pi.parameters(), ag.ls], lr=a.lr), dict(params=ag.vf.parameters(), lr=3e-4)], eps=1e-5); sc = torch.amp.GradScaler("cuda", init_scale=2. ** 12)
t0 = time.time(); best = -1; log = open(f"{a.out}/log.jsonl", "a"); WARM = 100
def ev(it, extra={}):
    global best
    ag.eval(); r = evaluate(ag, enc, n=256); ag.train(); r.update(it=it, hours=(time.time() - t0) / 3600, **extra); log.write(json.dumps(r) + "\n"); log.flush(); print("EVAL", r, flush=True)
    score = r["gates"] - 3 * r["crash"]
    if score > best: best = score; torch.save(dict(agent=ag.state_dict(), it=it, ev=r), f"{a.out}/best.pt")
    torch.save(dict(agent=ag.state_dict(), it=it, ev=r), f"{a.out}/last.pt")
ev(0)
for it in range(1, a.iters + 1):
    o = real() if a.real else imagine(); T, B = a.H, a.B; fl = lambda x: x.reshape(-1, *x.shape[2:])
    with torch.no_grad(), amp(): v = torch.cat([ag.value(fl(o["f"])[k:k + 1024].float(), fl(o["p"])[k:k + 1024]) for k in range(0, (T + 1) * B, 1024)]).view(T + 1, B)
    adv = torch.zeros(T, B, device="cuda"); l = 0
    for t in reversed(range(T)): de = o["r"][t] + a.gamma * o["c"][t] * v[t + 1] - v[t]; l = de + a.gamma * .95 * o["c"][t] * l; adv[t] = l
    wgt = torch.cat([torch.ones(1, B, device="cuda"), o["c"][:-1].cumprod(0)]) if not a.real else torch.ones(T, B, device="cuda")
    ret = adv + v[:-1]; fz, fp, fa, flp, fad, fre, fw = [fl(x) for x in (o["f"][:-1], o["p"][:-1], o["a"], o["lp"], adv, ret, wgt)]; fad = (fad - fad.mean()) / (fad.std() + 1e-6)
    for _ in range(2):
        for j in torch.randperm(T * B, device="cuda").split(576):
            with amp():
                d = ag.dist(fz[j].float(), fp[j]); rt = (d.log_prob(fa[j]).sum(-1) - flp[j]).exp(); vl = (F.smooth_l1_loss(ag.value(fz[j].float(), fp[j]), fre[j], reduction="none") * fw[j]).mean()
                with torch.no_grad(): mb = bc.pi(fz[j].float(), fp[j]).float()
                pg = -(torch.minimum(rt * fad[j], rt.clamp(.8, 1.2) * fad[j]) * fw[j]).mean(); kl = ((d.mean - mb) ** 2).sum(-1).mean()
                L = vl + (pg + a.kl * kl * 10 - 2e-3 * d.entropy().sum(-1).mean() if it > WARM else 0)
            opt.zero_grad(); sc.scale(L).backward(); sc.unscale_(opt); torch.nn.utils.clip_grad_norm_([*ag.pi.parameters(), ag.ls], .5); torch.nn.utils.clip_grad_norm_(ag.vf.parameters(), 1.); sc.step(opt); sc.update()
    if it % 20 == 0: print(it, "r", o["r"].mean().item(), "v", v.mean().item(), "vl", vl.item(), "kl", kl.item(), "std", ag.ls.exp().mean().item(), "s/it", (time.time() - t0) / it, flush=True)
    if it % 100 == 0: ev(it, dict(inter=inter) if a.real else {})
    if (time.time() - t0) / 3600 > a.hours: break
os._exit(0)
