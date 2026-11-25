import torch, time
from sim import *
from PIL import Image
for noise, v in ((False, 4.), (False, 6.), (False, 8.), (True, None)):
    s = Sim(1024, seed=1); dead = torch.zeros(1024, dtype=torch.bool, device="cuda"); ng = torch.zeros(1024, device="cuda"); ret = 0
    for _ in range(300):
        r, g, d = s.step(s.expert(noise, v)); ng += g * ~dead; ret += (r * ~dead).mean(); dead |= d
    print("noise", noise, "v", v, "crash", dead.float().mean().item(), "gates", ng.mean().item(), "ret", float(ret), "speed", s.v[~dead].norm(dim=-1).mean().item(), flush=True)
# free-fall and hover checks
s = Sim(4); s.v[:] = 0; s.q[:] = torch.tensor([1., 0, 0, 0]); s.p[:] = torch.tensor([0., 0, 5]); a = torch.zeros(4, 4, device="cuda"); a[:, 0] = 2/3 - 1
for _ in range(20): s.step(a)
print("hover drift", (s.p - torch.tensor([0., 0, 5], device="cuda")).abs().max().item())
s = Sim(512, seed=3); t = time.time(); torch.cuda.synchronize()
rows = []
for k in range(24):
    s.step(s.expert(True)); im = s.render()
    if k % 4 == 3: rows.append(im[:8].permute(0, 2, 3, 1).reshape(8, RES, RES, 3))
torch.cuda.synchronize(); print("render fps", 512 * 24 / (time.time() - t), "mem", torch.cuda.max_memory_allocated() / 1e9)
g = torch.cat([torch.cat(list(r), 1) for r in rows], 0).cpu().numpy(); Image.fromarray(g).save("/data1/j50057870/world-model/runs/sim.png")
