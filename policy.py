import torch, torch.nn as nn, torch.nn.functional as F
class Net(nn.Module):
    def __init__(s, out, d=192, L=2):
        super().__init__(); s.i = nn.Linear(384, d); s.p = nn.Parameter(torch.randn(1, 64, d) * .02)
        s.pr = nn.Sequential(nn.Linear(9, d), nn.GELU(), nn.Linear(d, d))
        s.b = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 4 * d, 0., "gelu", batch_first=True, norm_first=True), L, enable_nested_tensor=False)
        s.o = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, out)); nn.init.normal_(s.o[-1].weight, std=.01); nn.init.zeros_(s.o[-1].bias)
    def forward(s, z, p): return s.o(s.b(torch.cat([s.pr(p)[:, None], s.i(z) + s.p], 1))[:, 0])
class Agent(nn.Module):
    """reactive policy: normalized DINO tokens + proprio -> Gaussian over actions"""
    def __init__(s):
        super().__init__(); s.pi = Net(4); s.vf = Net(1); s.ls = nn.Parameter(torch.full((4,), -1.2))
        s.register_buffer("mu", torch.zeros(384)); s.register_buffer("sd", torch.ones(384))
    def norm(s, f): return (f.float() - s.mu) / s.sd
    def dist(s, z, p): return torch.distributions.Normal(s.pi(z, p).float(), s.ls.clamp(-3, 0).exp())
    def value(s, z, p): return s.vf(z, p).float()[:, 0]
    def reset(s, n): pass
    @torch.no_grad()
    def act(s, f, p):
        with torch.autocast("cuda", dtype=torch.float16): return s.pi(s.norm(f), p).float().clamp(-1, 1)
def load_agent(path):
    a = Agent().cuda(); a.load_state_dict(torch.load(path, map_location="cuda")["agent"]); return a.eval()
@torch.no_grad()
def evaluate(agent, enc, n=512, T=300, seed=999, kind="agent", v=6.):
    from sim import Sim
    s = Sim(n, seed=seed); al = torch.ones(n, dtype=torch.bool, device="cuda"); g = torch.zeros(n, device="cuda"); lap = torch.full((n,), float("nan"), device="cuda"); sp = 0; c = 0; ret = torch.zeros(n, device="cuda")
    for t in range(T):
        if kind == "expert": a = s.expert(False, v)
        elif kind == "random": a = (torch.randn(n, 4, device="cuda") * .5).clamp(-1, 1)
        else: a = agent.act(enc(s.render()), s.proprio())
        r, gg, d = s.step(a); g += gg & al; ret += r * al; lap = torch.where(al & (g >= 6) & lap.isnan(), torch.full_like(lap, (t + 1) * .05), lap); sp += (s.v.norm(dim=-1) * al).sum(); c += al.sum(); al &= ~d
    return dict(crash=1 - al.float().mean().item(), gates=g.mean().item(), lap_rate=(~lap.isnan()).float().mean().item(), lap_time=lap.nanmean().item(), speed=(sp / c).item(), progress=ret.mean().item())
