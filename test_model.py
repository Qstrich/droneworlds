import torch; from model import *
for dt in ("fp32", "amp"):
    torch.manual_seed(0); m = WM("s").cuda().eval()
    for p in m.parameters(): p.data += torch.randn_like(p) * .01
    z, p, a = torch.randn(2, 8, 64, 384).cuda(), torch.randn(2, 8, 9).cuda(), torch.randn(2, 8, 4).cuda()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=dt == "amp"):
        o = m(z, p, a); z2, a2, p2 = z.clone(), a.clone(), p.clone(); z2[:, 5:] = 9; a2[:, 5:] = 9; p2[:, 5:] = 9; o2 = m(z2, p2, a2)
        print(dt, "future->past", max((x[:, :5] - y[:, :5]).abs().max().item() for x, y in zip(o, o2)), "present changes", (o[0][:, 5:] - o2[0][:, 5:]).abs().max().item())
        c = [[] for _ in m.blocks]; oc = [m(z[:, t:t + 1], p[:, t:t + 1], a[:, t:t + 1], c, t) for t in range(8)]
        print(dt, "kv-cache vs full", max((torch.cat([x[i] for x in oc], 1) - o[i]).abs().max().item() for i in range(5)))
