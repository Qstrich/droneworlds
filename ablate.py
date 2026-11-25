import sys, torch; from policy import *; from enc import Enc; from sim import *
"""Does the policy use the images? normal / blank image / image from another env / zero proprio, on fresh seeds."""
ag = load_agent(sys.argv[1]); enc = Enc().cuda()
for name in ("normal", "blank", "shuffled", "zeroprop"):
    R = []
    for seed in (31337, 4242):
        n = 512; s = Sim(n, seed=seed); al = torch.ones(n, dtype=torch.bool, device="cuda"); g = torch.zeros(n, device="cuda")
        for t in range(300):
            im = s.render(); im = im * 0 + 128 if name == "blank" else im.roll(1, 0) if name == "shuffled" else im
            p = s.proprio(); a = ag.act(enc(im), p * 0 if name == "zeroprop" else p); r, gg, d = s.step(a); g += gg & al; al &= ~d
        R.append((1 - al.float().mean().item(), g.mean().item()))
    print(name, "crash/gates per seed", R, flush=True)
