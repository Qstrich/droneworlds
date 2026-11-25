import sys, torch, imageio, numpy as np
from policy import *; from enc import Enc; from sim import *
"""film.py policy.pt out.mp4 : first-person video, 9 fresh tracks tiled 3x3"""
ag = load_agent(sys.argv[1]); enc = Enc().cuda(); s = Sim(9, seed=70001); s.ss = 2; fr = []
for t in range(240):
    im = s.render(); fr.append(im.view(3, 3, 3, RES, RES).permute(0, 3, 1, 4, 2).reshape(3 * RES, 3 * RES, 3).cpu().numpy()); s.step(ag.act(enc(im), s.proprio()))
imageio.mimsave(sys.argv[2], fr, fps=20)
