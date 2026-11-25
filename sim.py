"""Batched drone-racing simulator + analytic RGB renderer, pure torch.
World: z up. Body: x forward, y left, z up. Action in [-1,1]^4: collective thrust, body rates (roll,pitch,yaw). 20 Hz."""
import math, torch
import torch.nn.functional as F
G = 9.81; DT = 0.05; SUB = 5; NG = 6
INNER, OUTER = 0.75, 1.05          # gate half-sizes: opening, outside of frame
TMAX = 3.0 * G; WMAX = (6.0, 6.0, 4.0); TAU = 0.04; DRAG = 0.15
RES = 112; FOV = 100.0; TILT = math.radians(20)


def qmul(a, b):
    w, x, y, z = a.unbind(-1); W, X, Y, Z = b.unbind(-1)
    return torch.stack([w*W-x*X-y*Y-z*Z, w*X+x*W+y*Z-z*Y, w*Y-x*Z+y*W+z*X, w*Z+x*Y-y*X+z*W], -1)


def q2R(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y), 2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)], -1).view(*q.shape[:-1], 3, 3)


def R2q_yaw(yaw):
    z = torch.zeros_like(yaw)
    return torch.stack([torch.cos(yaw/2), z, z, torch.sin(yaw/2)], -1)


class Sim:
    def __init__(self, n, device="cuda", seed=0, ss=2):
        self.n, self.d, self.ss = n, device, ss
        self.g = torch.Generator(device); self.g.manual_seed(seed)
        r = RES * ss; f = 0.5 * r / math.tan(math.radians(FOV) / 2)
        c = (torch.arange(r, device=device) + 0.5 - r / 2) / f
        yy, xx = torch.meshgrid(c, c, indexing="ij")
        ray = torch.stack([torch.ones_like(xx), -xx, -yy], -1).view(-1, 3)   # cam forward,left,up
        ct, st = math.cos(TILT), math.sin(TILT)                              # camera pitched up
        Rc = torch.tensor([[ct, 0, -st], [0, 1, 0], [st, 0, ct]], device=device)
        self.rays = F.normalize(ray @ Rc.T, dim=-1)
        z = lambda *s: torch.zeros(n, *s, device=device)
        self.p, self.v, self.w = z(3), z(3), z(3); self.q = z(4); self.q[:, 0] = 1
        self.gp, self.gyaw, self.next = z(NG, 3), z(NG), torch.zeros(n, dtype=torch.long, device=device)
        self.passed = torch.zeros(n, dtype=torch.long, device=device); self.t = torch.zeros_like(self.passed)
        self.col = z(8, 3); self.path = z(NG * 32, 3); self.si = torch.zeros_like(self.passed)
        self.vdes = z(); self.namp = z(); self.ou = z(4)
        self.reset(torch.arange(n, device=device))

    def u(self, *s, lo=0., hi=1.): return torch.rand(*s, device=self.d, generator=self.g) * (hi - lo) + lo

    def reset(self, i):
        k = len(i)
        if k == 0: return
        th = torch.arange(NG, device=self.d) * 2 * math.pi / NG + self.u(k, NG, lo=-.25, hi=.25) + self.u(k, 1, hi=6.283)
        rad = self.u(k, 1, lo=7, hi=11) * self.u(k, NG, lo=.8, hi=1.2)
        self.gp[i] = torch.stack([rad*torch.cos(th), rad*torch.sin(th), self.u(k, NG, lo=1.4, hi=3.2)], -1)
        nx, pv = self.gp[i].roll(-1, 1), self.gp[i].roll(1, 1)
        tan = nx - pv
        self.gyaw[i] = torch.atan2(tan[..., 1], tan[..., 0]) + self.u(k, NG, lo=-.25, hi=.25)
        nrm = self.normals(i)
        a, b = self.gp[i], nx
        L = (b - a).norm(dim=-1, keepdim=True)
        ta, tb = nrm * L, nrm.roll(-1, 1) * L
        s = (torch.arange(32, device=self.d) / 32.).view(1, 1, 32, 1)
        h = (2*s**3-3*s**2+1)*a[:, :, None]+(s**3-2*s**2+s)*ta[:, :, None]+(-2*s**3+3*s**2)*b[:, :, None]+(s**3-s**2)*tb[:, :, None]
        self.path[i] = h.reshape(k, -1, 3)
        st = torch.randint(NG, (k,), device=self.d, generator=self.g)
        self.si[i] = st * 32 + 6
        self.next[i] = (st + 1) % NG
        ii = torch.arange(k, device=self.d)
        pp = self.path[i]; p0 = pp[ii, self.si[i]]; dr = F.normalize(pp[ii, self.si[i] + 2] - p0, dim=-1)
        self.p[i] = p0 + self.u(k, 3, lo=-.25, hi=.25)
        self.p[i, 2] = self.p[i, 2].clamp(min=.8)
        self.q[i] = R2q_yaw(torch.atan2(dr[:, 1], dr[:, 0]) + self.u(k, lo=-.3, hi=.3))
        self.v[i] = dr * self.u(k, 1, hi=4.); self.w[i] = 0
        self.passed[i] = 0; self.t[i] = 0; self.ou[i] = 0
        self.vdes[i] = self.u(k, lo=3., hi=8.)
        self.namp[i] = self.u(k) ** 2 * 1.0
        c = self.u(k, 8, 3, lo=.15, hi=.75)
        c[:, 0] = torch.tensor([.95, .40, .05], device=self.d) + self.u(k, 3, lo=-.1, hi=.1)   # gate front
        c[:, 1] = torch.tensor([.10, .25, .85], device=self.d) + self.u(k, 3, lo=-.1, hi=.1)   # gate back
        self.col[i] = c.clamp(0, 1)

    def normals(self, i=slice(None)):
        y = self.gyaw[i]; return torch.stack([torch.cos(y), torch.sin(y), torch.zeros_like(y)], -1)

    def proprio(self):
        """what an onboard state estimator would give: body velocity, gravity direction in body, body rates"""
        R = q2R(self.q)
        return torch.cat([(R.transpose(1, 2) @ self.v[..., None])[..., 0] / 5., R[:, 2], self.w / 4.], -1)

    def step(self, a):
        a = a.clamp(-1, 1); thr = (a[:, :1] + 1) / 2 * TMAX
        wc = a[:, 1:] * torch.tensor(WMAX, device=self.d); h = DT / SUB
        ar = torch.arange(self.n, device=self.d)
        rew = torch.zeros(self.n, device=self.d); done = torch.zeros(self.n, dtype=torch.bool, device=self.d)
        gate = torch.zeros_like(done)
        d0 = (self.gp[ar, self.next] - self.p).norm(dim=-1)
        nrm = self.normals()
        for _ in range(SUB):
            old = self.p
            R = q2R(self.q)
            self.w = self.w + (wc - self.w) * (h / TAU)
            acc = R[:, :, 2] * thr - DRAG * self.v
            acc = acc - torch.tensor([0, 0, G], device=self.d)
            self.p = self.p + self.v * h + .5 * acc * h * h
            self.v = self.v + acc * h
            dq = torch.cat([torch.ones_like(thr), self.w * h / 2], -1)
            self.q = F.normalize(qmul(self.q, dq), dim=-1)
            s0 = ((old[:, None] - self.gp) * nrm).sum(-1); s1 = ((self.p[:, None] - self.gp) * nrm).sum(-1)
            cross = (s0 * s1) < 0
            tt = (s0 / (s0 - s1 + 1e-9))[..., None]
            rel = old[:, None] + (self.p - old)[:, None] * tt - self.gp
            lat = (rel[..., 0] * -nrm[..., 1] + rel[..., 1] * nrm[..., 0]).abs()
            m = torch.maximum(lat, rel[..., 2].abs())
            done |= (cross & (m > INNER - .1) & (m < OUTER + .1)).any(1)
            ok = (cross & (m <= INNER - .1) & (s0 < 0))[ar, self.next] & ~done
            rew += (d0 - (self.gp[ar, self.next] - self.p).norm(dim=-1)) * ok   # bank progress before switching target
            gate |= ok
            self.next = torch.where(ok, (self.next + 1) % NG, self.next); self.passed += ok
            d0 = torch.where(ok, (self.gp[ar, self.next] - self.p).norm(dim=-1), d0)
        d1 = (self.gp[ar, self.next] - self.p).norm(dim=-1)
        rew += d0 - d1
        done |= (self.p[:, 2] < .15) | (self.p[:, 2] > 8) | (self.p[:, :2].norm(dim=-1) > 22) | (d1 > 18)
        self.t += 1
        return rew, gate, done

    def expert(self, noise=True, vdes=None):
        ar = torch.arange(self.n, device=self.d); N = self.path.shape[1]
        win = (self.si[:, None] + torch.arange(24, device=self.d)) % N
        pw = self.path[ar[:, None], win]
        self.si = win[ar, (pw - self.p[:, None]).norm(dim=-1).argmin(1)]
        v = self.vdes if vdes is None else torch.full_like(self.vdes, vdes)
        pn = self.path[ar, self.si]; tn = F.normalize(self.path[ar, (self.si + 1) % N] - pn, dim=-1)
        la = self.path[ar, (self.si + 5) % N]
        t2 = F.normalize(self.path[ar, (self.si + 6) % N] - la, dim=-1)
        curv = (t2 - tn).norm(dim=-1) / ((la - pn).norm(dim=-1) + .1)
        v = torch.minimum(v, (12. / (curv + 1e-3)).sqrt()).clamp(min=2.)
        acc = 6. * (pn - self.p) + 3.5 * (tn * v[:, None] - self.v) + (t2 - tn) / ((la - pn).norm(dim=-1, keepdim=True) + .1) * (v[:, None] ** 2) * .8
        n = acc.norm(dim=-1, keepdim=True); acc = acc * (14. / n.clamp(min=14.))
        acc = acc + DRAG * self.v; acc[:, 2] += G
        R = q2R(self.q); zb = R[:, :, 2]; zd = F.normalize(acc, dim=-1)
        thr = (acc * zb).sum(-1).clamp(min=.2 * G)
        e = (R.transpose(1, 2) @ torch.linalg.cross(zb, zd)[..., None])[..., 0]
        fwd = la - self.p; yd = torch.atan2(fwd[:, 1], fwd[:, 0]); ya = torch.atan2(R[:, 1, 0], R[:, 0, 0])
        ey = torch.atan2(torch.sin(yd - ya), torch.cos(yd - ya))
        wd = torch.stack([9 * e[:, 0], 9 * e[:, 1], 3. * ey], -1)
        a = torch.cat([(thr / TMAX * 2 - 1)[:, None], wd / torch.tensor(WMAX, device=self.d)], -1).clamp(-1, 1)
        if noise:
            self.ou = .85 * self.ou + .5 * torch.randn(self.n, 4, device=self.d, generator=self.g)
            a = (a + self.ou * self.namp[:, None]).clamp(-1, 1)
        return a

    @torch.no_grad()
    def render(self, chunk=128):
        out = []
        for s in range(0, self.n, chunk): out.append(self._render(slice(s, s + chunk)))
        return torch.cat(out)

    def _render(self, i):
        R = q2R(self.q[i]); o = self.p[i]; c = self.col[i]; B = len(o)
        d = torch.einsum("bij,nj->bni", R, self.rays)
        up = d[..., 2]
        sky = c[:, None, 2] * .6 + .4 - .35 * up.clamp(0, 1)[..., None]
        t = (-o[:, 2, None] / torch.where(up.abs() < 1e-4, torch.full_like(up, -1e-4), up))
        ground = (up < -1e-4)
        xy = o[:, None, :2] + t[..., None] * d[..., :2]
        chk = ((torch.floor(xy[..., 0] / 1.5) + torch.floor(xy[..., 1] / 1.5)) % 2)[..., None]
        cell = torch.floor(xy / 1.5); h = torch.sin(cell[..., 0] * 12.9898 + cell[..., 1] * 78.233) * 43758.5453
        h = (h - torch.floor(h))[..., None]
        gcol = (c[:, None, 3] * chk + c[:, None, 4] * (1 - chk)) * (.75 + .5 * h)
        fog = torch.exp(-t.clamp(0, 200) / 25.)[..., None]
        hz = c[:, None, 2] * .6 + .4
        img = torch.where(ground[..., None], gcol * fog + hz * (1 - fog), sky)
        best = torch.where(ground, t, torch.full_like(t, 1e9))
        nrm = self.normals(i); gp = self.gp[i]
        for k in range(NG):
            n, g = nrm[:, k], gp[:, k]
            dn = torch.einsum("bni,bi->bn", d, n); dn = torch.where(dn.abs() < 1e-5, torch.full_like(dn, 1e-5), dn)
            tk = ((g - o) * n).sum(-1)[:, None] / dn
            r = o[:, None] + tk[..., None] * d - g[:, None]
            lat = -r[..., 0] * n[:, None, 1] + r[..., 1] * n[:, None, 0]
            m = torch.maximum(lat.abs(), r[..., 2].abs())
            hit = (tk > .05) & (tk < best) & (m > INNER) & (m < OUTER)
            fc = torch.where((dn > 0)[..., None], c[:, None, 0], c[:, None, 1])
            fk = torch.exp(-tk.clamp(0, 200) / 40.)[..., None]
            img = torch.where(hit[..., None], fc * fk + hz * (1 - fk), img)
            best = torch.where(hit, tk, best)
        img = img.clamp(0, 1).view(B, RES * self.ss, RES * self.ss, 3).permute(0, 3, 1, 2)
        if self.ss > 1: img = F.avg_pool2d(img, self.ss)
        return (img * 255).round().to(torch.uint8)
