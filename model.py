import torch, torch.nn as nn, torch.nn.functional as F
P = 64; TOK = P + 1        # patch tokens + 1 (proprio, action) token per frame
SIZES = dict(s=(384, 8, 6), m=(512, 8, 8), l=(768, 12, 12), xl=(1024, 16, 16), xxl=(1024, 24, 16))   # dim, depth, heads
def rms(x): return x * (x.pow(2).mean(-1, keepdim=True) + 1e-6).rsqrt()
class Block(nn.Module):
    def __init__(s, d, h):
        super().__init__(); s.h = h; s.n1 = nn.LayerNorm(d); s.n2 = nn.LayerNorm(d); s.qkv = nn.Linear(d, 3 * d); s.o = nn.Linear(d, d)
        s.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)); s.qs = nn.Parameter(torch.ones(h, 1, 1))
    def forward(s, x, mask=None, cache=None):
        B, N, D = x.shape
        q, k, v = s.qkv(s.n1(x)).view(B, N, 3, s.h, D // s.h).permute(2, 0, 3, 1, 4)
        q, k = (rms(q.float()) * s.qs).type_as(v), rms(k.float()).type_as(v)          # QK-norm, FP16 safe
        if cache is not None:
            if cache: k, v = torch.cat([cache[0], k], 2), torch.cat([cache[1], v], 2)
            cache[:] = [k, v]
        x = x + s.o(F.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2).reshape(B, N, D))
        return x + s.ff(s.n2(x))
class WM(nn.Module):
    """Frame-causal transformer. Input frame t: normalized DINO tokens, proprio_t, action_t -> tokens_{t+1}, proprio_{t+1}, reward_t, gate_t, done_t."""
    def __init__(s, size="s", T=16, df=384):
        super().__init__(); d, L, h = SIZES[size]; s.T = T; s.d = d
        s.inp = nn.Linear(df, d); s.ap = nn.Sequential(nn.Linear(13, d), nn.GELU(), nn.Linear(d, d))
        s.pos = nn.Parameter(torch.randn(1, 1, TOK, d) * .02); s.tim = nn.Parameter(torch.randn(1, T, 1, d) * .02)
        s.act = nn.Linear(13, d)            # action/proprio also added to every patch token
        s.blocks = nn.ModuleList(Block(d, h) for _ in range(L)); s.n = nn.LayerNorm(d)
        s.out = nn.Linear(d, df); s.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 9 + 1 + 1 + 1))
        nn.init.zeros_(s.out.weight); nn.init.zeros_(s.out.bias); nn.init.zeros_(s.head[-1].weight); nn.init.zeros_(s.head[-1].bias)
        for b in s.blocks:
            for m in (b.o, b.ff[2]): nn.init.normal_(m.weight, std=.02 / (2 * L) ** .5); nn.init.zeros_(m.bias)
        s.register_buffer("mu", torch.zeros(df)); s.register_buffer("sd", torch.ones(df)); s.register_buffer("dsd", torch.ones(9))
        s.ckpt = False
    def mask(s, T, dev):
        i = torch.arange(T, device=dev).repeat_interleave(TOK); return (i[None] <= i[:, None])[None, None]
    def norm(s, f): return (f.float() - s.mu) / s.sd
    def forward(s, z, prop, act, cache=None, t0=0):
        """z normalized (B,T,64,384). cache: list of per-layer [] for incremental decoding, one frame per call"""
        B, T = z.shape[:2]; pa = torch.cat([prop, act], -1)
        x = torch.cat([s.inp(z) + s.act(pa)[:, :, None], s.ap(pa)[:, :, None]], 2) + s.pos + s.tim[:, t0:t0 + T]
        x = x.reshape(B, T * TOK, s.d); m = s.mask(T, z.device) if cache is None and T > 1 else None
        for i, b in enumerate(s.blocks):
            if s.ckpt and s.training: x = torch.utils.checkpoint.checkpoint(b, x, m, use_reentrant=False)
            else: x = b(x, m, None if cache is None else cache[i])
        x = s.n(x).view(B, T, TOK, s.d)
        y = s.head(x[:, :, -1]).float()
        return z + s.out(x[:, :, :P]).float(), prop + y[..., :9] * s.dsd, y[..., 9], y[..., 10], y[..., 11]
