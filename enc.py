import torch, torch.nn as nn
from transformers import AutoModel
class Enc(nn.Module):
    """frozen DINOv2 ViT-S/14, uint8 (B,3,112,112) -> (B,64,384) patch tokens"""
    def __init__(s):
        super().__init__()
        s.m = AutoModel.from_pretrained("facebook/dinov2-small").eval().requires_grad_(False)
        s.register_buffer("mu", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1)); s.register_buffer("sd", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))
    @torch.no_grad()
    def forward(s, x):
        with torch.autocast("cuda", dtype=torch.float16):
            h = s.m(pixel_values=(x.float() / 255 - s.mu) / s.sd).last_hidden_state[:, 1:]
        return h.half()
