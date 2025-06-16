"""
Interference-inspired high-frequency generation layer.
"""
import torch
import torch.nn as nn
from einops import rearrange


class InterferenceLayer(nn.Module):
    def __init__(self, in_dim, out_dim, metric_dim=32, patch_size=8):
        super(InterferenceLayer, self).__init__()
        """
        Each patch as a light emitter.
        """
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.metric_dim = metric_dim
        self.patch_size = patch_size

        self.img2patch = nn.Conv2d(in_dim, metric_dim, kernel_size=patch_size, stride=patch_size)
        self.patch2img = nn.ConvTranspose2d(metric_dim, out_dim, kernel_size=patch_size, stride=patch_size)

        self.projs = nn.Linear(metric_dim, 3 * metric_dim)
    
    def forward(self, x):
        """
        x: [b, c, h, w]
        """
        b, c, h, w = x.shape
        assert c == self.in_dim, "Input dimension mismatch."
        # [b, c, h, w] -> [b, metric_dim, h//patch_size, w//patch_size]
        x = self.img2patch(x) 
        x = rearrange(x, 'b c h w -> b (h w) c')  # [b, (h//patch_size)*(w//patch_size), metric_dim]
        xs = self.projs(x)
        q, k, v = xs.chunk(3, dim=-1)
        
        light_path = torch.matmul(q, k.transpose(-1, -2)) # [b, (h//patch_size)*(w//patch_size), (h//patch_size)*(w//patch_size)]
        light_path = light_path - torch.diag_embed(torch.diagonal(light_path, dim1=-2, dim2=-1)) # remove self-interference
        phase = 2 * torch.pi * light_path
        interference = torch.matmul(torch.cos(phase), v) # [b, (h//patch_size)*(w//patch_size), metric_dim]
        interference = rearrange(interference, 'b (h w) c -> b c h w', h=h//self.patch_size, w=w//self.patch_size) # [b, metric_dim, h//patch_size, w//patch_size]
        out = self.patch2img(interference)
        return out


if __name__ == "__main__":
    layer = InterferenceLayer(3, 3, metric_dim=32, patch_size=8)
    x = torch.rand(1, 3, 128, 128)
    o = layer(x)
    print(o.shape)