"""TransDiffuser DDPM denoising decoder with multi-modal cross-attention."""
import math, torch, torch.nn as nn, torch.nn.functional as F
from typing import Dict, Tuple

class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, dim): super().__init__(); self.dim = dim
    def forward(self, t):
        half = self.dim//2
        freqs = torch.exp(-math.log(10000)*torch.arange(half, device=t.device, dtype=torch.float32)/half)
        args = t.float().unsqueeze(-1)*freqs.unsqueeze(0)
        return torch.cat([args.cos(), args.sin()], dim=-1)

class MotionEncoder(nn.Module):
    def __init__(self, trajectory_length=8, action_dim=2, ego_status_dim=7, embed_dim=256):
        super().__init__()
        self.action_enc = nn.Sequential(nn.Linear(trajectory_length*action_dim, embed_dim), nn.ReLU(True), nn.Linear(embed_dim, embed_dim))
        self.ego_enc = nn.Sequential(nn.Linear(ego_status_dim, embed_dim), nn.ReLU(True), nn.Linear(embed_dim, embed_dim))
    def forward(self, hist_traj, ego_status):
        B = hist_traj.shape[0]
        return self.action_enc(hist_traj.reshape(B,-1)).unsqueeze(1), self.ego_enc(ego_status).unsqueeze(1)

class DenosingDecoderLayer(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, ff_dim=1024, dropout=0.1, num_cross_attn=5):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.self_attn_norm = nn.LayerNorm(embed_dim)
        self.cross_attns = nn.ModuleList([nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True) for _ in range(num_cross_attn)])
        self.cross_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_cross_attn)])
        self.ffn = nn.Sequential(nn.Linear(embed_dim, ff_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff_dim, embed_dim), nn.Dropout(dropout))
        self.ffn_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, cond_feats, t_emb):
        x = x + t_emb.unsqueeze(1)
        r = x; x, _ = self.self_attn(x, x, x); x = self.self_attn_norm(r + x)
        for i, (ca, cn) in enumerate(zip(self.cross_attns, self.cross_norms)):
            if i < len(cond_feats):
                r = x; x, _ = ca(x, cond_feats[i], cond_feats[i]); x = cn(r + x)
        return self.ffn_norm(x + self.ffn(x))

class DiffusionTrajectoryDecoder(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_layers=6, ff_dim=1024, dropout=0.1,
                 trajectory_length=8, action_dim=2, num_diffusion_steps=10, num_candidates=30,
                 num_cross_attn_targets=5, beta_start=0.0001, beta_end=0.02):
        super().__init__()
        self.embed_dim = embed_dim; self.trajectory_length = trajectory_length
        self.action_dim = action_dim; self.num_steps = num_diffusion_steps; self.num_candidates = num_candidates
        betas = torch.linspace(beta_start, beta_end, num_diffusion_steps)
        alphas = 1.0-betas; alpha_bar = torch.cumprod(alphas, 0)
        self.register_buffer("betas", betas); self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_ab", alpha_bar.sqrt()); self.register_buffer("sqrt_omab", (1-alpha_bar).sqrt())
        self.action_embed = nn.Sequential(nn.Linear(action_dim, embed_dim), nn.ReLU(True), nn.Linear(embed_dim, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, trajectory_length, embed_dim)*0.02)
        self.time_embed = nn.Sequential(SinusoidalTimestepEmbedding(embed_dim), nn.Linear(embed_dim, embed_dim), nn.ReLU(True), nn.Linear(embed_dim, embed_dim))
        self.layers = nn.ModuleList([DenosingDecoderLayer(embed_dim, num_heads, ff_dim, dropout, num_cross_attn_targets) for _ in range(num_layers)])
        self.noise_pred = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(True), nn.Linear(embed_dim, action_dim))

    def _decode(self, x, cond, t):
        t_emb = self.time_embed(t)
        for layer in self.layers: x = layer(x, cond, t_emb)
        return self.noise_pred(x)

    def forward_train(self, gt_actions, cond_feats):
        B = gt_actions.shape[0]; device = gt_actions.device
        t = torch.randint(0, self.num_steps, (B,), device=device)
        noise = torch.randn_like(gt_actions)
        noisy = self.sqrt_ab[t].reshape(B,1,1)*gt_actions + self.sqrt_omab[t].reshape(B,1,1)*noise
        x = self.action_embed(noisy) + self.pos_embed
        eps = self._decode(x, cond_feats, t)
        return {"noise_pred": eps, "noise_target": noise, "fused_repr": x.reshape(B,-1)}

    @torch.no_grad()
    def forward_inference(self, cond_feats, batch_size):
        device = self.betas.device; T = self.trajectory_length; D = self.action_dim; K = self.num_candidates
        cond_exp = [f.unsqueeze(1).expand(-1,K,-1,-1).reshape(batch_size*K, f.shape[1], f.shape[2]) for f in cond_feats]
        x_t = torch.randn(batch_size*K, T, D, device=device)
        for step in reversed(range(self.num_steps)):
            t = torch.full((batch_size*K,), step, device=device, dtype=torch.long)
            x_emb = self.action_embed(x_t) + self.pos_embed
            eps = self._decode(x_emb, cond_exp, t)
            mean = (1.0/self.alphas[step].sqrt())*(x_t - (self.betas[step]/(1-self.alpha_bar[step]).sqrt())*eps)
            x_t = mean + (self.betas[step].sqrt()*torch.randn_like(x_t) if step > 0 else 0)
        return torch.cumsum(x_t.reshape(batch_size, K, T, D), dim=2)
