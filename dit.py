import math

import torch
import torch.nn as nn


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def timestep_embedding(timesteps, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(0, half, dtype=torch.float32, device=timesteps.device)
            / half
        )
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, timesteps):
        return self.mlp(self.timestep_embedding(timesteps, self.frequency_embedding_size))


class TextEmbedder(nn.Module):
    def __init__(self, text_emb_dim, hidden_size, dropout_prob=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(text_emb_dim, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.dropout_prob = dropout_prob
        self.uncond_embedding = nn.Parameter(torch.randn(hidden_size) * 0.02)

    def forward(self, text_emb, training, force_uncond=False, batch_size=None):
        if text_emb is None:
            if batch_size is None:
                raise ValueError("batch_size is required when text_emb is None")
            return self.uncond_embedding.unsqueeze(0).expand(batch_size, -1)

        batch_size = text_emb.shape[0]
        if force_uncond:
            return self.uncond_embedding.unsqueeze(0).expand(batch_size, -1)

        drop_mask = None
        if training and self.dropout_prob > 0:
            drop_mask = torch.rand(batch_size, device=text_emb.device) < self.dropout_prob
            text_emb = text_emb.clone()
            text_emb[drop_mask] = 0

        text_feat = self.proj(text_emb)
        if drop_mask is not None:
            text_feat[drop_mask] = self.uncond_embedding
        return text_feat


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_size),
            nn.Dropout(dropout),
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))

    def forward(self, x, cond):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(cond).chunk(6, dim=1)
        attn_input = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input)
        x = x + gate_msa.unsqueeze(1) * attn_output
        mlp_input = modulate(self.norm2(x), shift_mlp, scale_mlp)
        return x + gate_mlp.unsqueeze(1) * self.mlp(mlp_input)


class RewardDiT(nn.Module):
    def __init__(
        self,
        reward_dim=19,
        text_emb_dim=4096,
        hidden_size=384,
        depth=3,
        num_heads=6,
        mlp_ratio=4.0,
        dropout=0.2,
        class_dropout_prob=0.15,
    ):
        super().__init__()
        self.reward_dim = reward_dim
        self.hidden_size = hidden_size
        self.reward_embedder = nn.Linear(reward_dim * 2, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.text_embedder = TextEmbedder(text_emb_dim, hidden_size, class_dropout_prob)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        self.final_linear = nn.Linear(hidden_size, reward_dim)
        self.initialize_weights()

    def initialize_weights(self):
        def init_linear(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(init_linear)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.text_embedder.uncond_embedding, std=0.02)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_adaLN[-1].weight, 0)
        nn.init.constant_(self.final_adaLN[-1].bias, 0)
        nn.init.constant_(self.final_linear.weight, 0)
        nn.init.constant_(self.final_linear.bias, 0)

    def forward(self, noisy_reward, timesteps, text_emb, mask=None, force_uncond=False):
        batch_size = noisy_reward.shape[0]
        if mask is None:
            mask = torch.ones_like(noisy_reward)
        reward_token = self.reward_embedder(torch.cat([noisy_reward, mask], dim=-1)).unsqueeze(1)
        reward_token = reward_token + self.pos_embed
        cond = self.t_embedder(timesteps) + self.text_embedder(
            text_emb,
            self.training,
            force_uncond=force_uncond,
            batch_size=batch_size,
        )
        for block in self.blocks:
            reward_token = block(reward_token, cond)
        shift, scale = self.final_adaLN(cond).chunk(2, dim=1)
        reward_token = modulate(self.final_norm(reward_token), shift, scale)
        return self.final_linear(reward_token.squeeze(1)) * mask


@torch.no_grad()
def sample_rewards(
    scheduler,
    denoiser,
    embedding,
    mask,
    reward_dim,
    device,
    num_steps=20,
    guidance_scale=3.5,
    num_samples=8,
):
    batch_size = embedding.shape[0]
    cond_embedding = embedding.repeat_interleave(num_samples, dim=0).to(device)
    mask_batch = mask.repeat_interleave(num_samples, dim=0).to(device)
    rewards = torch.randn(batch_size * num_samples, reward_dim, device=device) * mask_batch
    scheduler.set_timesteps(num_steps, device=device)
    for timestep in scheduler.timesteps:
        t_batch = torch.full((batch_size * num_samples,), int(timestep), device=device, dtype=torch.long)
        eps_uncond = denoiser(rewards, t_batch, None, mask_batch, force_uncond=True)
        eps_cond = denoiser(rewards, t_batch, cond_embedding, mask_batch)
        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        rewards = scheduler.step(eps, timestep, rewards).prev_sample
    return rewards.reshape(batch_size, num_samples, reward_dim)
