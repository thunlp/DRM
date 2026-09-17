import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dit import RewardDiT, sample_rewards


class PairDataset(Dataset):
    def __init__(self, pairs, embeddings, row_targets):
        self.pairs = list(pairs)
        self.embeddings = embeddings
        self.row_targets = row_targets

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        first_idx, second_idx, pair_label, target_margin = self.pairs[index]
        return (
            torch.tensor(self.embeddings[first_idx], dtype=torch.float32),
            torch.tensor(self.embeddings[second_idx], dtype=torch.float32),
            torch.tensor([self.row_targets[first_idx]], dtype=torch.float32),
            torch.tensor([self.row_targets[second_idx]], dtype=torch.float32),
            torch.tensor(pair_label, dtype=torch.float32),
            torch.tensor(target_margin, dtype=torch.float32),
        )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_limit(limit):
    if str(limit).lower() in {"none", "all", "full", "0", "-1"}:
        return "None"
    return str(limit)


def load_arrays(data_root, limit):
    suffix = f"limit_{resolve_limit(limit)}"
    data_dir = os.path.join(data_root, "tulu3_preference_mixture", "FsfairX-LLaMA3-RM-v0.1")
    data = np.load(os.path.join(data_dir, f"armo_rm_embeddings_labels_{suffix}.npz"))
    return {
        "data_dir": data_dir,
        "embeddings": data["embeddings"].astype(np.float32),
        "pair_indices": np.load(os.path.join(data_dir, f"armo_rm_pair_indices_{suffix}.npy")).astype(np.int64),
        "pair_labels": np.load(os.path.join(data_dir, f"armo_rm_pair_labels_{suffix}.npy")).astype(np.int64),
        "pair_margins": np.load(os.path.join(data_dir, f"armo_rm_pair_margins_{suffix}.npy")).astype(np.float32),
    }


def build_targets(num_rows, pair_indices, pair_labels, target_margin):
    margins = pair_labels.astype(np.float32) * target_margin
    row_targets = np.zeros(num_rows, dtype=np.float32)
    counts = np.zeros(num_rows, dtype=np.float32)
    for (first_idx, second_idx), margin in zip(pair_indices, margins):
        row_targets[first_idx] += -0.5 * margin
        row_targets[second_idx] += 0.5 * margin
        counts[first_idx] += 1.0
        counts[second_idx] += 1.0
    valid = counts > 0
    row_targets[valid] /= counts[valid]
    rows = [(int(i), int(j), int(label), float(margin)) for (i, j), label, margin in zip(pair_indices, pair_labels, margins)]
    return row_targets, rows


def split_pairs(rows, train_ratio, seed):
    rows = list(rows)
    rng = random.Random(seed)
    rng.shuffle(rows)
    pivot = int(len(rows) * train_ratio)
    return rows[:pivot], rows[pivot:]


def predict_x0_from_epsilon(scheduler, noisy_reward, timesteps, pred_noise):
    alphas_cumprod = scheduler.alphas_cumprod.to(noisy_reward.device)
    alpha_bar = alphas_cumprod[timesteps].view(-1, 1)
    return (noisy_reward - (1.0 - alpha_bar).sqrt() * pred_noise) / alpha_bar.sqrt().clamp_min(1e-8)


def pairwise_loss(first_score, second_score, pair_label):
    sign = pair_label.view(-1, 1)
    signed_margin = sign * (second_score - first_score)
    non_tie = sign.abs() > 0
    if not non_tie.any():
        return first_score.new_tensor(0.0), signed_margin
    loss = -F.logsigmoid(signed_margin[non_tie]).mean()
    return loss, signed_margin


def pair_accuracy(signed_margin, pair_label):
    non_tie = pair_label.view(-1, 1).abs() > 0
    if not non_tie.any():
        return 0.0
    return float((signed_margin[non_tie] > 0).float().mean().item())


def train_epoch(model, loader, optimizer, scheduler, args, device):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    used_batches = 0
    for batch in tqdm(loader, desc="train", leave=False):
        first_emb, second_emb, first_target, second_target, pair_label, _ = batch
        first_emb = first_emb.to(device)
        second_emb = second_emb.to(device)
        first_target = first_target.to(device)
        second_target = second_target.to(device)
        pair_label = pair_label.to(device)

        embedding = torch.cat([first_emb, second_emb], dim=0)
        clean_reward = torch.cat([first_target, second_target], dim=0)
        mask = torch.ones_like(clean_reward)
        timesteps = torch.randint(0, args.num_train_timesteps, (clean_reward.shape[0],), device=device).long()
        noise = torch.randn_like(clean_reward)
        noisy_reward = scheduler.add_noise(clean_reward, noise, timesteps)
        pred_noise = model(noisy_reward, timesteps, embedding, mask)
        denoise_loss = F.mse_loss(pred_noise, noise)
        x0_pred = predict_x0_from_epsilon(scheduler, noisy_reward, timesteps, pred_noise)
        first_score, second_score = x0_pred.chunk(2, dim=0)
        bt_loss, signed_margin = pairwise_loss(first_score, second_score, pair_label)
        reward_reg_loss = x0_pred.pow(2).mean()
        loss = args.denoise_loss_weight * denoise_loss + args.bt_alpha * bt_loss + args.reward_reg_weight * reward_reg_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        total_loss += loss.item()
        total_acc += pair_accuracy(signed_margin.detach(), pair_label.detach())
        used_batches += 1
    return total_loss / max(1, used_batches), total_acc / max(1, used_batches)


@torch.no_grad()
def validate(model, loader, scheduler, args, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    used_batches = 0
    for batch in tqdm(loader, desc="val", leave=False):
        first_emb, second_emb, first_target, second_target, pair_label, _ = batch
        first_emb = first_emb.to(device)
        second_emb = second_emb.to(device)
        first_target = first_target.to(device)
        second_target = second_target.to(device)
        pair_label = pair_label.to(device)

        embedding = torch.cat([first_emb, second_emb], dim=0)
        clean_reward = torch.cat([first_target, second_target], dim=0)
        mask = torch.ones_like(clean_reward)
        sampled = sample_rewards(
            scheduler,
            model,
            embedding,
            mask,
            1,
            device,
            num_steps=args.val_num_steps,
            guidance_scale=args.guidance_scale,
            num_samples=args.num_samples,
        )
        score = sampled.mean(dim=1)
        first_score, second_score = score.chunk(2, dim=0)
        mse_loss = F.mse_loss(score, clean_reward)
        bt_loss, signed_margin = pairwise_loss(first_score, second_score, pair_label)
        reward_reg_loss = score.pow(2).mean()
        loss = args.denoise_loss_weight * mse_loss + args.bt_alpha * bt_loss + args.reward_reg_weight * reward_reg_loss
        total_loss += loss.item()
        total_acc += pair_accuracy(signed_margin, pair_label)
        used_batches += 1
    return total_loss / max(1, used_batches), total_acc / max(1, used_batches)


def train(args):
    set_seed(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    arrays = load_arrays(args.data_root, args.limit)
    row_targets, rows = build_targets(
        len(arrays["embeddings"]),
        arrays["pair_indices"],
        arrays["pair_labels"],
        args.tulu_target_margin,
    )
    train_rows, val_rows = split_pairs(rows, args.train_ratio, args.seed)
    train_loader = DataLoader(PairDataset(train_rows, arrays["embeddings"], row_targets), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(PairDataset(val_rows, arrays["embeddings"], row_targets), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    text_emb_dim = int(arrays["embeddings"].shape[1])
    model = RewardDiT(
        reward_dim=1,
        text_emb_dim=text_emb_dim,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        class_dropout_prob=args.class_dropout_prob,
    ).to(device)
    train_scheduler = DDPMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        beta_schedule=args.beta_schedule,
        prediction_type=args.prediction_type,
    )
    val_scheduler = DDIMScheduler.from_config(train_scheduler.config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(args.output_dir, exist_ok=True)
    best_loss = float("inf")
    config = vars(args).copy()
    config.update({"data_dir": arrays["data_dir"], "text_emb_dim": text_emb_dim, "reward_dim": 1})

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, train_scheduler, args, device)
        val_loss, val_acc = validate(model, val_loader, val_scheduler, args, device)
        print(f"Epoch {epoch}: train_loss={train_loss:.6f} train_acc={train_acc:.4f} val_loss={val_loss:.6f} val_acc={val_acc:.4f}")
        payload = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "best_metric": min(best_loss, val_loss),
            "config": config,
        }
        torch.save(payload, os.path.join(args.output_dir, "ckpt_last.pth"))
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(payload, os.path.join(args.output_dir, "ckpt_best.pth"))

    with open(os.path.join(args.output_dir, "train_config.json"), "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Train RewardDiT on Tulu3 pair-preference data.")
    parser.add_argument("--data_root", default="data/armo_dataset")
    parser.add_argument("--output_dir", default="outputs/dit_tulu3_pair")
    parser.add_argument("--limit", default="None")
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--hidden_size", type=int, default=384)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--mlp_ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--class_dropout_prob", type=float, default=0.15)
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--beta_schedule", default="squaredcos_cap_v2")
    parser.add_argument("--prediction_type", default="epsilon")
    parser.add_argument("--val_num_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--bt_alpha", type=float, default=0.5)
    parser.add_argument("--denoise_loss_weight", type=float, default=1.0)
    parser.add_argument("--reward_reg_weight", type=float, default=1e-3)
    parser.add_argument("--tulu_target_margin", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
