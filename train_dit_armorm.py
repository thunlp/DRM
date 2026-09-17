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
from prepare_armorm_data import ARMO_ATTRIBUTES


class RewardDataset(Dataset):
    def __init__(self, rows):
        self.rows = list(rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        embedding, label = self.rows[index]
        mask = ~np.isnan(label)
        label = np.where(mask, label, 0.0).astype(np.float32)
        return (
            torch.tensor(embedding, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
            torch.tensor(mask.astype(np.float32), dtype=torch.float32),
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
    data_dir = os.path.join(data_root, "armo_rm_19d", "FsfairX-LLaMA3-RM-v0.1")
    data = np.load(os.path.join(data_dir, f"armo_rm_embeddings_labels_{suffix}.npz"))
    dataset_srcs = np.load(os.path.join(data_dir, f"armo_rm_dataset_srcs_{suffix}.npy"), allow_pickle=True)
    return data_dir, data["embeddings"].astype(np.float32), data["labels"].astype(np.float32), dataset_srcs


def split_rows(embeddings, labels, dataset_srcs, train_ratio, seed):
    dataset_srcs_set = set(dataset_srcs)
    grouped_rows = {dataset_src: [] for dataset_src in dataset_srcs_set}
    for embedding, label, dataset_src in zip(embeddings, labels, dataset_srcs):
        grouped_rows[dataset_src].append((embedding, label))

    rng = random.Random(seed)
    train_rows = []
    val_rows = []
    for dataset_src in dataset_srcs_set:
        rows = grouped_rows[dataset_src]
        rng.shuffle(rows)
        pivot = int(len(rows) * train_ratio)
        train_rows.extend(rows[:pivot])
        val_rows.extend(rows[pivot:])
    return train_rows, val_rows


@torch.no_grad()
def validate(model, loader, scheduler, args, device):
    model.eval()
    total_loss = 0.0
    used_batches = 0
    for embedding, label, mask in tqdm(loader, desc="val", leave=False):
        embedding = embedding.to(device)
        label = label.to(device)
        mask = mask.to(device)
        sampled = sample_rewards(
            scheduler,
            model,
            embedding,
            mask,
            label.shape[1],
            device,
            num_steps=args.val_num_steps,
            guidance_scale=args.guidance_scale,
            num_samples=args.num_samples,
        )
        pred = sampled.mean(dim=1)
        loss = ((pred - label).pow(2) * mask).sum() / mask.sum().clamp_min(1.0)
        total_loss += loss.item()
        used_batches += 1
    return total_loss / max(1, used_batches)


def train(args):
    set_seed(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    data_dir, embeddings, labels, dataset_srcs = load_arrays(args.data_root, args.limit)
    train_rows, val_rows = split_rows(embeddings, labels, dataset_srcs, args.train_ratio, args.seed)
    train_loader = DataLoader(RewardDataset(train_rows), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(RewardDataset(val_rows), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    text_emb_dim = int(embeddings.shape[1])
    reward_dim = int(labels.shape[1])
    model = RewardDiT(
        reward_dim=reward_dim,
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
    config.update(
        {
            "data_dir": data_dir,
            "text_emb_dim": text_emb_dim,
            "reward_dim": reward_dim,
            "label_names": ARMO_ATTRIBUTES,
        }
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        used_batches = 0
        for embedding, label, mask in tqdm(train_loader, desc="train", leave=False):
            embedding = embedding.to(device)
            label = label.to(device)
            mask = mask.to(device)
            timesteps = torch.randint(0, args.num_train_timesteps, (embedding.shape[0],), device=device).long()
            noise = torch.randn_like(label) * mask
            noisy_reward = train_scheduler.add_noise(label, noise, timesteps)
            if args.drop_cond_prob > 0:
                drop_mask = torch.rand(embedding.shape[0], device=device) < args.drop_cond_prob
                embedding = embedding.clone()
                embedding[drop_mask] = 0
            pred_noise = model(noisy_reward, timesteps, embedding, mask)
            loss = ((pred_noise - noise).pow(2) * mask).sum() / mask.sum().clamp_min(1.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += loss.item()
            used_batches += 1

        train_loss = total_loss / max(1, used_batches)
        val_loss = validate(model, val_loader, val_scheduler, args, device)
        print(f"Epoch {epoch}: train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
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
    parser = argparse.ArgumentParser(description="Train RewardDiT on ArmoRM multi-objective labels.")
    parser.add_argument("--data_root", default="data/armo_dataset")
    parser.add_argument("--output_dir", default="outputs/dit_armorm")
    parser.add_argument("--limit", default="None")
    parser.add_argument("--train_ratio", type=float, default=0.8)
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
    parser.add_argument("--drop_cond_prob", type=float, default=0.1)
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--beta_schedule", default="squaredcos_cap_v2")
    parser.add_argument("--prediction_type", default="epsilon")
    parser.add_argument("--val_num_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--num_samples", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
