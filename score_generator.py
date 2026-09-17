import argparse
import os
import re

import torch
import torch.nn as nn
from diffusers import DDIMScheduler
from transformers import AutoModel, AutoTokenizer

from dit import RewardDiT, sample_rewards


def torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_state_dict(payload):
    if isinstance(payload, dict) and isinstance(payload.get("model_state_dict"), dict):
        return payload["model_state_dict"]
    if isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
        return payload["state_dict"]
    return payload


def infer_config(ckpt_path, payload):
    config = dict(payload.get("config", {})) if isinstance(payload, dict) else {}
    state_dict = extract_state_dict(payload)
    if "reward_dim" not in config and isinstance(state_dict, dict):
        weight = state_dict.get("final_linear.weight")
        if torch.is_tensor(weight):
            config["reward_dim"] = int(weight.shape[0])
    if "text_emb_dim" not in config and isinstance(state_dict, dict):
        weight = state_dict.get("text_embedder.proj.0.weight")
        if torch.is_tensor(weight):
            config["text_emb_dim"] = int(weight.shape[1])
    if not config:
        base_name = os.path.basename(ckpt_path)
        match = re.search(r"h(?P<hidden>\d+)_d(?P<depth>\d+)_h(?P<heads>\d+)_dp(?P<dropout_int>\d+)p(?P<dropout_dec>\d+)", base_name)
        if match:
            config["hidden_size"] = int(match.group("hidden"))
            config["depth"] = int(match.group("depth"))
            config["num_heads"] = int(match.group("heads"))
            config["dropout"] = float(f"{match.group('dropout_int')}.{match.group('dropout_dec')}")
    defaults = {
        "reward_dim": 19,
        "text_emb_dim": 4096,
        "hidden_size": 384,
        "depth": 3,
        "num_heads": 6,
        "mlp_ratio": 4.0,
        "dropout": 0.2,
        "class_dropout_prob": 0.15,
        "num_train_timesteps": 1000,
        "beta_schedule": "squaredcos_cap_v2",
        "prediction_type": "epsilon",
    }
    defaults.update(config)
    return defaults


def resolve_model_kwargs():
    kwargs = {"torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32}
    if torch.cuda.is_available():
        try:
            import flash_attn

            kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            kwargs["attn_implementation"] = "sdpa"
    else:
        kwargs["attn_implementation"] = "eager"
    return kwargs


class ScoreGenerator(nn.Module):
    def __init__(
        self,
        dit_ckpt_path,
        device="auto",
        model_path="sfairXC/FsfairX-LLaMA3-RM-v0.1",
        max_length=4096,
        load_encoder_model=True,
    ):
        super().__init__()
        self.device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_path = model_path
        self.max_length = max_length
        payload = torch_load(dit_ckpt_path)
        self.config_dict = infer_config(dit_ckpt_path, payload)
        self.reward_dim = int(self.config_dict["reward_dim"])
        self.text_emb_dim = int(self.config_dict["text_emb_dim"])
        self.denoiser = RewardDiT(
            reward_dim=self.reward_dim,
            text_emb_dim=self.text_emb_dim,
            hidden_size=int(self.config_dict["hidden_size"]),
            depth=int(self.config_dict["depth"]),
            num_heads=int(self.config_dict["num_heads"]),
            mlp_ratio=float(self.config_dict["mlp_ratio"]),
            dropout=float(self.config_dict["dropout"]),
            class_dropout_prob=float(self.config_dict["class_dropout_prob"]),
        )
        self.denoiser.load_state_dict(extract_state_dict(payload))
        self.denoiser.to(self.device)
        self.denoiser.eval()
        self.scheduler = DDIMScheduler(
            num_train_timesteps=int(self.config_dict["num_train_timesteps"]),
            beta_schedule=str(self.config_dict["beta_schedule"]),
            prediction_type=str(self.config_dict["prediction_type"]),
        )
        if load_encoder_model:
            self.encoder = AutoModel.from_pretrained(model_path, **resolve_model_kwargs())
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.encoder.to(self.device)
            self.encoder.eval()
            self.config = self.encoder.config
        else:
            self.encoder = None
            self.tokenizer = None
            self.config = None

    def format_message(self, prompt, response):
        return [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]

    @torch.no_grad()
    def encode_messages(self, messages):
        if self.encoder is None or self.tokenizer is None:
            raise RuntimeError("Encoder model is not loaded")
        is_batch = isinstance(messages, list) and messages and isinstance(messages[0], list)
        batch_messages = messages if is_batch else [messages]
        texts = []
        for message in batch_messages:
            text = self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False)
            if self.tokenizer.bos_token:
                text = text.replace(self.tokenizer.bos_token, "")
            texts.append(text)
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
        ).to(self.device)
        outputs = self.encoder(**inputs)
        last_hidden = outputs.last_hidden_state
        lengths = inputs["attention_mask"].sum(dim=1) - 1
        embeddings = last_hidden[torch.arange(last_hidden.shape[0], device=self.device), lengths].float()
        return embeddings if is_batch else embeddings[0]

    @torch.no_grad()
    def score_embeddings(self, embeddings, num_steps=10, guidance_scale=7.0, num_samples=32):
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)
        embeddings = embeddings.float().to(self.device)
        mask = torch.ones((embeddings.shape[0], self.reward_dim), dtype=torch.float32, device=self.device)
        rewards = sample_rewards(
            self.scheduler,
            self.denoiser,
            embeddings,
            mask,
            self.reward_dim,
            self.device,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            num_samples=num_samples,
        )
        reward_mean = rewards.mean(dim=1)
        return reward_mean.mean(dim=-1)

    def forward_message(self, messages, num_steps=10, guidance_scale=7.0, num_samples=32):
        embeddings = self.encode_messages(messages)
        return self.score_embeddings(
            embeddings,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            num_samples=num_samples,
        )

    def forward(self, prompt, response, num_steps=10, guidance_scale=7.0, num_samples=32):
        return self.forward_message(
            self.format_message(prompt, response),
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            num_samples=num_samples,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Run RewardDiT score inference.")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--response", required=True)
    parser.add_argument("--model_path", default="sfairXC/FsfairX-LLaMA3-RM-v0.1")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--num_samples", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model = ScoreGenerator(args.ckpt, device=args.device, model_path=args.model_path)
    score = model(args.prompt, args.response, args.num_steps, args.guidance_scale, args.num_samples)
    print(score.detach().cpu().tolist())
