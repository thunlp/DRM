import argparse
import json
import os

DEFAULT_HF_ENDPOINT = "https://huggingface.co"
os.environ["HF_ENDPOINT"] = DEFAULT_HF_ENDPOINT

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


ARMO_ATTRIBUTES = [
    "helpsteer-helpfulness",
    "helpsteer-correctness",
    "helpsteer-coherence",
    "helpsteer-complexity",
    "helpsteer-verbosity",
    "ultrafeedback-overall_score",
    "ultrafeedback-instruction_following",
    "ultrafeedback-truthfulness",
    "ultrafeedback-honesty",
    "ultrafeedback-helpfulness",
    "beavertails-is_safe",
    "prometheus-score",
    "argilla-overall_quality",
    "argilla-judge_lm",
    "code-complexity",
    "code-style",
    "code-explanation",
    "code-instruction-following",
    "code-readability",
]


def configure_hf_endpoint(endpoint):
    os.environ["HF_ENDPOINT"] = endpoint
    try:
        import huggingface_hub.constants as hf_constants

        hf_constants.ENDPOINT = endpoint
    except Exception:
        pass
    try:
        import datasets.config as datasets_config

        datasets_config.HF_ENDPOINT = endpoint
    except Exception:
        pass


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


def load_encoder(model_path, device):
    encoder = AutoModel.from_pretrained(model_path, **resolve_model_kwargs())
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    encoder.eval()
    encoder.to(device)
    return encoder, tokenizer


@torch.no_grad()
def encode_messages(encoder, tokenizer, messages, max_length, device):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    if tokenizer.bos_token:
        text = text.replace(tokenizer.bos_token, "")
    token_count = len(tokenizer(text, add_special_tokens=False)["input_ids"])
    if token_count > max_length:
        return None, token_count
    inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=False).to(device)
    outputs = encoder(**inputs)
    embedding = outputs.last_hidden_state[0, -1].float().cpu().numpy().astype(np.float32)
    return embedding, token_count


def prepare_data(args):
    configure_hf_endpoint(args.hf_endpoint)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = load_dataset("RLHFlow/ArmoRM-Multi-Objective-Data-v0.1")["train"].shuffle(seed=args.seed)
    if args.limit > 0:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    encoder, tokenizer = load_encoder(args.model_path, device)

    embeddings = []
    labels = []
    dataset_srcs = []
    messages_list = []
    skipped_over_length = 0

    for example in tqdm(dataset, desc="encoding"):
        embedding, token_count = encode_messages(encoder, tokenizer, example["messages"], args.max_length, device)
        if embedding is None:
            skipped_over_length += 1
            continue
        label = [np.nan if example.get(name) is None else float(example[name]) for name in ARMO_ATTRIBUTES]
        embeddings.append(embedding)
        labels.append(label)
        dataset_srcs.append(example.get("dataset", "unknown"))
        messages_list.append(example["messages"])

    output_dir = os.path.join(args.output_root, "armo_rm_19d", "FsfairX-LLaMA3-RM-v0.1")
    os.makedirs(output_dir, exist_ok=True)
    suffix = "limit_None" if args.limit <= 0 else f"limit_{args.limit}"
    np.savez_compressed(
        os.path.join(output_dir, f"armo_rm_embeddings_labels_{suffix}.npz"),
        embeddings=np.asarray(embeddings, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.float32),
    )
    np.save(os.path.join(output_dir, f"armo_rm_dataset_srcs_{suffix}.npy"), np.asarray(dataset_srcs, dtype=object))
    np.save(os.path.join(output_dir, f"armo_rm_messages_{suffix}.npy"), np.asarray(messages_list, dtype=object))
    metadata = {
        "dataset": "RLHFlow/ArmoRM-Multi-Objective-Data-v0.1",
        "split": "train",
        "embedding_model": args.model_path,
        "label_names": ARMO_ATTRIBUTES,
        "label_dim": len(ARMO_ATTRIBUTES),
        "num_rows": len(embeddings),
        "num_skipped_over_length": skipped_over_length,
        "max_length": args.max_length,
    }
    with open(os.path.join(output_dir, f"armo_rm_metadata_{suffix}.json"), "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    print(f"Saved prepared data to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare ArmoRM embeddings and labels for RewardDiT.")
    parser.add_argument("--model_path", default="sfairXC/FsfairX-LLaMA3-RM-v0.1")
    parser.add_argument("--output_root", default="data/armo_dataset")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hf_endpoint", default=DEFAULT_HF_ENDPOINT)
    return parser.parse_args()


if __name__ == "__main__":
    prepare_data(parse_args())
