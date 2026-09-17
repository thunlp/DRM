import argparse
import json
import os

DEFAULT_HF_ENDPOINT = "https://huggingface.co"
os.environ["HF_ENDPOINT"] = DEFAULT_HF_ENDPOINT

import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


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


def normalize_messages(messages):
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    normalized = []
    for index, message in enumerate(messages):
        role = str(message.get("role") or ("user" if index == 0 else "assistant")).strip()
        content = str(message.get("content") or "").strip()
        normalized.append({"role": role, "content": content})
    return normalized


def prompt_response_messages(prompt, response):
    return [
        {"role": "user", "content": str(prompt or "").strip()},
        {"role": "assistant", "content": str(response or "").strip()},
    ]


def normalize_candidate(example, key):
    candidate = example[key]
    if isinstance(candidate, list):
        return normalize_messages(candidate)
    return prompt_response_messages(example.get("prompt", ""), candidate)


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


def resolve_splits(dataset_dict, split_arg):
    if split_arg == "all":
        return list(dataset_dict.keys())
    requested = [item.strip() for item in split_arg.split(",") if item.strip()]
    missing = [name for name in requested if name not in dataset_dict]
    if missing:
        raise ValueError(f"Missing splits: {missing}")
    return requested


def prepare_data(args):
    configure_hf_endpoint(args.hf_endpoint)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset_dict = load_dataset("allenai/llama-3.1-tulu-3-8b-preference-mixture")
    splits = resolve_splits(dataset_dict, args.dataset_split)
    split_datasets = []
    for split in splits:
        data = dataset_dict[split]
        if "_source_split" not in data.column_names:
            data = data.add_column("_source_split", [split] * len(data))
        split_datasets.append(data)
    dataset = concatenate_datasets(split_datasets) if len(split_datasets) > 1 else split_datasets[0]
    dataset = dataset.shuffle(seed=args.seed)
    if args.limit > 0:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    encoder, tokenizer = load_encoder(args.model_path, device)
    embeddings = []
    labels = []
    dataset_srcs = []
    messages_list = []
    row_roles = []
    pair_indices = []
    pair_labels = []
    pair_margins = []
    pair_metadata = []
    skipped_over_length = 0

    for row_index, example in enumerate(tqdm(dataset, desc="encoding pairs")):
        rejected = normalize_candidate(example, "rejected")
        chosen = normalize_candidate(example, "chosen")
        rejected_embedding, rejected_tokens = encode_messages(encoder, tokenizer, rejected, args.max_length, device)
        chosen_embedding, chosen_tokens = encode_messages(encoder, tokenizer, chosen, args.max_length, device)
        if rejected_embedding is None or chosen_embedding is None:
            skipped_over_length += 1
            continue

        first_index = len(embeddings)
        second_index = first_index + 1
        source = str(example.get("source") or "unknown")
        split = str(example.get("_source_split") or "train")
        raw_id = str(example.get("id") or row_index)
        dataset_src = f"allenai/llama-3.1-tulu-3-8b-preference-mixture:{split}:{source}"
        embeddings.extend([rejected_embedding, chosen_embedding])
        labels.extend([[0.0, np.nan], [1.0, np.nan]])
        dataset_srcs.extend([dataset_src, dataset_src])
        messages_list.extend([rejected, chosen])
        row_roles.extend(["rejected", "chosen"])
        pair_indices.append([first_index, second_index])
        pair_labels.append(1)
        pair_margins.append(1.0)
        pair_metadata.append(
            {
                "pair_id": f"tulu3_preference_mixture:{split}:{raw_id}",
                "row_indices": [first_index, second_index],
                "row_roles": ["rejected", "chosen"],
                "pair_label": 1,
                "pair_label_meaning": "1=second_candidate_is_chosen",
                "token_counts": [rejected_tokens, chosen_tokens],
                "source": source,
            }
        )

    output_dir = os.path.join(args.output_root, "tulu3_preference_mixture", "FsfairX-LLaMA3-RM-v0.1")
    os.makedirs(output_dir, exist_ok=True)
    suffix = "limit_None" if args.limit <= 0 else f"limit_{args.limit}"
    np.savez_compressed(
        os.path.join(output_dir, f"armo_rm_embeddings_labels_{suffix}.npz"),
        embeddings=np.asarray(embeddings, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.float32),
    )
    np.save(os.path.join(output_dir, f"armo_rm_dataset_srcs_{suffix}.npy"), np.asarray(dataset_srcs, dtype=object))
    np.save(os.path.join(output_dir, f"armo_rm_messages_{suffix}.npy"), np.asarray(messages_list, dtype=object))
    np.save(os.path.join(output_dir, f"armo_rm_pair_indices_{suffix}.npy"), np.asarray(pair_indices, dtype=np.int64))
    np.save(os.path.join(output_dir, f"armo_rm_pair_labels_{suffix}.npy"), np.asarray(pair_labels, dtype=np.int8))
    np.save(os.path.join(output_dir, f"armo_rm_pair_margins_{suffix}.npy"), np.asarray(pair_margins, dtype=np.float32))
    np.save(os.path.join(output_dir, f"armo_rm_row_roles_{suffix}.npy"), np.asarray(row_roles, dtype=object))
    with open(os.path.join(output_dir, f"armo_rm_pair_metadata_{suffix}.jsonl"), "w", encoding="utf-8") as file:
        for row in pair_metadata:
            file.write(json.dumps(row) + "\n")
    metadata = {
        "dataset": "allenai/llama-3.1-tulu-3-8b-preference-mixture",
        "splits": splits,
        "embedding_model": args.model_path,
        "label_names": ["preference_label", "overall_preference"],
        "pair_label_meaning": "-1=first_candidate_better, 0=tie, 1=second_candidate_better",
        "num_pairs": len(pair_indices),
        "num_rows": len(embeddings),
        "num_skipped_over_length": skipped_over_length,
        "max_length": args.max_length,
    }
    with open(os.path.join(output_dir, f"armo_rm_metadata_{suffix}.json"), "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    print(f"Saved prepared data to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Tulu3 pair-preference embeddings for RewardDiT.")
    parser.add_argument("--model_path", default="sfairXC/FsfairX-LLaMA3-RM-v0.1")
    parser.add_argument("--output_root", default="data/armo_dataset")
    parser.add_argument("--dataset_split", default="all")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hf_endpoint", default=DEFAULT_HF_ENDPOINT)
    return parser.parse_args()


if __name__ == "__main__":
    prepare_data(parse_args())
