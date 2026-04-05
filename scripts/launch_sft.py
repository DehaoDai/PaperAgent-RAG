from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to use launch_sft.py.") from exc
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _dump_yaml(payload: dict, path: Path) -> None:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to use launch_sft.py.") from exc
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _multimodal_jsonl_to_sharegpt_records(input_path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line in input_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        sharegpt_messages: list[dict[str, str]] = []
        images: list[str] = []

        for message in item.get("messages", []):
            role = message.get("role")
            content = message.get("content")
            if isinstance(content, str):
                sharegpt_messages.append({"role": role, "content": content})
                continue
            if isinstance(content, list):
                content_chunks: list[str] = []
                for block in content:
                    block_type = block.get("type")
                    if block_type == "image":
                        images.append(str(block.get("image")))
                        content_chunks.append("<image>")
                    elif block_type == "text":
                        content_chunks.append(str(block.get("text", "")))
                sharegpt_messages.append(
                    {
                        "role": role,
                        "content": "".join(content_chunks).strip(),
                    }
                )
                continue
            raise ValueError(f"Unsupported message content format in record {item.get('id')}.")

        records.append(
            {
                "messages": sharegpt_messages,
                "images": images,
            }
        )
    return records


def prepare_llamafactory_run(
    *,
    config_path: Path,
    train_file: Path,
    run_name: str | None,
) -> dict[str, Path]:
    config_path = config_path.expanduser().resolve()
    train_file = train_file.expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"SFT config not found: {config_path}")
    if not train_file.exists():
        raise FileNotFoundError(f"Train JSONL not found: {train_file}")

    config = _load_yaml(config_path)

    run_id = run_name or datetime.now().strftime("qwen2_5_vl_sft_%Y%m%d_%H%M%S")
    run_root = Path("workspace_data") / "sft_runs" / run_id
    dataset_dir = run_root / "llamafactory_data"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    train_records = _multimodal_jsonl_to_sharegpt_records(train_file)
    if not train_records:
        raise ValueError("No training records found in the input JSONL.")

    dataset_name = "agent_rag_train"
    dataset_file = dataset_dir / f"{dataset_name}.json"
    dataset_file.write_text(json.dumps(train_records, ensure_ascii=False, indent=2), encoding="utf-8")

    dataset_info = {
        dataset_name: {
            "file_name": dataset_file.name,
            "formatting": "sharegpt",
            "columns": {
                "messages": "messages",
                "images": "images",
            },
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }
    }
    dataset_info_path = dataset_dir / "dataset_info.json"
    dataset_info_path.write_text(json.dumps(dataset_info, ensure_ascii=False, indent=2), encoding="utf-8")

    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    loss_cfg = config.get("loss", {})

    output_dir = Path(training_cfg.get("output_dir", run_root / "checkpoint")).expanduser()
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir

    train_yaml = {
        "model_name_or_path": model_cfg.get("name_or_path", "Qwen/Qwen2.5-VL-3B-Instruct"),
        "trust_remote_code": bool(model_cfg.get("trust_remote_code", True)),
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_rank": 64,
        "lora_alpha": 128,
        "lora_dropout": 0.0,
        "lora_target": "all",
        "dataset": dataset_name,
        "dataset_dir": str(dataset_dir),
        "template": "qwen2_vl",
        "cutoff_len": int(loss_cfg.get("max_length", 4096)),
        "preprocessing_num_workers": 4,
        "output_dir": str(output_dir),
        "overwrite_output_dir": True,
        "logging_steps": int(training_cfg.get("logging_steps", 10)),
        "save_steps": int(training_cfg.get("save_steps", 200)),
        "plot_loss": True,
        "report_to": "none",
        "per_device_train_batch_size": int(training_cfg.get("per_device_train_batch_size", 1)),
        "gradient_accumulation_steps": int(training_cfg.get("gradient_accumulation_steps", 8)),
        "learning_rate": float(training_cfg.get("learning_rate", 2e-5)),
        "num_train_epochs": float(training_cfg.get("num_train_epochs", 2)),
        "lr_scheduler_type": "cosine",
        "max_grad_norm": 1.0,
        "warmup_ratio": float(training_cfg.get("warmup_ratio", 0.03)),
        "packing": False,
        "bf16": bool(training_cfg.get("bf16", True)),
        "gradient_checkpointing": bool(training_cfg.get("gradient_checkpointing", True)),
        "ddp_timeout": 180000000,
    }
    train_yaml_path = run_root / "llamafactory_train.yaml"
    run_root.mkdir(parents=True, exist_ok=True)
    _dump_yaml(train_yaml, train_yaml_path)

    return {
        "run_root": run_root,
        "dataset_dir": dataset_dir,
        "dataset_file": dataset_file,
        "dataset_info_path": dataset_info_path,
        "train_yaml_path": train_yaml_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and optionally launch a Qwen2.5-VL SFT run with LLaMA-Factory.")
    parser.add_argument("--config", type=Path, required=True, help="SFT config template path.")
    parser.add_argument("--train-file", type=Path, required=True, help="Input multimodal JSONL train file.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional run name.")
    parser.add_argument("--execute", action="store_true", help="Actually run llamafactory-cli train after preparation.")
    args = parser.parse_args()

    paths = prepare_llamafactory_run(
        config_path=args.config,
        train_file=args.train_file,
        run_name=args.run_name,
    )

    command = ["llamafactory-cli", "train", str(paths["train_yaml_path"])]
    summary = {
        "run_root": str(paths["run_root"]),
        "dataset_dir": str(paths["dataset_dir"]),
        "dataset_file": str(paths["dataset_file"]),
        "dataset_info_path": str(paths["dataset_info_path"]),
        "train_yaml_path": str(paths["train_yaml_path"]),
        "command": " ".join(command),
    }

    if not args.execute:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if shutil.which("llamafactory-cli") is None:
        raise RuntimeError(
            "llamafactory-cli is not installed or not on PATH. Run without --execute to generate the run package first."
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
