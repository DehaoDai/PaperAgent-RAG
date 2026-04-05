from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _stable_bucket(identifier: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{identifier}".encode("utf-8")).hexdigest()
    value = int(digest[:16], 16)
    return value / float(16**16 - 1)


def split_jsonl(
    *,
    input_path: Path,
    train_output_path: Path,
    eval_output_path: Path,
    eval_ratio: float,
    seed: int,
) -> dict[str, object]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")
    if not 0.0 < eval_ratio < 1.0:
        raise ValueError("eval_ratio must be between 0 and 1.")

    lines = [line for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    train_output_path.parent.mkdir(parents=True, exist_ok=True)
    eval_output_path.parent.mkdir(parents=True, exist_ok=True)

    train_records: list[str] = []
    eval_records: list[str] = []

    for line in lines:
        record = json.loads(line)
        identifier = str(record.get("id") or line)
        if _stable_bucket(identifier, seed) < eval_ratio:
            eval_records.append(line)
        else:
            train_records.append(line)

    if not train_records:
        raise ValueError("Split produced an empty train set. Reduce eval_ratio or add more data.")
    if not eval_records:
        raise ValueError("Split produced an empty eval set. Increase eval_ratio or add more data.")

    train_output_path.write_text("\n".join(train_records) + "\n", encoding="utf-8")
    eval_output_path.write_text("\n".join(eval_records) + "\n", encoding="utf-8")

    return {
        "input_path": str(input_path),
        "train_output_path": str(train_output_path),
        "eval_output_path": str(eval_output_path),
        "input_count": len(lines),
        "train_count": len(train_records),
        "eval_count": len(eval_records),
        "eval_ratio": eval_ratio,
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Split SFT JSONL into deterministic train/eval files.")
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL path.")
    parser.add_argument("--train-output", type=Path, required=True, help="Train JSONL output path.")
    parser.add_argument("--eval-output", type=Path, required=True, help="Eval JSONL output path.")
    parser.add_argument("--eval-ratio", type=float, default=0.1, help="Fraction of records assigned to eval.")
    parser.add_argument("--seed", type=int, default=42, help="Hash seed for deterministic split.")
    args = parser.parse_args()

    summary = split_jsonl(
        input_path=args.input,
        train_output_path=args.train_output,
        eval_output_path=args.eval_output,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
