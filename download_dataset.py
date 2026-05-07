import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

try:
    import pandas as pd  # type: ignore
except ImportError:
    pd = None


HF_FIRST_ROWS_URL = "https://datasets-server.huggingface.co/first-rows"


def fetch_first_rows(
    dataset: str,
    config: str = "default",
    split: str = "train",
    limit: int = 100,
    timeout: int = 60,
) -> Dict[str, Any]:
    """
    Fetch a sample of rows from Hugging Face datasets-server 'first-rows' endpoint.
    Note: this endpoint returns only the first rows (a preview), not the full dataset.
    """
    params = {
        "dataset": dataset,
        "config": config,
        "split": split,
        "limit": limit,
    }
    resp = requests.get(HF_FIRST_ROWS_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _rows_to_records(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    datasets-server returns rows like:
      {"row_idx": 0, "row": {...}}
    Flatten to list of dicts containing the row content plus row_idx.
    """
    records: List[Dict[str, Any]] = []
    for item in rows:
        rec = {}
        rec["row_idx"] = item.get("row_idx")
        row = item.get("row") or {}
        if isinstance(row, dict):
            rec.update(row)
        records.append(rec)
    return records


def save_json(data: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_jsonl(records: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def save_csv(records: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Union of all keys for a stable header
    fieldnames = sorted({k for r in records for k in r.keys()})
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def save_csv_via_dataframe(records: List[Dict[str, Any]], out_path: Path) -> None:
    if pd is None:
        raise RuntimeError("pandas is not installed. Run: python3 -m pip install pandas")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame.from_records(records)
    df.to_csv(out_path, index=False)


def main(
    dataset: str,
    config: str,
    split: str,
    limit: int,
    out_dir: Path,
    fmt: str,
    to_dataframe: bool,
) -> None:
    data = fetch_first_rows(dataset=dataset, config=config, split=split, limit=limit)

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Always save the raw JSON response
    raw_path = out_dir / f"{dataset.replace('/', '__')}__{config}__{split}__first_rows_raw.json"
    save_json(data, raw_path)
    print(f"Saved raw JSON to: {raw_path}")

    rows = data.get("rows") or []
    records = _rows_to_records(rows)

    # 2) Always save CSV too (prefer pandas if installed)
    csv_path = out_dir / f"{dataset.replace('/', '__')}__{config}__{split}__first_rows.csv"
    if pd is not None:
        save_csv_via_dataframe(records, csv_path)
        print(f"Saved CSV (via pandas DataFrame) to: {csv_path}")
    else:
        save_csv(records, csv_path)
        print(f"Saved CSV (via csv module; install pandas for DataFrame) to: {csv_path}")

    # Keep the old optional outputs if you still want them
    if fmt == "jsonl":
        out_path = out_dir / f"{dataset.replace('/', '__')}__{config}__{split}__first_rows.jsonl"
        save_jsonl(records, out_path)
        print(f"Saved rows as JSONL to: {out_path}")
    elif fmt == "csv":
        # already saved above; do nothing
        pass
    elif fmt == "none":
        pass
    else:
        raise ValueError(f"Unknown format: {fmt}")

    # If you no longer want a separate df CSV, disable this block (or leave it)
    if to_dataframe:
        df_path = out_dir / f"{dataset.replace('/', '__')}__{config}__{split}__first_rows_df.csv"
        save_csv_via_dataframe(records, df_path)
        print(f"Saved DataFrame CSV to: {df_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Helsinki-NLP/shroom-cap")
    parser.add_argument("--config", default="default")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument("--format", choices=["jsonl", "csv", "none"], default="jsonl")
    parser.add_argument("--to-dataframe", action="store_true", help="Also save CSV via pandas DataFrame")
    args = parser.parse_args()

    main(
        dataset=args.dataset,
        config=args.config,
        split=args.split,
        limit=args.limit,
        out_dir=args.out_dir,
        fmt=args.format,
        to_dataframe=args.to_dataframe,
    )