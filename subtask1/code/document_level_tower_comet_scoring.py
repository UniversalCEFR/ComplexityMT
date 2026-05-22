#!/usr/bin/env python3

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from comet import download_model, load_from_checkpoint
except ImportError as e:
    raise SystemExit("COMET is not installed. Run: pip install unbabel-comet") from e


CEFR_TO_NUM = {
    "A1": 1,
    "A2": 2,
    "B1": 3,
    "B2": 4,
    "C1": 5,
    "C2": 6,
}

DEFAULT_CLASSIFIER_MODEL = "UniversalCEFR/xlm-roberta-base-cefr-all-classifier"
DEFAULT_COMET_MODEL = "Unbabel/wmt22-cometkiwi-da"
DEFAULT_CLASSIFIER_BATCH_SIZE = 16
DEFAULT_COMET_BATCH_SIZE = 8


def read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        first = handle.read(1)
        handle.seek(0)

        if first == "[":
            data = json.load(handle)
            if not isinstance(data, list):
                raise ValueError("JSON input must contain a list of objects.")
            return data

        rows: List[Dict[str, Any]] = []
        for line_num, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at line {line_num}: {e}") from e
        return rows


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)


def mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def cefr_distribution(rows: Sequence[Dict[str, Any]], field: str) -> Dict[str, int]:
    return {
        label: count
        for label in CEFR_TO_NUM
        if (count := sum(1 for row in rows if row.get(field) == label))
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score MT rows with CEFR and COMET-QE.")
    parser.add_argument("--input", required=True, help="Input JSON or JSONL with source_text and translation")
    parser.add_argument("--output", required=True, help="Output JSONL with scores filled in")
    parser.add_argument("--summary", required=True, help="Summary JSON path")
    parser.add_argument("--source-field", default="source_text", help="Field containing the source text")
    parser.add_argument("--translation-field", default="translation", help="Field containing the translation text")
    parser.add_argument("--classifier-model", default=DEFAULT_CLASSIFIER_MODEL)
    parser.add_argument("--comet-model", default=DEFAULT_COMET_MODEL)
    parser.add_argument("--classifier-batch-size", type=int, default=DEFAULT_CLASSIFIER_BATCH_SIZE)
    parser.add_argument("--comet-batch-size", type=int, default=DEFAULT_COMET_BATCH_SIZE)
    parser.add_argument("--skip-cefr", action="store_true")
    parser.add_argument("--skip-comet", action="store_true")
    return parser


def ensure_row_fields(rows: List[Dict[str, Any]]) -> None:
    for row in rows:
        row.setdefault("translated_cefr", None)
        row.setdefault("translated_cefr_num", None)
        row.setdefault("translated_cefr_confidence", None)
        row.setdefault("comet", None)


def load_or_initialize_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if os.path.exists(args.output):
        rows = read_json_or_jsonl(args.output)
        if not rows:
            raise ValueError(f"Existing output file is empty: {args.output}")
        ensure_row_fields(rows)
        return rows

    rows = read_json_or_jsonl(args.input)
    if not rows:
        raise ValueError("Input dataset is empty.")
    ensure_row_fields(rows)
    write_jsonl(args.output, rows)
    return rows


class CEFRClassifier:
    def __init__(self, model_name: str) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=dtype,
        ).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict_batch(self, texts: Sequence[str], batch_size: int) -> List[Dict[str, Any]]:
        predictions: List[Dict[str, Any]] = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start:start + batch_size])
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)
            pred_ids = torch.argmax(probs, dim=-1)
            confs = torch.max(probs, dim=-1).values

            for pred_id, conf in zip(pred_ids.tolist(), confs.tolist()):
                label = str(self.model.config.id2label[pred_id]).strip().upper()
                if label not in CEFR_TO_NUM:
                    raise ValueError(f"Unexpected classifier label: {label}")
                predictions.append(
                    {
                        "translated_cefr": label,
                        "translated_cefr_num": CEFR_TO_NUM[label],
                        "translated_cefr_confidence": float(conf),
                    }
                )
        return predictions


class CometQEScorer:
    def __init__(self, model_name: str) -> None:
        model_path = download_model(model_name)
        self.model = load_from_checkpoint(model_path)

    def score(self, rows: List[Dict[str, str]], batch_size: int) -> List[float]:
        results = self.model.predict(
            rows,
            batch_size=batch_size,
            gpus=1 if torch.cuda.is_available() else 0,
        )
        if hasattr(results, "scores"):
            return list(results.scores)
        if isinstance(results, dict) and "scores" in results:
            return list(results["scores"])
        if isinstance(results, tuple) and results:
            return list(results[0])
        raise ValueError("Unexpected COMET output format")


def classify_translations(rows: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    pending_indices = [
        index for index, row in enumerate(rows)
        if row.get(args.translation_field) and row.get("translated_cefr") is None
    ]
    if not pending_indices:
        return

    classifier = CEFRClassifier(args.classifier_model)
    texts = [str(rows[index][args.translation_field]) for index in pending_indices]
    predictions = classifier.predict_batch(texts, batch_size=args.classifier_batch_size)

    for index, prediction in zip(pending_indices, predictions):
        rows[index]["translated_cefr"] = prediction["translated_cefr"]
        rows[index]["translated_cefr_num"] = prediction["translated_cefr_num"]
        rows[index]["translated_cefr_confidence"] = prediction["translated_cefr_confidence"]

    write_jsonl(args.output, rows)


def score_with_comet(rows: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    pending_indices = [
        index for index, row in enumerate(rows)
        if (
            row.get(args.source_field)
            and row.get(args.translation_field)
            and row.get("comet") is None
        )
    ]
    if not pending_indices:
        return

    scorer = CometQEScorer(args.comet_model)
    for start in tqdm(range(0, len(pending_indices), args.comet_batch_size), desc="Scoring COMET"):
        batch = pending_indices[start:start + args.comet_batch_size]
        inputs = [
            {
                "src": str(rows[index][args.source_field]),
                "mt": str(rows[index][args.translation_field]),
            }
            for index in batch
        ]
        scores = scorer.score(inputs, batch_size=min(args.comet_batch_size, len(inputs)))
        for index, score in zip(batch, scores):
            rows[index]["comet"] = None if score is None else float(score)
        write_jsonl(args.output, rows)


def build_summary(rows: List[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    scored_rows = [row for row in rows if row.get(args.translation_field)]

    by_direction: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in scored_rows:
        source_lang = row.get("source_lang") or row.get("lang")
        target_lang = row.get("target_lang")
        if source_lang and target_lang:
            by_direction[f"{source_lang}->{target_lang}"].append(row)

    def summarize(items: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "n": len(items),
            "mean_comet": mean([float(row["comet"]) for row in items if row.get("comet") is not None]),
            "translated_cefr_distribution": cefr_distribution(items, "translated_cefr"),
        }

    return {
        "config": {
            "input": args.input,
            "output": args.output,
            "summary": args.summary,
            "source_field": args.source_field,
            "translation_field": args.translation_field,
            "classifier_model": None if args.skip_cefr else args.classifier_model,
            "comet_model": None if args.skip_comet else args.comet_model,
        },
        "overall": summarize(scored_rows),
        "by_direction": {
            direction: summarize(items)
            for direction, items in sorted(by_direction.items())
        },
    }


def main() -> None:
    args = build_parser().parse_args()
    rows = load_or_initialize_rows(args)

    if not args.skip_cefr:
        classify_translations(rows, args)
    if not args.skip_comet:
        score_with_comet(rows, args)

    summary = build_summary(rows, args)
    with open(args.summary, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"Wrote scored rows to {args.output}")
    print(f"Wrote summary to {args.summary}")


if __name__ == "__main__":
    main()
