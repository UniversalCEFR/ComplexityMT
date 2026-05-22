#!/usr/bin/env python3

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from gemba import get_gemba_scores


LANG_NAMES = {
    "ar": "Arabic",
    "cs": "Czech",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "nl": "Dutch",
    "ru": "Russian",
    "zh": "Chinese",
}

DEFAULT_METHOD = "GEMBA-DA"
DEFAULT_MODEL = "gpt-4"
DEFAULT_SCORE_FIELD = "gemba_da_gpt4"
DEFAULT_BATCH_SIZE = 50
DEFAULT_TRIES = 3
DEFAULT_SLEEP = 0.5


def chunked(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def read_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        raise ValueError("Input must be a JSON object with a 'results' list.")
    return data


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def normalize_gemba_return(gemba_result: Any) -> Any:
    if isinstance(gemba_result, tuple):
        return gemba_result[0]
    return gemba_result


def score_gemba_batch(
    batch: Sequence[Tuple[int, Dict[str, Any]]],
    source_lang: str,
    target_lang: str,
    method: str,
    model: str,
    tries: int,
    sleep_seconds: float,
) -> Tuple[List[Any], List[Any]]:
    sources = [item["src_text"] for _, item in batch]
    hypotheses = [item["tgt_text"] for _, item in batch]
    answers: List[Any] = [None] * len(batch)
    answer_sources: List[Any] = [None] * len(batch)
    remaining = list(range(len(batch)))

    for attempt in range(1, tries + 1):
        if not remaining:
            break

        try:
            gemba_result = get_gemba_scores(
                source=[sources[position] for position in remaining],
                hypothesis=[hypotheses[position] for position in remaining],
                source_lang=source_lang,
                target_lang=target_lang,
                method=method,
                model=model,
            )
            parsed_answers = normalize_gemba_return(gemba_result)
        except Exception as exc:
            print(f"  GEMBA batch try {attempt}/{tries} failed: {exc}")
            time.sleep(sleep_seconds)
            continue

        next_remaining = []
        for position, answer in zip(remaining, parsed_answers):
            if answer is None:
                next_remaining.append(position)
                continue
            answers[position] = answer
            answer_sources[position] = f"gemba_package_try_{attempt}"
        remaining = next_remaining
        time.sleep(sleep_seconds)

    return answers, answer_sources


def pending_groups(
    results: Sequence[Dict[str, Any]],
    score_field: str,
) -> Dict[Tuple[str, str], List[Tuple[int, Dict[str, Any]]]]:
    groups: Dict[Tuple[str, str], List[Tuple[int, Dict[str, Any]]]] = defaultdict(list)
    for index, item in enumerate(results):
        score = item.setdefault("score", {})
        if score.get(score_field) is not None:
            continue
        groups[(item["src_lang"], item["tgt_lang"])].append((index, item))
    return groups


def fill_scores(data: Dict[str, Any], args: argparse.Namespace) -> None:
    results = data["results"]
    groups = pending_groups(results, args.score_field)
    remaining = sum(len(group) for group in groups.values())
    print(f"Remaining GEMBA scores: {remaining}")

    source_field = f"{args.score_field}_source"
    error_field = f"{args.score_field}_error"

    for (source_code, target_code), group in groups.items():
        source_lang = LANG_NAMES.get(source_code, source_code)
        target_lang = LANG_NAMES.get(target_code, target_code)
        print(f"\n{source_lang} -> {target_lang}: {len(group)} items")

        for batch_num, batch in enumerate(chunked(group, args.batch_size), start=1):
            print(f"Batch {batch_num}: {len(batch)} items")
            answers, answer_sources = score_gemba_batch(
                batch=batch,
                source_lang=source_lang,
                target_lang=target_lang,
                method=args.method,
                model=args.model,
                tries=args.tries,
                sleep_seconds=args.sleep,
            )

            for (index, item), answer, answer_source in zip(batch, answers, answer_sources):
                score = item.setdefault("score", {})
                if answer is not None:
                    score[args.score_field] = answer
                    score[source_field] = answer_source
                    score.pop(error_field, None)
                    continue

                score[args.score_field] = None
                score[error_field] = f"no_parseable_gemba_answer_after_{args.tries}_tries"
                print(f"  GEMBA did not return a score for item {index}")

            write_json(args.output, data)
            print("Saved progress.")
            time.sleep(args.sleep)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fill missing GEMBA-DA scores in a structured results JSON.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", default=DEFAULT_METHOD)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--score-field", default=DEFAULT_SCORE_FIELD)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--tries", type=int, default=DEFAULT_TRIES)
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data = read_json(args.input)
    try:
        fill_scores(data, args)
    finally:
        write_json(args.output, data)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
