#!/usr/bin/env python3

import argparse
import json
import math
import os
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from scipy.stats import kendalltau, pearsonr, spearmanr
from tqdm import tqdm
from transformers import pipeline

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

LANG_NAME = {
    "nl": "Dutch",
    "en": "English",
    "fr": "French",
}

MODEL_NAME = "Unbabel/TowerInstruct-7B-v0.1"
COMET_MODEL = "Unbabel/wmt22-cometkiwi-da"
LANGUAGES = ["nl", "en", "fr"]

DEFAULT_MAX_NEW_TOKENS = 2048
DEFAULT_MIN_NEW_TOKENS = 32
DEFAULT_BATCH_SIZE = 8
COMET_BATCH_SIZE = 8

MIN_TARGET_WORDS = 20
MIN_LENGTH_RATIO = 0.35

BAD_PREFIX_PATTERNS = [
    r"^\s*translate the following",
    r"^\s*traduisez le texte",
    r"^\s*(english|french|dutch)\s*:",
]

ROLE_TOKEN_PATTERNS = [
    r"<\|im_start\|>\s*assistant\s*",
    r"<\|im_start\|>\s*user\s*",
    r"<\|im_end\|>",
    r"<\|[^>]+\|>",
]


def read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)

        if first == "[":
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("JSON file must contain a list of objects.")
            return data

        rows: List[Dict[str, Any]] = []
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at line {line_num}: {e}") from e
        return rows


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)


def normalize_lang(s: str) -> str:
    return str(s).strip().lower()


def ensure_cefr_num(row: Dict[str, Any]) -> int:
    level = str(row.get("cefr_level", "")).strip().upper()
    if level not in CEFR_TO_NUM:
        raise ValueError(f"Unsupported or missing CEFR level '{level}' in row: {row}")
    return CEFR_TO_NUM[level]


def normalize_text_for_prompt(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def approx_word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


class TowerTranslator:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        default_max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        default_min_new_tokens: int = DEFAULT_MIN_NEW_TOKENS,
    ) -> None:
        pipe_kwargs: Dict[str, Any] = {
            "task": "text-generation",
            "model": model_name,
            "device_map": "auto",
        }

        if torch.cuda.is_available():
            pipe_kwargs["torch_dtype"] = torch.bfloat16

        self.pipe = pipeline(**pipe_kwargs)

        if self.pipe.tokenizer.pad_token is None:
            self.pipe.tokenizer.pad_token = self.pipe.tokenizer.eos_token

        if hasattr(self.pipe.tokenizer, "padding_side"):
            self.pipe.tokenizer.padding_side = "left"

        self.default_max_new_tokens = default_max_new_tokens
        self.default_min_new_tokens = default_min_new_tokens

    def _build_prompt(self, text: str, src_lang: str, tgt_lang: str) -> str:
        src_name = LANG_NAME[src_lang]
        tgt_name = LANG_NAME[tgt_lang]
        text = normalize_text_for_prompt(text)

        messages = [
            {
                "role": "user",
                "content": (
                    f"Translate the following text from {src_name} into {tgt_name}.\n"
                    f"{src_name}: {text}\n"
                    f"{tgt_name}:"
                ),
            }
        ]

        return self.pipe.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _build_retry_prompt(self, text: str, src_lang: str, tgt_lang: str) -> str:
        src_name = LANG_NAME[src_lang]
        tgt_name = LANG_NAME[tgt_lang]
        text = normalize_text_for_prompt(text)

        messages = [
            {
                "role": "user",
                "content": (
                    f"Translate the full text from {src_name} into {tgt_name}.\n"
                    f"Translate all paragraphs completely.\n"
                    f"Do not summarize.\n"
                    f"Do not omit content.\n"
                    f"Output only the translation.\n\n"
                    f"{src_name}: {text}\n"
                    f"{tgt_name}:"
                ),
            }
        ]

        return self.pipe.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _clean_translation(self, text: str, tgt_lang: str) -> str:
        text = text.strip()

        for pattern in ROLE_TOKEN_PATTERNS:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        tgt_name = LANG_NAME[tgt_lang]
        text = re.sub(
            rf"^\s*{re.escape(tgt_name)}\s*:\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"^\s*(English|French|Dutch)\s*:\s*", "", text, flags=re.IGNORECASE)
        return text.strip()

    def _looks_bad(self, source: str, translation: str) -> bool:
        translation = translation.strip()
        if not translation:
            return True

        src_words = approx_word_count(source)
        tgt_words = approx_word_count(translation)

        if tgt_words < MIN_TARGET_WORDS:
            return True

        if src_words >= 40 and tgt_words < max(MIN_TARGET_WORDS, int(src_words * MIN_LENGTH_RATIO)):
            return True

        lowered = translation.lower()
        return any(re.search(pattern, lowered) for pattern in BAD_PREFIX_PATTERNS)

    def _estimate_generation_budget(self, texts: List[str]) -> Tuple[int, int]:
        text_lens = [
            len(self.pipe.tokenizer(t, add_special_tokens=False)["input_ids"])
            for t in texts
        ]

        max_text_len = max(text_lens) if text_lens else 0
        dynamic_max = max(
            self.default_max_new_tokens,
            int(max_text_len * 1.3),
        )
        max_new_tokens = min(4096, dynamic_max)

        dynamic_min = max(
            self.default_min_new_tokens,
            min(256, max(32, int(max_text_len * 0.15))),
        )

        if dynamic_min >= max_new_tokens:
            dynamic_min = max(1, max_new_tokens // 4)

        return max_new_tokens, dynamic_min

    def generate_batch(
        self,
        prompts: List[str],
        texts: List[str],
        max_new_tokens: Optional[int] = None,
        min_new_tokens: Optional[int] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> List[str]:
        dyn_max, dyn_min = self._estimate_generation_budget(texts)
        max_new_tokens = max_new_tokens or dyn_max
        min_new_tokens = min_new_tokens if min_new_tokens is not None else dyn_min

        outputs = self.pipe(
            prompts,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            do_sample=False,
            return_full_text=False,
            batch_size=batch_size,
            pad_token_id=self.pipe.tokenizer.pad_token_id,
        )

        generations: List[str] = []
        for out in outputs:
            if isinstance(out, list):
                generations.append(out[0]["generated_text"])
            else:
                generations.append(out["generated_text"])
        return generations

    def translate_batch(
        self,
        texts: List[str],
        src_lang: str,
        tgt_lang: str,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> List[str]:
        if not texts:
            return []

        prompts = [self._build_prompt(t, src_lang, tgt_lang) for t in texts]
        raw = self.generate_batch(
            prompts=prompts,
            texts=texts,
            batch_size=batch_size,
        )
        cleaned = [self._clean_translation(x, tgt_lang) for x in raw]

        bad_indices = [
            i for i, (src, hyp) in enumerate(zip(texts, cleaned))
            if self._looks_bad(src, hyp)
        ]

        if bad_indices:
            retry_texts = [texts[i] for i in bad_indices]
            retry_prompts = [self._build_retry_prompt(t, src_lang, tgt_lang) for t in retry_texts]
            retry_raw = self.generate_batch(
                prompts=retry_prompts,
                texts=retry_texts,
                batch_size=min(batch_size, len(retry_texts)),
                max_new_tokens=min(4096, self.default_max_new_tokens * 2),
                min_new_tokens=max(self.default_min_new_tokens, 64),
            )
            retry_cleaned = [self._clean_translation(x, tgt_lang) for x in retry_raw]

            for idx, hyp in zip(bad_indices, retry_cleaned):
                cleaned[idx] = hyp

        return cleaned


class CometQEScorer:
    def __init__(self, model_name: str = COMET_MODEL) -> None:
        model_path = download_model(model_name)
        self.model = load_from_checkpoint(model_path)

    def score(self, rows: List[Dict[str, str]], batch_size: int = COMET_BATCH_SIZE) -> List[float]:
        results = self.model.predict(
            rows,
            batch_size=batch_size,
            gpus=1 if torch.cuda.is_available() else 0,
        )
        if hasattr(results, "scores"):
            return list(results.scores)
        if isinstance(results, dict) and "scores" in results:
            return list(results["scores"])
        if isinstance(results, tuple) and len(results) > 0:
            return list(results[0])
        raise ValueError("Unexpected COMET output format")


def safe_corr(x: List[float], y: List[float]) -> Dict[str, Optional[float]]:
    if len(x) < 2 or len(y) < 2 or len(set(x)) < 2 or len(set(y)) < 2:
        return {
            "n": len(x),
            "pearson_r": None,
            "pearson_p": None,
            "spearman_rho": None,
            "spearman_p": None,
            "kendall_tau": None,
            "kendall_p": None,
        }

    pear_r, pear_p = pearsonr(x, y)
    spear_r, spear_p = spearmanr(x, y)
    kend_r, kend_p = kendalltau(x, y)

    return {
        "n": len(x),
        "pearson_r": float(pear_r),
        "pearson_p": float(pear_p),
        "spearman_rho": float(spear_r),
        "spearman_p": float(spear_p),
        "kendall_tau": float(kend_r),
        "kendall_p": float(kend_p),
    }


def compute_correlations(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [r for r in rows if r.get("comet") is not None and r.get("cefr_num") is not None]

    summary: Dict[str, Any] = {}
    summary["overall"] = safe_corr(
        [float(r["cefr_num"]) for r in valid],
        [float(r["comet"]) for r in valid],
    )

    def group_and_corr(key_fn):
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in valid:
            groups[key_fn(r)].append(r)

        out: Dict[str, Any] = {}
        for k, items in groups.items():
            out[str(k)] = safe_corr(
                [float(r["cefr_num"]) for r in items],
                [float(r["comet"]) for r in items],
            )
        return out

    summary["by_direction"] = group_and_corr(lambda r: f"{r['source_lang']}->{r['target_lang']}")
    summary["by_source_lang"] = group_and_corr(lambda r: r["source_lang"])
    summary["by_target_lang"] = group_and_corr(lambda r: r["target_lang"])
    summary["by_format"] = group_and_corr(lambda r: r.get("format", "unknown"))
    return summary


def build_rows_from_input(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    expanded_rows: List[Dict[str, Any]] = []

    for row in data:
        src_lang = normalize_lang(row.get("lang", ""))
        if src_lang not in LANG_NAME:
            raise ValueError(f"Unsupported source language '{src_lang}' in row: {row}")

        source_text = row.get("text")
        if not source_text or not isinstance(source_text, str):
            raise ValueError(f"Missing or invalid text field 'text' in row: {row}")

        cefr_level = str(row.get("cefr_level", "")).strip().upper()
        cefr_num = ensure_cefr_num(row)

        for tgt_lang in LANGUAGES:
            if tgt_lang == src_lang:
                continue

            new_row = dict(row)
            new_row["source_lang"] = src_lang
            new_row["target_lang"] = tgt_lang
            new_row["format"] = row.get("format", "unknown")
            new_row["cefr_level"] = cefr_level
            new_row["cefr_num"] = cefr_num
            new_row["model"] = MODEL_NAME
            new_row["source_text"] = source_text
            new_row.setdefault("translation", None)
            new_row.setdefault("comet", None)
            new_row.setdefault("translated_cefr", None)
            new_row.setdefault("chunk_count", 1)
            expanded_rows.append(new_row)

    return expanded_rows


def build_direction_buckets(
    expanded_rows: List[Dict[str, Any]],
    pending_indices: List[int],
) -> Dict[Tuple[str, str], List[int]]:
    groups: Dict[Tuple[str, str], List[int]] = defaultdict(list)

    for idx in pending_indices:
        row = expanded_rows[idx]
        key = (row["source_lang"], row["target_lang"])
        groups[key].append(idx)

    for key in groups:
        groups[key].sort(key=lambda i: len(expanded_rows[i]["source_text"]))
    return groups


def batched_indices(indices: List[int], batch_size: int) -> Iterable[List[int]]:
    for start in range(0, len(indices), batch_size):
        yield indices[start:start + batch_size]


def translate_documents(
    expanded_rows: List[Dict[str, Any]],
    translator: TowerTranslator,
    output_path: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    save_every_batch: bool = True,
) -> None:
    pending = [i for i, row in enumerate(expanded_rows) if not row.get("translation")]
    if not pending:
        print("No pending translations found.")
        return

    buckets = build_direction_buckets(expanded_rows, pending)
    total_batches = sum(math.ceil(len(v) / batch_size) for v in buckets.values())

    pbar = tqdm(total=total_batches, desc="Translating")

    for (src_lang, tgt_lang), indices in buckets.items():
        for batch in batched_indices(indices, batch_size):
            texts = [expanded_rows[i]["source_text"] for i in batch]
            translations = translator.translate_batch(
                texts=texts,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                batch_size=min(batch_size, len(batch)),
            )

            for idx, translation in zip(batch, translations):
                expanded_rows[idx]["translation"] = translation
                expanded_rows[idx]["chunk_count"] = 1

            if save_every_batch:
                write_jsonl(output_path, expanded_rows)

            pbar.update(1)

    pbar.close()

    if not save_every_batch:
        write_jsonl(output_path, expanded_rows)


def score_with_comet(
    expanded_rows: List[Dict[str, Any]],
    output_path: str,
    batch_size: int = COMET_BATCH_SIZE,
) -> None:
    pending_comet = [
        i for i, row in enumerate(expanded_rows)
        if row.get("translation") and row.get("comet") is None
    ]
    if not pending_comet:
        print("No pending COMET scoring found.")
        return

    scorer = CometQEScorer()

    for start in tqdm(range(0, len(pending_comet), batch_size), desc="Scoring with COMET-QE"):
        batch_indices = pending_comet[start:start + batch_size]
        comet_inputs = [
            {"src": expanded_rows[i]["source_text"], "mt": expanded_rows[i]["translation"]}
            for i in batch_indices
        ]
        scores = scorer.score(comet_inputs, batch_size=batch_size)

        for i, score in zip(batch_indices, scores):
            expanded_rows[i]["comet"] = None if score is None else float(score)

        write_jsonl(output_path, expanded_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSON or JSONL")
    parser.add_argument("--output", required=True, help="Output JSONL with translations and COMET")
    parser.add_argument("--summary", required=True, help="Summary JSON with correlations")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Translation batch size")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--min-new-tokens", type=int, default=DEFAULT_MIN_NEW_TOKENS)
    parser.add_argument("--skip-comet", action="store_true")
    args = parser.parse_args()

    if os.path.exists(args.output):
        print(f"Resuming from existing output: {args.output}")
        expanded_rows = read_json_or_jsonl(args.output)
    else:
        data = read_json_or_jsonl(args.input)
        if not data:
            raise ValueError("Input dataset is empty")
        expanded_rows = build_rows_from_input(data)
        write_jsonl(args.output, expanded_rows)
        print(f"Initialized output file with {len(expanded_rows)} rows: {args.output}")

    translator = TowerTranslator(
        model_name=MODEL_NAME,
        default_max_new_tokens=args.max_new_tokens,
        default_min_new_tokens=args.min_new_tokens,
    )

    translate_documents(
        expanded_rows=expanded_rows,
        translator=translator,
        output_path=args.output,
        batch_size=args.batch_size,
        save_every_batch=True,
    )

    if not args.skip_comet:
        score_with_comet(expanded_rows, args.output)

    summary = compute_correlations(expanded_rows)
    with open(args.summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Wrote scored translations to {args.output}")
    print(f"Wrote summary to {args.summary}")


if __name__ == "__main__":
    main()
