#!/usr/bin/env python3

import argparse
import json
import math
import os
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import pipeline


LANG_NAME = {
    "nl": "Dutch",
    "en": "English",
    "fr": "French",
}

MODEL_NAME = "Unbabel/TowerInstruct-7B-v0.1"
DEFAULT_MAX_NEW_TOKENS = 2048
DEFAULT_MIN_NEW_TOKENS = 32
DEFAULT_BATCH_SIZE = 8

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
    with open(path, "r", encoding="utf-8") as handle:
        first = handle.read(1)
        handle.seek(0)

        if first == "[":
            data = json.load(handle)
            if not isinstance(data, list):
                raise ValueError("JSON file must contain a list of objects.")
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


def normalize_text_for_prompt(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def approx_word_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


class TowerBackTranslator:
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
            len(self.pipe.tokenizer(text, add_special_tokens=False)["input_ids"])
            for text in texts
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
        for output in outputs:
            if isinstance(output, list):
                generations.append(output[0]["generated_text"])
            else:
                generations.append(output["generated_text"])
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

        prompts = [self._build_prompt(text, src_lang, tgt_lang) for text in texts]
        raw = self.generate_batch(
            prompts=prompts,
            texts=texts,
            batch_size=batch_size,
        )
        cleaned = [self._clean_translation(output, tgt_lang) for output in raw]

        bad_indices = [
            index for index, (src, hyp) in enumerate(zip(texts, cleaned))
            if self._looks_bad(src, hyp)
        ]

        if bad_indices:
            retry_texts = [texts[index] for index in bad_indices]
            retry_prompts = [self._build_retry_prompt(text, src_lang, tgt_lang) for text in retry_texts]
            retry_raw = self.generate_batch(
                prompts=retry_prompts,
                texts=retry_texts,
                batch_size=min(batch_size, len(retry_texts)),
                max_new_tokens=min(4096, self.default_max_new_tokens * 2),
                min_new_tokens=max(self.default_min_new_tokens, 64),
            )
            retry_cleaned = [self._clean_translation(output, tgt_lang) for output in retry_raw]

            for index, hypothesis in zip(bad_indices, retry_cleaned):
                cleaned[index] = hypothesis

        return cleaned


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Existing translation JSON or JSONL")
    parser.add_argument("--output", required=True, help="Output JSONL with back translations")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--min-new-tokens", type=int, default=DEFAULT_MIN_NEW_TOKENS)
    return parser


def load_or_initialize_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if os.path.exists(args.output):
        rows = read_json_or_jsonl(args.output)
        if not rows:
            raise ValueError(f"Existing output file is empty: {args.output}")
        return rows

    rows = read_json_or_jsonl(args.input)
    if not rows:
        raise ValueError("Input dataset is empty")

    for row in rows:
        row.setdefault("back_translation", None)
        row.setdefault("back_translation_model", MODEL_NAME)

    write_jsonl(args.output, rows)
    print(f"Initialized output file with {len(rows)} rows: {args.output}")
    return rows


def build_direction_buckets(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str], List[int]]:
    groups: Dict[Tuple[str, str], List[int]] = defaultdict(list)

    for index, row in enumerate(rows):
        if row.get("back_translation"):
            continue
        if not row.get("translation"):
            continue

        reverse_source = row.get("target_lang")
        reverse_target = row.get("source_lang")
        if reverse_source not in LANG_NAME or reverse_target not in LANG_NAME:
            continue

        groups[(reverse_source, reverse_target)].append(index)

    for key in groups:
        groups[key].sort(key=lambda idx: len(str(rows[idx]["translation"])))
    return groups


def batched_indices(indices: List[int], batch_size: int) -> Iterable[List[int]]:
    for start in range(0, len(indices), batch_size):
        yield indices[start:start + batch_size]


def back_translate_documents(
    rows: List[Dict[str, Any]],
    translator: TowerBackTranslator,
    output_path: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    save_every_batch: bool = True,
) -> None:
    buckets = build_direction_buckets(rows)
    if not buckets:
        print("No pending back translations found.")
        return

    total_batches = sum(math.ceil(len(indices) / batch_size) for indices in buckets.values())
    progress = tqdm(total=total_batches, desc="Back-translating")

    for (src_lang, tgt_lang), indices in buckets.items():
        for batch in batched_indices(indices, batch_size):
            texts = [str(rows[index]["translation"]) for index in batch]
            back_translations = translator.translate_batch(
                texts=texts,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                batch_size=min(batch_size, len(batch)),
            )

            for index, back_translation in zip(batch, back_translations):
                rows[index]["back_translation"] = back_translation
                rows[index]["back_translation_model"] = MODEL_NAME

            if save_every_batch:
                write_jsonl(output_path, rows)

            progress.update(1)

    progress.close()

    if not save_every_batch:
        write_jsonl(output_path, rows)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    rows = load_or_initialize_rows(args)

    translator = TowerBackTranslator(
        model_name=MODEL_NAME,
        default_max_new_tokens=args.max_new_tokens,
        default_min_new_tokens=args.min_new_tokens,
    )

    back_translate_documents(
        rows=rows,
        translator=translator,
        output_path=args.output,
        batch_size=args.batch_size,
        save_every_batch=True,
    )

    print(f"Wrote back-translated rows to {args.output}")


if __name__ == "__main__":
    main()
