"""
Experiment 2 - Label Shift Approximation (One-Way Analysis)
RQ: Does MT systematically up-level or down-level CEFR text complexity?
Metric: delta_l = l_trans - l_orig (shift in CEFR label after translation)

Pipeline:
1. Load data from shared splits (splits/sentence_level.json or splits/document_level.json)
2. Direction 1 (X->EN): Translate non-EN texts to English via MT model
3. Direction 2 (EN->X): Translate EN texts to other languages via MT model
4. Predict CEFR level on translated text using UniversalCEFR classifier
5. Compute delta_l and save all intermediate results

Usage:
    # Sentence-level with GPT-5.2
    python exp2_gpt-5.4_forward.py --model openai/gpt-5.2 --split sentence

    # Document-level with GPT-5.2
    python exp2_gpt-5.4_forward.py --model openai/gpt-5.2 --split document

    # Sentence-level, only FR<->EN
    python exp2_gpt-5.4_forward.py --model openai/gpt-5.2 --split sentence --source-langs fr --target-langs fr

    # Both splits
    python exp2_gpt-5.4_forward.py --model openai/gpt-5.2 --split sentence document

    # List suggested models
    python exp2_gpt-5.4_forward.py --model x --list-models
"""

import argparse
import json
import os
import re
import time
from collections import Counter
import pandas as pd
import torch
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Config ──────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
CLASSIFIER_MODEL = "UniversalCEFR/xlm-roberta-base-cefr-all-classifier"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPLITS_DIR = os.path.join(BASE_DIR, "splits")
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Exp 2: Label Shift Approximation - translate CEFR-labeled text and measure delta_l"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="OpenRouter model ID for translation (e.g., openai/gpt-5.2, google/gemini-2.5-flash)"
    )
    parser.add_argument(
        "--split", type=str, nargs="+", default=["sentence"],
        choices=["sentence", "document"],
        help="Which data split(s) to run (default: sentence)"
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="OpenRouter API key (overrides built-in key)"
    )
    parser.add_argument(
        "--classifier", type=str, default=CLASSIFIER_MODEL,
        help="HuggingFace CEFR classifier model (default: %s)" % CLASSIFIER_MODEL
    )
    parser.add_argument(
        "--directions", type=str, nargs="+", default=["x-to-en", "en-to-x"],
        choices=["x-to-en", "en-to-x"],
        help="Which translation directions to run (default: both). Ignored if --pairs is set."
    )
    parser.add_argument(
        "--pairs", type=str, nargs="+", default=None,
        help="Explicit directed pairs like 'fr-en en-fr de-ru'. Overrides --directions/--source-langs/--target-langs."
    )
    parser.add_argument(
        "--all-pairs", action="store_true",
        help="Run the full cross-product of all languages present in the split (overrides --pairs)."
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip pairs whose output CSV already exists."
    )
    parser.add_argument(
        "--source-langs", type=str, nargs="+", default=None,
        help="Source languages for X->EN direction (default: all non-EN in split)"
    )
    parser.add_argument(
        "--target-langs", type=str, nargs="+", default=None,
        help="Target languages for EN->X direction (default: all non-EN in split)"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Max samples per language/direction for testing (default: all)"
    )
    parser.add_argument(
        "--hf-token", type=str, default=None,
        help="HuggingFace token for private classifier model (overrides built-in)"
    )
    parser.add_argument(
        "--list-models", action="store_true",
        help="Print suggested OpenRouter models and exit"
    )
    return parser.parse_args()


SUGGESTED_MODELS = [
    "openai/gpt-5.4",
    "openai/gpt-5.2",
    "openai/gpt-4.1",
    "anthropic/claude-sonnet-4",
    "google/gemini-2.5-flash",
    "meta-llama/llama-4-maverick",
    "deepseek/deepseek-r1",
    "mistralai/mistral-large-2",
    "qwen/qwen3-235b-a22b",
]

CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]
CEFR_TO_INT = {level: i for i, level in enumerate(CEFR_LEVELS)}
INT_TO_CEFR = {i: level for i, level in enumerate(CEFR_LEVELS)}

LANG_NAMES = {
    "fr": "French",
    "ru": "Russian",
    "ar": "Arabic",
    "hi": "Hindi",
    "en": "English",
    "de": "German",
    "nl": "Dutch",
    "es": "Spanish",
}


# ── Step 1: Load data from shared splits ────────────────────────────────────
def load_split(split_name, source_langs=None, target_langs=None, max_samples=None):
    split_file = os.path.join(SPLITS_DIR, "%s_level.json" % split_name)
    with open(split_file, encoding="utf-8") as f:
        data = json.load(f)

    # Discover available non-EN languages in the split
    available_langs = sorted(set(d["lang"] for d in data if d["lang"] != "en"))
    source_langs = source_langs or available_langs
    target_langs = target_langs or available_langs

    # Filter to only requested source langs
    source_langs = [l for l in source_langs if l in available_langs]
    target_langs = [l for l in target_langs if l in available_langs]

    # Direction 1 (X->EN): non-EN texts in requested source languages
    x_to_en = [d for d in data if d["lang"] in source_langs]

    # Direction 2 (EN->X): EN texts
    en_to_x = [d for d in data if d["lang"] == "en"]

    # Apply max_samples limit per language
    if max_samples:
        limited = []
        for lang in source_langs:
            lang_data = [d for d in x_to_en if d["lang"] == lang]
            limited.extend(lang_data[:max_samples])
        x_to_en = limited
        en_to_x = en_to_x[:max_samples]

    print("Split: %s-level" % split_name)
    print("Direction 1 (X->EN): %d samples" % len(x_to_en))
    for lang in source_langs:
        count = sum(1 for d in x_to_en if d["lang"] == lang)
        levels = Counter(d["cefr_level"] for d in x_to_en if d["lang"] == lang)
        level_str = ", ".join("%s:%d" % (l, levels[l]) for l in CEFR_LEVELS if l in levels)
        print("  %s: %d (%s)" % (lang, count, level_str))

    print("Direction 2 (EN->X): %d samples -> %s" % (len(en_to_x), [l.upper() for l in target_langs]))
    for src in sorted(set(d["source_name"] for d in en_to_x)):
        count = sum(1 for d in en_to_x if d["source_name"] == src)
        print("  %s: %d" % (src, count))

    return x_to_en, en_to_x, source_langs, target_langs


# ── Step 2: Translate via OpenRouter ─────────────────────────────────────────
def create_translation_client(api_key):
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def translate_text(client, model_id, text, source_lang, target_lang, is_document=False, max_retries=3):
    source_name = LANG_NAMES.get(source_lang, source_lang)
    target_name = LANG_NAMES.get(target_lang, target_lang)
    if is_document:
        prompt = "Translate the following %s text to %s. Output only the translation, nothing else.\n\n%s" % (source_name, target_name, text)
    else:
        prompt = "Translate the following %s sentence to %s. Output only the translation, nothing else.\n\n%s" % (source_name, target_name, text)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=2048 if is_document else 512,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print("  Retry %d/%d: %s" % (attempt + 1, max_retries, e))
            time.sleep(2 ** attempt)
    return None


def translate_batch(client, model_id, samples, target_lang, direction_label, is_document=False):
    results = []
    total = len(samples)
    for i, sample in enumerate(samples):
        source_lang = sample["lang"]
        translation = translate_text(client, model_id, sample["text"], source_lang, target_lang, is_document=is_document)
        results.append({
            "direction": direction_label,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "source_name": sample["source_name"],
            "format": sample["format"],
            "mt_model": model_id,
            "cefr_orig": sample["cefr_level"],
            "text_orig": sample["text"],
            "text_translated": translation,
        })
        if (i + 1) % 20 == 0 or (i + 1) == total:
            print("  [%s] Translated %d/%d" % (direction_label, i + 1, total))
    return results


# ── Step 3: Predict CEFR with UniversalCEFR classifier ──────────────────────
def load_classifier(model_name, hf_token=None):
    token = hf_token or HF_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    os.environ["HF_TOKEN"] = token
    print("Loading classifier: %s" % model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, use_auth_token=token)
    model.eval()
    return tokenizer, model


def predict_cefr(tokenizer, model, text):
    if text is None:
        return None
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_idx = torch.argmax(logits, dim=-1).item()
    label = model.config.id2label.get(pred_idx, INT_TO_CEFR.get(pred_idx, str(pred_idx)))
    return label


def predict_batch(tokenizer, model, results):
    total = len(results)
    for i, r in enumerate(results):
        r["cefr_pred"] = predict_cefr(tokenizer, model, r["text_translated"])
        if (i + 1) % 50 == 0 or (i + 1) == total:
            print("  Predicted %d/%d" % (i + 1, total))
    return results


# ── Step 4: Compute delta_l ─────────────────────────────────────────────────
def compute_delta(results):
    for r in results:
        orig = r["cefr_orig"]
        pred = r["cefr_pred"]
        if orig in CEFR_TO_INT and pred in CEFR_TO_INT:
            r["delta_l"] = CEFR_TO_INT[pred] - CEFR_TO_INT[orig]
        else:
            r["delta_l"] = None
    return results


# ── Step 5: Save results ────────────────────────────────────────────────────
def model_slug(model_id):
    """Convert 'openai/gpt-5.2' -> 'gpt-5.2' for filenames."""
    name = model_id.split("/")[-1]
    return re.sub(r"[^a-zA-Z0-9._-]", "_", name)


def save_results(results, filename):
    df = pd.DataFrame(results)
    filepath = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(filepath, index=False, encoding="utf-8")
    print("Saved %d rows to %s" % (len(df), filepath))
    return df


def print_summary(df, direction_label):
    print("\n" + "=" * 60)
    print("Summary: %s" % direction_label)
    print("=" * 60)

    valid = df.dropna(subset=["delta_l"])
    print("Total samples: %d" % len(valid))
    print("Mean delta_l: %.3f" % valid["delta_l"].mean())
    print("Std delta_l:  %.3f" % valid["delta_l"].std())

    print("\ndelta_l distribution:")
    print(valid["delta_l"].value_counts().sort_index().to_string())

    if "source_lang" in valid.columns:
        print("\nMean delta_l by source language:")
        print(valid.groupby("source_lang")["delta_l"].agg(["mean", "std", "count"]).to_string())

    print("\nMean delta_l by original CEFR level:")
    print(valid.groupby("cefr_orig")["delta_l"].agg(["mean", "std", "count"]).to_string())


# ── Pair-summary export (heatmap-ready) ─────────────────────────────────────
def write_pair_summary(df_all, split_name, slug):
    """Emit (src,tgt) matrix + (src,tgt,cefr_orig) breakdown as CSV and JSON."""
    valid = df_all.dropna(subset=["delta_l"]).copy()
    if valid.empty:
        return

    # (src, tgt) matrix
    agg = (
        valid.groupby(["source_lang", "target_lang"])["delta_l"]
        .agg(n="count", mean="mean", std="std", median="median")
        .reset_index()
    )
    agg["pct_up"] = (
        valid.assign(up=(valid["delta_l"] > 0).astype(int))
             .groupby(["source_lang", "target_lang"])["up"].mean().values
    )
    agg["pct_down"] = (
        valid.assign(dn=(valid["delta_l"] < 0).astype(int))
             .groupby(["source_lang", "target_lang"])["dn"].mean().values
    )
    agg["pct_preserved"] = (
        valid.assign(eq=(valid["delta_l"] == 0).astype(int))
             .groupby(["source_lang", "target_lang"])["eq"].mean().values
    )

    csv_path = os.path.join(OUTPUT_DIR, "exp2_%s_pair_summary_%s.csv" % (split_name, slug))
    agg.to_csv(csv_path, index=False, encoding="utf-8")
    print("Saved pair summary -> %s" % csv_path)

    # (src, tgt, cefr_orig) breakdown
    by_level = (
        valid.groupby(["source_lang", "target_lang", "cefr_orig"])["delta_l"]
        .agg(n="count", mean="mean", std="std")
        .reset_index()
    )
    csv_lvl = os.path.join(OUTPUT_DIR, "exp2_%s_pair_by_level_%s.csv" % (split_name, slug))
    by_level.to_csv(csv_lvl, index=False, encoding="utf-8")
    print("Saved pair-by-level summary -> %s" % csv_lvl)

    # JSON nested by src -> tgt
    nested = {}
    for _, row in agg.iterrows():
        nested.setdefault(row["source_lang"], {})[row["target_lang"]] = {
            "n": int(row["n"]),
            "mean_delta_l": float(row["mean"]),
            "std_delta_l": float(row["std"]) if pd.notna(row["std"]) else None,
            "median_delta_l": float(row["median"]),
            "pct_up": float(row["pct_up"]),
            "pct_down": float(row["pct_down"]),
            "pct_preserved": float(row["pct_preserved"]),
        }
    json_path = os.path.join(OUTPUT_DIR, "exp2_%s_pair_summary_%s.json" % (split_name, slug))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(nested, f, indent=2)
    print("Saved pair summary JSON -> %s" % json_path)


# ── Run one directed pair (arbitrary src -> tgt) ────────────────────────────
def load_samples_for_lang(split_name, lang, max_samples=None):
    split_file = os.path.join(SPLITS_DIR, "%s_level.json" % split_name)
    with open(split_file, encoding="utf-8") as f:
        data = json.load(f)
    samples = [d for d in data if d["lang"] == lang]
    if max_samples:
        samples = samples[:max_samples]
    return samples


def run_pair(split_name, src, tgt, args, client, tokenizer, model, slug):
    is_document = (split_name == "document")
    out_name = "exp2_%s_%s_to_%s_%s.csv" % (split_name, src, tgt, slug)
    out_path = os.path.join(OUTPUT_DIR, out_name)

    if args.skip_existing and os.path.exists(out_path):
        print("\n[skip] %s already exists" % out_name)
        df = pd.read_csv(out_path)
        return df.to_dict("records")

    samples = load_samples_for_lang(split_name, src, max_samples=args.max_samples)
    if not samples:
        print("\n[warn] no samples for source=%s in %s-level split" % (src, split_name))
        return []

    print("\n" + "=" * 60)
    print("Pair: %s -> %s  (%d samples, %s-level) [%s]" % (
        src.upper(), tgt.upper(), len(samples), split_name, args.model
    ))
    print("=" * 60)

    direction_label = "%s->%s" % (src.upper(), tgt.upper())
    results = translate_batch(client, args.model, samples, tgt, direction_label, is_document=is_document)
    results = predict_batch(tokenizer, model, results)
    results = compute_delta(results)
    df = save_results(results, out_name)
    print_summary(df, direction_label)
    return results


# ── Run one split ───────────────────────────────────────────────────────────
def run_split(split_name, args, client, tokenizer, model, slug):
    is_document = (split_name == "document")

    print("\n" + "#" * 60)
    print("# SPLIT: %s-level" % split_name.upper())
    print("#" * 60)

    # ── Pair mode (arbitrary src -> tgt, supports full cross-product) ──
    if args.all_pairs or args.pairs:
        if args.all_pairs:
            split_file = os.path.join(SPLITS_DIR, "%s_level.json" % split_name)
            with open(split_file, encoding="utf-8") as f:
                data = json.load(f)
            langs = sorted(set(d["lang"] for d in data))
            pairs = [(s, t) for s in langs for t in langs if s != t]
        else:
            pairs = []
            for p in args.pairs:
                s, t = p.split("-")
                pairs.append((s.lower(), t.lower()))

        print("Running %d directed pair(s) for %s-level:" % (len(pairs), split_name))
        for s, t in pairs:
            print("  %s -> %s" % (s.upper(), t.upper()))

        all_results = []
        for s, t in pairs:
            results = run_pair(split_name, s, t, args, client, tokenizer, model, slug)
            all_results.extend(results)

        if all_results:
            df_all = save_results(all_results, "exp2_%s_pairs_all_%s.csv" % (split_name, slug))
            print("\n" + "=" * 60)
            print("PAIR SUMMARY: %s-level [%s]" % (split_name.upper(), args.model))
            print("=" * 60)
            valid = df_all.dropna(subset=["delta_l"])
            print("Total valid samples: %d" % len(valid))
            print("\nMean delta_l by pair:")
            print(valid.groupby("direction")["delta_l"].agg(["mean", "std", "count"]).to_string())
            write_pair_summary(df_all, split_name, slug)
        return all_results

    x_to_en, en_to_x, source_langs, target_langs = load_split(
        split_name,
        source_langs=args.source_langs,
        target_langs=args.target_langs,
        max_samples=args.max_samples,
    )

    all_results = []

    # ── Direction 1: X -> EN ────────────────────────────────────────────
    if "x-to-en" in args.directions and x_to_en:
        print("\n" + "=" * 60)
        lang_str = "/".join(l.upper() for l in source_langs)
        print("Direction 1: X -> EN (%s -> English) [%s]" % (lang_str, args.model))
        print("=" * 60)
        results_x_to_en = translate_batch(client, args.model, x_to_en, "en", "X->EN", is_document=is_document)
        results_x_to_en = predict_batch(tokenizer, model, results_x_to_en)
        results_x_to_en = compute_delta(results_x_to_en)
        df_x_to_en = save_results(results_x_to_en, "exp2_%s_x_to_en_%s.csv" % (split_name, slug))
        print_summary(df_x_to_en, "X -> EN (%s)" % split_name)
        all_results.extend(results_x_to_en)

    # ── Direction 2: EN -> X ────────────────────────────────────────────
    if "en-to-x" in args.directions and en_to_x:
        print("\n" + "=" * 60)
        lang_str = "/".join(l.upper() for l in target_langs)
        print("Direction 2: EN -> X (English -> %s) [%s]" % (lang_str, args.model))
        print("=" * 60)
        all_en_to_x = []
        for target_lang in target_langs:
            print("\nTranslating EN -> %s..." % target_lang.upper())
            results = translate_batch(
                client, args.model, en_to_x, target_lang,
                "EN->%s" % target_lang.upper(), is_document=is_document
            )
            results = predict_batch(tokenizer, model, results)
            results = compute_delta(results)
            all_en_to_x.extend(results)

        df_en_to_x = save_results(all_en_to_x, "exp2_%s_en_to_x_%s.csv" % (split_name, slug))
        print_summary(df_en_to_x, "EN -> X (%s)" % split_name)
        all_results.extend(all_en_to_x)

    # ── Combined for this split ─────────────────────────────────────────
    if all_results:
        df_all = save_results(all_results, "exp2_%s_all_%s.csv" % (split_name, slug))
        print("\n" + "=" * 60)
        print("SPLIT SUMMARY: %s-level [%s]" % (split_name.upper(), args.model))
        print("=" * 60)
        valid = df_all.dropna(subset=["delta_l"])
        print("Total valid samples: %d" % len(valid))
        print("Overall mean delta_l: %.3f" % valid["delta_l"].mean())
        print("\nMean delta_l by direction:")
        print(valid.groupby("direction")["delta_l"].agg(["mean", "std", "count"]).to_string())

    return all_results


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    if args.list_models:
        print("Suggested OpenRouter models for translation:")
        for m in SUGGESTED_MODELS:
            print("  %s" % m)
        print("\nUsage: python exp2_gpt-5.4_forward.py --model <model-id> --split sentence")
        return

    slug = model_slug(args.model)
    api_key = args.api_key or OPENROUTER_API_KEY

    print("MT Model:    %s" % args.model)
    print("Classifier:  %s" % args.classifier)
    print("Splits:      %s" % args.split)
    print("Directions:  %s" % args.directions)
    if args.source_langs:
        print("Source langs: %s" % args.source_langs)
    if args.target_langs:
        print("Target langs: %s" % args.target_langs)
    if args.max_samples:
        print("Max samples: %d per language" % args.max_samples)
    print("Output slug: %s" % slug)

    # Init translation client
    client = create_translation_client(api_key)

    # Init classifier
    tokenizer, model = load_classifier(args.classifier, hf_token=args.hf_token)

    # Run each split
    all_results = []
    for split_name in args.split:
        results = run_split(split_name, args, client, tokenizer, model, slug)
        all_results.extend(results)

    # Overall summary if multiple splits
    if len(args.split) > 1 and all_results:
        df_all = save_results(all_results, "exp2_combined_all_%s.csv" % slug)
        print("\n" + "=" * 60)
        print("OVERALL SUMMARY (all splits) [%s]" % args.model)
        print("=" * 60)
        valid = df_all.dropna(subset=["delta_l"])
        print("Total valid samples: %d" % len(valid))
        print("Overall mean delta_l: %.3f" % valid["delta_l"].mean())
        print("\nMean delta_l by split + direction:")
        valid["split_dir"] = valid["format"] + " | " + valid["direction"]
        print(valid.groupby("split_dir")["delta_l"].agg(["mean", "std", "count"]).to_string())


if __name__ == "__main__":
    main()
