"""
Exp 2 - TowerInstruct-7B-v0.2 end-to-end runner (Colab-friendly).

Produces the same set of result CSV/JSON files as exp2_translategemma_run.py
but using Unbabel/TowerInstruct-7B-v0.2 as the MT system, with output slug
'tower-7b'.

==============================================================================
Language coverage
==============================================================================

TowerInstruct was trained on 10 languages (EN, PT, ES, FR, DE, NL, IT, KO,
ZH, RU). Our splits also include AR and HI, which are *outside* its training
distribution — we still run those pairs to keep coverage apples-to-apples
with the other MT systems (gpt-5.4 / gtrans / translategemma), accepting
that quality on AR and HI may be poor. The drop in fidelity for those pairs
is itself a finding worth recording.

Full coverage matches translategemma:
    Sentence-level: 6 langs -> 30 pairs -> 6,000 fwd + 6,000 back
    Document-level: 3 langs ->  6 pairs -> 1,030 fwd + 1,030 back
    Total:                                  ~14,000 MT calls

==============================================================================
Colab setup (no Drive mount)
==============================================================================

Upload via the Colab Files panel into /content:
  - exp2_tower_run.py
  - sentence_level.json
  - document_level.json

Then in a notebook cell:

    # 1) Install deps (don't upgrade pandas)
    !pip install -q -U transformers accelerate sentencepiece bitsandbytes

    # 2) Authenticate (XLM-R classifier is gated)
    from huggingface_hub import login
    login(token='hf_xxx')

    # 3) Run on A100 in bf16
    !python exp2_tower_run.py

    # 4) Zip + download
    !zip -r results.zip results/
    from google.colab import files; files.download('results.zip')

The script is resumable: a re-run skips any (split, src, tgt) pair whose
forward/back final CSV already exists.
"""

import argparse
import gc
import json
import os
import re
import time
from itertools import product

import pandas as pd
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)


TOWER_MODEL = "Unbabel/TowerInstruct-7B-v0.2"
SLUG = "tower-7b"
MT_MODEL_LABEL = "towerinstruct-7b-v0.2"

CLASSIFIER_MODEL = "UniversalCEFR/xlm-roberta-base-cefr-all-classifier"

# ISO -> language name used in the natural-language translation prompt.
# Languages outside TowerInstruct's training set (ar, hi) are included so that
# coverage matches the other MT systems; quality on those pairs may be poor.
LANG_NAMES = {
    "en": "English", "fr": "French", "de": "German", "nl": "Dutch",
    "ru": "Russian", "ar": "Arabic", "hi": "Hindi",
    "pt": "Portuguese", "es": "Spanish", "it": "Italian",
    "ko": "Korean", "zh": "Chinese",
}

CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]
CEFR_TO_INT = {l: i for i, l in enumerate(CEFR_LEVELS)}
INT_TO_CEFR = {i: l for i, l in enumerate(CEFR_LEVELS)}

# Per-pair file regex used by load_combined_* to exclude aggregate filenames.
REAL_LANGS = "(?:en|fr|de|nl|ar|hi|ru)"
PER_PAIR_FORWARD_RE = re.compile(
    r"^exp2_(?:sentence|document)_%s_to_%s_.+\.csv$" % (REAL_LANGS, REAL_LANGS)
)
PER_PAIR_BACK_RE = re.compile(
    r"^exp2_(?:sentence|document)_back_%s_to_%s_.+\.csv$" % (REAL_LANGS, REAL_LANGS)
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="results")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--quant-4bit", action="store_true",
                   help="Load Tower in 4-bit (bitsandbytes) — useful on <16GB GPUs")
    p.add_argument("--mt-batch", type=int, default=8)
    p.add_argument("--cls-batch", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--splits-dir", default=".")
    p.add_argument("--splits", nargs="+", default=["sentence", "document"],
                   choices=["sentence", "document"])
    p.add_argument("--stage", nargs="+", default=["forward", "back"],
                   choices=["forward", "back"])
    p.add_argument("--limit-per-pair", type=int, default=None)
    return p.parse_args()


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def load_split_samples(splits_dir, split):
    path = os.path.join(splits_dir, "%s_level.json" % split)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def pairs_for(samples):
    """All (src, tgt) pairs from the split's languages — no SUPPORTED filter."""
    langs = sorted(set(s["lang"] for s in samples))
    return [(a, b) for a in langs for b in langs if a != b]


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------

def load_tower(quant_4bit, hf_token):
    print("Loading TowerInstruct: %s (quant_4bit=%s)" % (TOWER_MODEL, quant_4bit))
    tokenizer = AutoTokenizer.from_pretrained(TOWER_MODEL, token=hf_token)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = dict(token=hf_token, device_map="auto")
    if quant_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(TOWER_MODEL, **kwargs)
    model.eval()
    return tokenizer, model


def load_classifier(hf_token):
    print("Loading classifier: %s" % CLASSIFIER_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_MODEL, token=hf_token)
    model = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER_MODEL, token=hf_token)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device


# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------

def _build_translation_prompt(text, src_iso, tgt_iso):
    """Natural-language translation prompt per the TowerInstruct model card."""
    src_name = LANG_NAMES[src_iso]
    tgt_name = LANG_NAMES[tgt_iso]
    user_msg = (
        "Translate the following text from %s into %s.\n"
        "%s: %s\n"
        "%s:"
    ) % (src_name, tgt_name, src_name, text, tgt_name)
    return [{"role": "user", "content": user_msg}]


def translate_batch(tokenizer, model, texts, src_iso, tgt_iso, max_new_tokens):
    if not texts:
        return []
    prompts = []
    for t in texts:
        msgs = _build_translation_prompt(t, src_iso, tgt_iso)
        prompts.append(tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True))
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                    max_length=2048).to(model.device)
    input_len = enc["input_ids"].shape[1]
    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = out[:, input_len:]
    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
    return [s.strip() for s in decoded]


def predict_cefr_batch(tokenizer, clf, device, texts):
    keep_idx, keep_texts = [], []
    for i, t in enumerate(texts):
        if t is not None and not (isinstance(t, float) and pd.isna(t)) and str(t).strip():
            keep_idx.append(i)
            keep_texts.append(str(t))
    preds = [None] * len(texts)
    if not keep_texts:
        return preds
    enc = tokenizer(keep_texts, return_tensors="pt", truncation=True,
                    max_length=512, padding=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.inference_mode():
        logits = clf(**enc).logits
    idxs = torch.argmax(logits, dim=-1).tolist()
    for j, i in enumerate(keep_idx):
        preds[i] = clf.config.id2label.get(idxs[j], INT_TO_CEFR.get(idxs[j], str(idxs[j])))
    return preds


def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


# -----------------------------------------------------------------------------
# Forward stage
# -----------------------------------------------------------------------------

FORWARD_COLS = ["direction", "source_lang", "target_lang", "source_name", "format",
                "mt_model", "cefr_orig", "text_orig", "text_translated",
                "cefr_pred", "delta_l"]


def forward_pair_path(out_dir, split, src, tgt):
    return os.path.join(out_dir, "exp2_%s_%s_to_%s_%s.csv" % (split, src, tgt, SLUG))


def run_forward_pair(samples, src, tgt, split, tokenizer_mt, mt_model,
                     clf_tok, clf_model, clf_device, args):
    src_samples = [s for s in samples if s["lang"] == src]
    if args.limit_per_pair:
        src_samples = src_samples[:args.limit_per_pair]
    texts = [s["text"] for s in src_samples]

    translations = []
    t0 = time.time()
    for batch in chunked(texts, args.mt_batch):
        translations.extend(translate_batch(tokenizer_mt, mt_model, batch, src, tgt, args.max_new_tokens))
    mt_time = time.time() - t0

    t1 = time.time()
    preds = []
    for batch in chunked(translations, args.cls_batch):
        preds.extend(predict_cefr_batch(clf_tok, clf_model, clf_device, batch))
    cls_time = time.time() - t1

    rows = []
    for s, trans, pred in zip(src_samples, translations, preds):
        orig = s["cefr_level"]
        delta = (CEFR_TO_INT[pred] - CEFR_TO_INT[orig]) if (pred in CEFR_TO_INT and orig in CEFR_TO_INT) else None
        rows.append({
            "direction": "%s->%s" % (src.upper(), tgt.upper()),
            "source_lang": src,
            "target_lang": tgt,
            "source_name": s.get("source_name", ""),
            "format": split,
            "mt_model": MT_MODEL_LABEL,
            "cefr_orig": orig,
            "text_orig": s["text"],
            "text_translated": trans,
            "cefr_pred": pred,
            "delta_l": delta,
        })
    print("  %s->%s [%s]: %d rows, MT %.1fs, CLS %.1fs"
          % (src, tgt, split, len(rows), mt_time, cls_time))
    return rows


def run_forward_stage(args, tokenizer_mt, mt_model, clf_tok, clf_model, clf_device):
    for split in args.splits:
        samples = load_split_samples(args.splits_dir, split)
        pairs = pairs_for(samples)
        print("[forward] %s: %d pairs" % (split, len(pairs)))
        for src, tgt in pairs:
            out_path = forward_pair_path(args.out_dir, split, src, tgt)
            if os.path.exists(out_path):
                print("[forward] skip existing %s" % out_path)
                continue
            rows = run_forward_pair(samples, src, tgt, split, tokenizer_mt, mt_model,
                                    clf_tok, clf_model, clf_device, args)
            pd.DataFrame(rows)[FORWARD_COLS].to_csv(out_path, index=False, encoding="utf-8")
            print("[forward] wrote %s" % out_path)


# -----------------------------------------------------------------------------
# Backtranslation stage
# -----------------------------------------------------------------------------

BACK_COLS = ["round_trip", "source_lang", "pivot_lang", "format", "mt_model",
             "cefr_orig", "cefr_forward", "cefr_back",
             "text_orig", "text_forward", "text_back",
             "delta_l_forward", "delta_l_back"]


def back_pair_path(out_dir, split, src, pivot):
    return os.path.join(out_dir, "exp2_%s_back_%s_to_%s_%s.csv" % (split, src, pivot, SLUG))


def run_back_pair(forward_df, src, pivot, split, tokenizer_mt, mt_model,
                  clf_tok, clf_model, clf_device, args):
    fwd = forward_df[(forward_df["source_lang"] == src) & (forward_df["target_lang"] == pivot)].copy()
    if fwd.empty:
        return []
    if args.limit_per_pair:
        fwd = fwd.head(args.limit_per_pair)

    forward_texts = fwd["text_translated"].astype(str).tolist()

    backs = []
    t0 = time.time()
    for batch in chunked(forward_texts, args.mt_batch):
        backs.extend(translate_batch(tokenizer_mt, mt_model, batch, pivot, src, args.max_new_tokens))
    mt_time = time.time() - t0

    t1 = time.time()
    cefr_backs = []
    for batch in chunked(backs, args.cls_batch):
        cefr_backs.extend(predict_cefr_batch(clf_tok, clf_model, clf_device, batch))
    cls_time = time.time() - t1

    rows = []
    for (_, fr), text_back, cefr_back in zip(fwd.iterrows(), backs, cefr_backs):
        cefr_orig = fr["cefr_orig"]
        cefr_fwd = fr["cefr_pred"]
        delta_back = (CEFR_TO_INT[cefr_back] - CEFR_TO_INT[cefr_orig]
                      if cefr_back in CEFR_TO_INT and cefr_orig in CEFR_TO_INT else None)
        rows.append({
            "round_trip": "%s->%s->%s" % (src.upper(), pivot.upper(), src.upper()),
            "source_lang": src,
            "pivot_lang": pivot,
            "format": split,
            "mt_model": MT_MODEL_LABEL,
            "cefr_orig": cefr_orig,
            "cefr_forward": cefr_fwd,
            "cefr_back": cefr_back,
            "text_orig": fr["text_orig"],
            "text_forward": fr["text_translated"],
            "text_back": text_back,
            "delta_l_forward": fr["delta_l"],
            "delta_l_back": delta_back,
        })
    print("  back %s->%s->%s [%s]: %d rows, MT %.1fs, CLS %.1fs"
          % (src, pivot, src, split, len(rows), mt_time, cls_time))
    return rows


def load_combined_forward(out_dir, split):
    """Load every forward per-pair CSV for this split + tower slug."""
    suffix = "_%s.csv" % SLUG
    pieces = []
    for fname in sorted(os.listdir(out_dir)):
        if not fname.startswith("exp2_%s_" % split):
            continue
        if not fname.endswith(suffix):
            continue
        if not PER_PAIR_FORWARD_RE.match(fname):
            continue
        pieces.append(pd.read_csv(os.path.join(out_dir, fname)))
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True)


def run_back_stage(args, tokenizer_mt, mt_model, clf_tok, clf_model, clf_device):
    for split in args.splits:
        fwd_df = load_combined_forward(args.out_dir, split)
        if fwd_df.empty:
            print("[back] no forward results for %s — skipping" % split)
            continue
        pairs = sorted({(r["source_lang"], r["target_lang"]) for _, r in fwd_df.iterrows()})
        for src, pivot in pairs:
            out_path = back_pair_path(args.out_dir, split, src, pivot)
            if os.path.exists(out_path):
                print("[back] skip existing %s" % out_path)
                continue
            rows = run_back_pair(fwd_df, src, pivot, split, tokenizer_mt, mt_model,
                                 clf_tok, clf_model, clf_device, args)
            if rows:
                pd.DataFrame(rows)[BACK_COLS].to_csv(out_path, index=False, encoding="utf-8")
                print("[back] wrote %s" % out_path)


# -----------------------------------------------------------------------------
# Combined + summary writers (mirror gtrans schema exactly)
# -----------------------------------------------------------------------------

def write_forward_aggregates(out_dir, splits):
    for split in splits:
        df = load_combined_forward(out_dir, split)
        if df.empty:
            continue
        combined_cols = ["direction", "source_lang", "target_lang", "format",
                         "mt_model", "cefr_orig", "text_orig", "text_translated",
                         "cefr_pred", "delta_l"]
        combined = df[[c for c in combined_cols if c in df.columns]]
        combined_path = os.path.join(out_dir, "exp2_%s_pairs_all_%s.csv" % (split, SLUG))
        combined.to_csv(combined_path, index=False, encoding="utf-8")
        print("[summary] wrote %s (%d rows)" % (combined_path, len(combined)))

        valid = df.dropna(subset=["delta_l"]).copy()
        if valid.empty:
            continue
        agg = (valid.groupby(["source_lang", "target_lang"])["delta_l"]
                    .agg(n="count", mean="mean", std="std", median="median")
                    .reset_index())
        agg["pct_up"] = (valid.assign(u=(valid["delta_l"] > 0).astype(int))
                              .groupby(["source_lang", "target_lang"])["u"].mean().values)
        agg["pct_down"] = (valid.assign(d=(valid["delta_l"] < 0).astype(int))
                                .groupby(["source_lang", "target_lang"])["d"].mean().values)
        agg["pct_preserved"] = (valid.assign(e=(valid["delta_l"] == 0).astype(int))
                                     .groupby(["source_lang", "target_lang"])["e"].mean().values)
        agg.to_csv(os.path.join(out_dir, "exp2_%s_pair_summary_%s.csv" % (split, SLUG)),
                   index=False, encoding="utf-8")

        by_level = (valid.groupby(["source_lang", "target_lang", "cefr_orig"])["delta_l"]
                         .agg(n="count", mean="mean", std="std").reset_index())
        by_level.to_csv(os.path.join(out_dir, "exp2_%s_pair_by_level_%s.csv" % (split, SLUG)),
                        index=False, encoding="utf-8")

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
        with open(os.path.join(out_dir, "exp2_%s_pair_summary_%s.json" % (split, SLUG)),
                  "w", encoding="utf-8") as f:
            json.dump(nested, f, indent=2)


def load_combined_back(out_dir, split):
    suffix = "_%s.csv" % SLUG
    pieces = []
    for fname in sorted(os.listdir(out_dir)):
        if not fname.startswith("exp2_%s_back_" % split):
            continue
        if not fname.endswith(suffix):
            continue
        if not PER_PAIR_BACK_RE.match(fname):
            continue
        pieces.append(pd.read_csv(os.path.join(out_dir, fname)))
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True)


def write_back_aggregates(out_dir, splits):
    for split in splits:
        df = load_combined_back(out_dir, split)
        if df.empty:
            continue
        combined_path = os.path.join(out_dir, "exp2_%s_back_pairs_all_%s.csv" % (split, SLUG))
        df[[c for c in BACK_COLS if c in df.columns]].to_csv(combined_path, index=False, encoding="utf-8")
        print("[summary] wrote %s (%d rows)" % (combined_path, len(df)))

        valid = df.dropna(subset=["delta_l_back"]).copy()
        if valid.empty:
            continue
        agg = (valid.groupby(["source_lang", "pivot_lang"])
                    .agg(n=("delta_l_back", "count"),
                         mean_back=("delta_l_back", "mean"),
                         std_back=("delta_l_back", "std"),
                         median_back=("delta_l_back", "median"),
                         mean_forward=("delta_l_forward", "mean"))
                    .reset_index())
        agg["pct_up_back"] = (valid.assign(u=(valid["delta_l_back"] > 0).astype(int))
                                    .groupby(["source_lang", "pivot_lang"])["u"].mean().values)
        agg["pct_down_back"] = (valid.assign(d=(valid["delta_l_back"] < 0).astype(int))
                                      .groupby(["source_lang", "pivot_lang"])["d"].mean().values)
        agg["pct_preserved_back"] = (valid.assign(e=(valid["delta_l_back"] == 0).astype(int))
                                           .groupby(["source_lang", "pivot_lang"])["e"].mean().values)
        agg.to_csv(os.path.join(out_dir, "exp2_%s_back_pair_summary_%s.csv" % (split, SLUG)),
                   index=False, encoding="utf-8")

        nested = {}
        for _, row in agg.iterrows():
            nested.setdefault(row["source_lang"], {})[row["pivot_lang"]] = {
                "n": int(row["n"]),
                "mean_delta_l_back": float(row["mean_back"]),
                "std_delta_l_back": float(row["std_back"]) if pd.notna(row["std_back"]) else None,
                "median_delta_l_back": float(row["median_back"]),
                "mean_delta_l_forward": float(row["mean_forward"]) if pd.notna(row["mean_forward"]) else None,
                "pct_up_back": float(row["pct_up_back"]),
                "pct_down_back": float(row["pct_down_back"]),
                "pct_preserved_back": float(row["pct_preserved_back"]),
            }
        with open(os.path.join(out_dir, "exp2_%s_back_pair_summary_%s.json" % (split, SLUG)),
                  "w", encoding="utf-8") as f:
            json.dump(nested, f, indent=2)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer_mt, mt_model = load_tower(args.quant_4bit, args.hf_token)
    clf_tok, clf_model, clf_device = load_classifier(args.hf_token)

    if "forward" in args.stage:
        print("\n=== FORWARD ===")
        run_forward_stage(args, tokenizer_mt, mt_model, clf_tok, clf_model, clf_device)
        write_forward_aggregates(args.out_dir, args.splits)

    if "back" in args.stage:
        print("\n=== BACKTRANSLATION ===")
        run_back_stage(args, tokenizer_mt, mt_model, clf_tok, clf_model, clf_device)
        write_back_aggregates(args.out_dir, args.splits)

    print("\nDone. Output dir: %s" % os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()
