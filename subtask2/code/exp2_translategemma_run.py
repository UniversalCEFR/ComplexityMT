"""
Exp 2 - TranslateGemma end-to-end runner (Colab-friendly).

Produces the same set of result CSV/JSON files as
    exp2_gtrans_classify.py        (forward)
    exp2_gtrans_back_classify.py   (backtranslation)
but using google/translategemma-4b-it or google/translategemma-12b-it as the
MT system, with output slug 'translategemma-4b' or 'translategemma-12b'.

The downstream visualizers (visualize_exp2.py / visualize_exp2_backtranslate.py
/ visualize_exp2_heatmap.py) and the rebuild_perlevel_rows scripts pick up
these files automatically once the slug is added to the slugs list.

==============================================================================
Colab setup (no Drive mount — files dropped straight into /content)
==============================================================================

Upload via the Colab Files panel into /content:
  - exp2_translategemma_run.py
  - sentence_level.json
  - document_level.json

Then in a notebook cell:

    # 1) Install deps (don't upgrade pandas — Colab's 2.2.2 is fine and pandas 3.x
    #    conflicts with gradio / cudf / google-colab)
    !pip install -q -U transformers accelerate sentencepiece bitsandbytes

    # 2) Authenticate (translategemma + UniversalCEFR are gated repos)
    from huggingface_hub import login
    login(token='hf_xxx')   # your HF token with access to both repos

    # 3) Run 4B in bf16
    !python exp2_translategemma_run.py --size 4b

    # 4) Run 12B in bf16 (A100 40GB fits this with room to spare)
    !python exp2_translategemma_run.py --size 12b

    # 5) Zip results and download before disconnect
    !zip -r results_translategemma.zip results/
    from google.colab import files; files.download('results_translategemma.zip')

The script is resumable: per-pair output CSVs are written incrementally and a
re-run skips any (split, src, tgt) pair whose forward file already exists.
Without Drive, a runtime disconnect wipes /content — download results/ before
that happens, or run --size 4b and --size 12b in separate sessions.
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
    AutoModelForImageTextToText,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)


TRANSLATEGEMMA_MODELS = {
    "4b": "google/translategemma-4b-it",
    "12b": "google/translategemma-12b-it",
}
CLASSIFIER_MODEL = "UniversalCEFR/xlm-roberta-base-cefr-all-classifier"
CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]
CEFR_TO_INT = {l: i for i, l in enumerate(CEFR_LEVELS)}
INT_TO_CEFR = {i: l for i, l in enumerate(CEFR_LEVELS)}

# Pattern matching per-pair files only (excludes pairs_all / pair_summary /
# pair_by_level / all / en_to_x / x_to_en aggregates that share a prefix).
# Bug history: forward aggregates written by write_forward_aggregates were
# being re-read by load_combined_forward in the back stage, doubling the
# rows the back pairs processed. Filtering on this regex fixes that.
REAL_LANGS = "(?:en|fr|de|nl|ar|hi|ru)"
PER_PAIR_FORWARD_RE = re.compile(
    r"^exp2_(?:sentence|document)_%s_to_%s_.+\.csv$" % (REAL_LANGS, REAL_LANGS)
)
PER_PAIR_BACK_RE = re.compile(
    r"^exp2_(?:sentence|document)_back_%s_to_%s_.+\.csv$" % (REAL_LANGS, REAL_LANGS)
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--size", choices=list(TRANSLATEGEMMA_MODELS), required=True,
                   help="TranslateGemma model size: 4b or 12b")
    p.add_argument("--splits-dir", default=".",
                   help="Directory containing sentence_level.json and document_level.json")
    p.add_argument("--out-dir", default="results",
                   help="Where to write per-pair, combined, and summary CSV/JSON files")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
                   help="HuggingFace token (or set $HF_TOKEN)")
    p.add_argument("--quant-4bit", action="store_true",
                   help="Load TranslateGemma in 4-bit (bitsandbytes) — needed for 12B on <24GB GPUs")
    p.add_argument("--mt-batch", type=int, default=8,
                   help="Batch size for translation generation")
    p.add_argument("--cls-batch", type=int, default=32,
                   help="Batch size for the CEFR classifier")
    p.add_argument("--max-new-tokens", type=int, default=512,
                   help="Max new tokens per translation (raise for long doc-level texts)")
    p.add_argument("--splits", nargs="+", default=["sentence", "document"],
                   choices=["sentence", "document"])
    p.add_argument("--stage", nargs="+", default=["forward", "back"],
                   choices=["forward", "back"],
                   help="Which stages to run. 'back' requires forward results already on disk.")
    p.add_argument("--limit-per-pair", type=int, default=None,
                   help="Process only the first N samples per pair (debugging)")
    return p.parse_args()


def slug_for(size):
    return "translategemma-%s" % size


def mt_model_label(size):
    return "translategemma-%s-it" % size


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def load_split_samples(splits_dir, split):
    path = os.path.join(splits_dir, "%s_level.json" % split)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def pairs_for(samples):
    langs = sorted(set(s["lang"] for s in samples))
    return [(a, b) for a in langs for b in langs if a != b]


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------

def load_translategemma(size, quant_4bit, hf_token):
    repo = TRANSLATEGEMMA_MODELS[size]
    print("Loading TranslateGemma: %s (quant_4bit=%s)" % (repo, quant_4bit))
    processor = AutoProcessor.from_pretrained(repo, token=hf_token)
    # Generation requires left-padding so generated tokens start at consistent position.
    processor.tokenizer.padding_side = "left"
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
    model = AutoModelForImageTextToText.from_pretrained(repo, **kwargs)
    model.eval()
    return processor, model


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

def _build_message(text, src, tgt):
    return [{
        "role": "user",
        "content": [{
            "type": "text",
            "source_lang_code": src,
            "target_lang_code": tgt,
            "text": text,
        }],
    }]


def translate_batch(processor, model, texts, src, tgt, max_new_tokens):
    """Translate a list of texts src -> tgt using TranslateGemma. Returns list[str]."""
    if not texts:
        return []
    messages = [_build_message(t, src, tgt) for t in texts]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True,
        return_dict=True,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    generated = out[:, input_len:]
    return processor.batch_decode(generated, skip_special_tokens=True)


def predict_cefr_batch(tokenizer, clf, device, texts):
    """Return list of CEFR labels (or None for empty/None inputs)."""
    keep_idx, keep_texts = [], []
    for i, t in enumerate(texts):
        if t is not None and not (isinstance(t, float) and pd.isna(t)) and str(t).strip():
            keep_idx.append(i)
            keep_texts.append(str(t))
    preds = [None] * len(texts)
    if not keep_texts:
        return preds
    enc = tokenizer(keep_texts, return_tensors="pt", truncation=True, max_length=512, padding=True)
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


def append_rows(path, rows, cols):
    """Append rows to a CSV (write header if the file does not yet exist)."""
    df = pd.DataFrame(rows)[cols]
    write_header = not os.path.exists(path)
    df.to_csv(path, mode="w" if write_header else "a",
              header=write_header, index=False, encoding="utf-8")


def existing_partial_count(partial_path):
    """Return how many data rows are already in a .partial CSV (0 if missing/corrupt)."""
    if not os.path.exists(partial_path):
        return 0
    try:
        return len(pd.read_csv(partial_path))
    except Exception:
        print("[resume] partial file %s unreadable; restarting that pair" % partial_path)
        os.remove(partial_path)
        return 0


# -----------------------------------------------------------------------------
# Forward stage
# -----------------------------------------------------------------------------

FORWARD_COLS = ["direction", "source_lang", "target_lang", "source_name", "format",
                "mt_model", "cefr_orig", "text_orig", "text_translated",
                "cefr_pred", "delta_l"]


def forward_pair_path(out_dir, split, src, tgt, size):
    return os.path.join(out_dir, "exp2_%s_%s_to_%s_%s.csv" % (split, src, tgt, slug_for(size)))


def run_forward_pair(samples, src, tgt, split, processor, mt_model,
                     clf_tok, clf_model, clf_device, args):
    final_path = forward_pair_path(args.out_dir, split, src, tgt, args.size)
    partial_path = final_path + ".partial"

    if os.path.exists(final_path):
        print("[forward] skip existing %s" % final_path)
        return

    src_samples = [s for s in samples if s["lang"] == src]
    if args.limit_per_pair:
        src_samples = src_samples[:args.limit_per_pair]

    done = existing_partial_count(partial_path)
    if done:
        print("[forward] resume %s->%s [%s]: %d/%d rows already on disk"
              % (src, tgt, split, done, len(src_samples)))

    remaining = src_samples[done:]
    if not remaining:
        os.replace(partial_path, final_path)
        print("[forward] wrote %s (already complete on resume)" % final_path)
        return

    t_start = time.time()
    done_now = 0
    for batch_samples in chunked(remaining, args.mt_batch):
        texts = [s["text"] for s in batch_samples]
        translations = translate_batch(processor, mt_model, texts, src, tgt, args.max_new_tokens)
        preds = predict_cefr_batch(clf_tok, clf_model, clf_device, translations)

        rows = []
        for s, trans, pred in zip(batch_samples, translations, preds):
            orig = s["cefr_level"]
            delta = (CEFR_TO_INT[pred] - CEFR_TO_INT[orig]
                     if (pred in CEFR_TO_INT and orig in CEFR_TO_INT) else None)
            rows.append({
                "direction": "%s->%s" % (src.upper(), tgt.upper()),
                "source_lang": src,
                "target_lang": tgt,
                "source_name": s.get("source_name", ""),
                "format": split,
                "mt_model": mt_model_label(args.size),
                "cefr_orig": orig,
                "text_orig": s["text"],
                "text_translated": trans,
                "cefr_pred": pred,
                "delta_l": delta,
            })
        append_rows(partial_path, rows, FORWARD_COLS)

        done_now += len(batch_samples)
        elapsed = time.time() - t_start
        rate = done_now / max(elapsed, 1e-6)
        eta = (len(remaining) - done_now) / max(rate, 1e-6)
        print("  %s->%s [%s]: %d/%d  (%.1f/s, ETA %.0fs)"
              % (src, tgt, split, done + done_now, len(src_samples), rate, eta), flush=True)

    os.replace(partial_path, final_path)
    print("[forward] wrote %s" % final_path)


def run_forward_stage(args, processor, mt_model, clf_tok, clf_model, clf_device):
    for split in args.splits:
        samples = load_split_samples(args.splits_dir, split)
        for src, tgt in pairs_for(samples):
            run_forward_pair(samples, src, tgt, split, processor, mt_model,
                             clf_tok, clf_model, clf_device, args)


# -----------------------------------------------------------------------------
# Backtranslation stage
# -----------------------------------------------------------------------------

BACK_COLS = ["round_trip", "source_lang", "pivot_lang", "format", "mt_model",
             "cefr_orig", "cefr_forward", "cefr_back",
             "text_orig", "text_forward", "text_back",
             "delta_l_forward", "delta_l_back"]


def back_pair_path(out_dir, split, src, pivot, size):
    return os.path.join(out_dir, "exp2_%s_back_%s_to_%s_%s.csv"
                        % (split, src, pivot, slug_for(size)))


def run_back_pair(forward_df, src, pivot, split, processor, mt_model,
                  clf_tok, clf_model, clf_device, args):
    final_path = back_pair_path(args.out_dir, split, src, pivot, args.size)
    partial_path = final_path + ".partial"

    if os.path.exists(final_path):
        print("[back] skip existing %s" % final_path)
        return

    fwd = forward_df[(forward_df["source_lang"] == src) & (forward_df["target_lang"] == pivot)].copy()
    if fwd.empty:
        return
    if args.limit_per_pair:
        fwd = fwd.head(args.limit_per_pair)
    fwd_records = fwd.to_dict(orient="records")

    done = existing_partial_count(partial_path)
    if done:
        print("[back] resume %s->%s->%s [%s]: %d/%d rows already on disk"
              % (src, pivot, src, split, done, len(fwd_records)))

    remaining = fwd_records[done:]
    if not remaining:
        os.replace(partial_path, final_path)
        print("[back] wrote %s (already complete on resume)" % final_path)
        return

    t_start = time.time()
    done_now = 0
    for batch_records in chunked(remaining, args.mt_batch):
        forward_texts = [str(r["text_translated"]) for r in batch_records]
        backs = translate_batch(processor, mt_model, forward_texts, pivot, src, args.max_new_tokens)
        cefr_backs = predict_cefr_batch(clf_tok, clf_model, clf_device, backs)

        rows = []
        for fr, text_back, cefr_back in zip(batch_records, backs, cefr_backs):
            cefr_orig = fr["cefr_orig"]
            cefr_fwd = fr["cefr_pred"]
            delta_back = (CEFR_TO_INT[cefr_back] - CEFR_TO_INT[cefr_orig]
                          if cefr_back in CEFR_TO_INT and cefr_orig in CEFR_TO_INT else None)
            rows.append({
                "round_trip": "%s->%s->%s" % (src.upper(), pivot.upper(), src.upper()),
                "source_lang": src,
                "pivot_lang": pivot,
                "format": split,
                "mt_model": mt_model_label(args.size),
                "cefr_orig": cefr_orig,
                "cefr_forward": cefr_fwd,
                "cefr_back": cefr_back,
                "text_orig": fr["text_orig"],
                "text_forward": fr["text_translated"],
                "text_back": text_back,
                "delta_l_forward": fr["delta_l"],
                "delta_l_back": delta_back,
            })
        append_rows(partial_path, rows, BACK_COLS)

        done_now += len(batch_records)
        elapsed = time.time() - t_start
        rate = done_now / max(elapsed, 1e-6)
        eta = (len(remaining) - done_now) / max(rate, 1e-6)
        print("  back %s->%s->%s [%s]: %d/%d  (%.1f/s, ETA %.0fs)"
              % (src, pivot, src, split, done + done_now, len(fwd_records), rate, eta), flush=True)

    os.replace(partial_path, final_path)
    print("[back] wrote %s" % final_path)


def load_combined_forward(out_dir, split, size):
    """Load every forward per-pair CSV for this split + slug into one DataFrame.

    Strict match on per-pair filename pattern — aggregate files (pairs_all,
    pair_summary, pair_by_level, en_to_x, x_to_en) are excluded.
    """
    slug = slug_for(size)
    suffix = "_%s.csv" % slug
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


def run_back_stage(args, processor, mt_model, clf_tok, clf_model, clf_device):
    for split in args.splits:
        fwd_df = load_combined_forward(args.out_dir, split, args.size)
        if fwd_df.empty:
            print("[back] no forward results for %s — skipping" % split)
            continue
        pairs = sorted({(r["source_lang"], r["target_lang"]) for _, r in fwd_df.iterrows()})
        for src, pivot in pairs:
            run_back_pair(fwd_df, src, pivot, split, processor, mt_model,
                          clf_tok, clf_model, clf_device, args)


# -----------------------------------------------------------------------------
# Combined + summary writers (mirror gtrans schema exactly)
# -----------------------------------------------------------------------------

def write_forward_aggregates(out_dir, size, splits):
    slug = slug_for(size)
    for split in splits:
        df = load_combined_forward(out_dir, split, size)
        if df.empty:
            continue

        combined = df.copy()
        combined_cols = ["direction", "source_lang", "target_lang", "format",
                         "mt_model", "cefr_orig", "text_orig", "text_translated",
                         "cefr_pred", "delta_l"]
        combined = combined[[c for c in combined_cols if c in combined.columns]]
        combined_path = os.path.join(out_dir, "exp2_%s_pairs_all_%s.csv" % (split, slug))
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
        agg.to_csv(os.path.join(out_dir, "exp2_%s_pair_summary_%s.csv" % (split, slug)),
                   index=False, encoding="utf-8")

        by_level = (valid.groupby(["source_lang", "target_lang", "cefr_orig"])["delta_l"]
                         .agg(n="count", mean="mean", std="std").reset_index())
        by_level.to_csv(os.path.join(out_dir, "exp2_%s_pair_by_level_%s.csv" % (split, slug)),
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
        with open(os.path.join(out_dir, "exp2_%s_pair_summary_%s.json" % (split, slug)),
                  "w", encoding="utf-8") as f:
            json.dump(nested, f, indent=2)


def load_combined_back(out_dir, split, size):
    """Same as load_combined_forward but for back per-pair files. Excludes
    back_pairs_all and back_pair_summary aggregates."""
    slug = slug_for(size)
    suffix = "_%s.csv" % slug
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


def write_back_aggregates(out_dir, size, splits):
    slug = slug_for(size)
    for split in splits:
        df = load_combined_back(out_dir, split, size)
        if df.empty:
            continue
        combined_path = os.path.join(out_dir, "exp2_%s_back_pairs_all_%s.csv" % (split, slug))
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
        agg.to_csv(os.path.join(out_dir, "exp2_%s_back_pair_summary_%s.csv" % (split, slug)),
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
        with open(os.path.join(out_dir, "exp2_%s_back_pair_summary_%s.json" % (split, slug)),
                  "w", encoding="utf-8") as f:
            json.dump(nested, f, indent=2)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    processor, mt_model = load_translategemma(args.size, args.quant_4bit, args.hf_token)
    clf_tok, clf_model, clf_device = load_classifier(args.hf_token)

    if "forward" in args.stage:
        print("\n=== FORWARD ===")
        run_forward_stage(args, processor, mt_model, clf_tok, clf_model, clf_device)
        write_forward_aggregates(args.out_dir, args.size, args.splits)

    if "back" in args.stage:
        print("\n=== BACKTRANSLATION ===")
        run_back_stage(args, processor, mt_model, clf_tok, clf_model, clf_device)
        write_back_aggregates(args.out_dir, args.size, args.splits)

    print("\nDone. Output dir: %s" % os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()
