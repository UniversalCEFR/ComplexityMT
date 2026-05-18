"""
Exp 2 - Backtranslation (round-trip drift)

For each forward pair (src -> tgt) saved in results/exp2_{split}_{src}_to_{tgt}_{slug}.csv,
take the saved `text_translated` (in tgt) and translate it back to src.
Then classify CEFR on the back-translated text and compute:

    delta_l_forward = cefr_forward - cefr_orig   (already in input)
    delta_l_back    = cefr_back   - cefr_orig    (round-trip drift)

A near-zero `delta_l_back` means the round trip preserves CEFR; a non-zero value
exposes systematic translation bias of the MT model itself.

Usage:
    python exp2_gpt-5.4_back.py --model openai/gpt-5.4 --split sentence
    python exp2_gpt-5.4_back.py --model openai/gpt-5.4 --split document
    python exp2_gpt-5.4_back.py --model openai/gpt-5.4 --split sentence --pairs fr-en en-fr
"""

import argparse
import json
import os
import re
import time
from glob import glob
import pandas as pd
import torch
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForSequenceClassification

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
CLASSIFIER_MODEL = "UniversalCEFR/xlm-roberta-base-cefr-all-classifier"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]
CEFR_TO_INT = {l: i for i, l in enumerate(CEFR_LEVELS)}
INT_TO_CEFR = {i: l for i, l in enumerate(CEFR_LEVELS)}

LANG_NAMES = {
    "fr": "French", "ru": "Russian", "ar": "Arabic", "hi": "Hindi",
    "en": "English", "de": "German", "nl": "Dutch", "es": "Spanish",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="OpenRouter model id, e.g. openai/gpt-5.4")
    p.add_argument("--split", required=True, choices=["sentence", "document"])
    p.add_argument("--pairs", nargs="+", default=None,
                   help="Restrict to specific forward pairs like 'fr-en en-de'. Default: all per-pair files for slug.")
    p.add_argument("--api-key", default=None)
    p.add_argument("--hf-token", default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--skip-existing", action="store_true")
    return p.parse_args()


def model_slug(model_id):
    name = model_id.split("/")[-1]
    return re.sub(r"[^a-zA-Z0-9._-]", "_", name)


def create_client(api_key):
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def translate(client, model_id, text, source_lang, target_lang, is_document, max_retries=3):
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return None
    src = LANG_NAMES.get(source_lang, source_lang)
    tgt = LANG_NAMES.get(target_lang, target_lang)
    unit = "text" if is_document else "sentence"
    prompt = "Translate the following %s %s to %s. Output only the translation, nothing else.\n\n%s" % (
        src, unit, tgt, text)
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=2048 if is_document else 512,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print("  Retry %d/%d: %s" % (attempt + 1, max_retries, e))
            time.sleep(2 ** attempt)
    return None


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
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return None
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_idx = torch.argmax(logits, dim=-1).item()
    label = model.config.id2label.get(pred_idx, INT_TO_CEFR.get(pred_idx, str(pred_idx)))
    return label


def discover_pairs(split, slug):
    pat = os.path.join(RESULTS_DIR, "exp2_%s_*_to_*_%s.csv" % (split, slug))
    files = sorted(glob(pat))
    pairs = []
    rx = re.compile(r"exp2_%s_([a-z]{2})_to_([a-z]{2})_%s\.csv$" % (split, re.escape(slug)))
    for f in files:
        m = rx.search(f.replace("\\", "/"))
        if m:
            pairs.append((m.group(1), m.group(2), f))
    # Skip backtranslation outputs that may already exist
    pairs = [(s, t, f) for (s, t, f) in pairs if "_back" not in os.path.basename(f)]
    return pairs


def run_backtrans_pair(src, tgt, fwd_path, args, client, tokenizer, model, slug):
    is_document = (args.split == "document")
    out_name = "exp2_%s_back_%s_to_%s_%s.csv" % (args.split, src, tgt, slug)
    out_path = os.path.join(RESULTS_DIR, out_name)

    if args.skip_existing and os.path.exists(out_path):
        print("\n[skip] %s already exists" % out_name)
        return pd.read_csv(out_path).to_dict("records")

    fwd = pd.read_csv(fwd_path)
    if args.max_samples:
        fwd = fwd.head(args.max_samples).copy()

    print("\n" + "=" * 60)
    print("Round trip: %s -> %s -> %s  (n=%d, %s-level) [%s]" % (
        src.upper(), tgt.upper(), src.upper(), len(fwd), args.split, args.model
    ))
    print("=" * 60)

    rows = []
    total = len(fwd)
    for i, r in fwd.iterrows():
        text_back = translate(
            client, args.model, r["text_translated"],
            source_lang=tgt, target_lang=src, is_document=is_document,
        )
        cefr_back = predict_cefr(tokenizer, model, text_back)
        cefr_orig = r["cefr_orig"]
        cefr_fwd = r.get("cefr_pred")

        delta_fwd = (CEFR_TO_INT[cefr_fwd] - CEFR_TO_INT[cefr_orig]
                     if (cefr_fwd in CEFR_TO_INT and cefr_orig in CEFR_TO_INT) else None)
        delta_back = (CEFR_TO_INT[cefr_back] - CEFR_TO_INT[cefr_orig]
                      if (cefr_back in CEFR_TO_INT and cefr_orig in CEFR_TO_INT) else None)

        rows.append({
            "round_trip": "%s->%s->%s" % (src.upper(), tgt.upper(), src.upper()),
            "source_lang": src,
            "pivot_lang": tgt,
            "source_name": r.get("source_name"),
            "format": r.get("format"),
            "mt_model": args.model,
            "cefr_orig": cefr_orig,
            "cefr_forward": cefr_fwd,
            "cefr_back": cefr_back,
            "text_orig": r["text_orig"],
            "text_forward": r["text_translated"],
            "text_back": text_back,
            "delta_l_forward": delta_fwd,
            "delta_l_back": delta_back,
        })

        done = len(rows)
        if done % 20 == 0 or done == total:
            print("  [%s->%s->%s] %d/%d" % (src.upper(), tgt.upper(), src.upper(), done, total))

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, encoding="utf-8")
    print("Saved %d rows to %s" % (len(df), out_path))

    valid = df.dropna(subset=["delta_l_back"])
    if len(valid):
        print("Mean delta_l_back: %.3f  (forward was %.3f)" % (
            valid["delta_l_back"].mean(),
            valid["delta_l_forward"].dropna().mean() if valid["delta_l_forward"].notna().any() else float("nan"),
        ))
    return rows


def write_backtrans_summary(df_all, split, slug):
    valid = df_all.dropna(subset=["delta_l_back"]).copy()
    if valid.empty:
        return
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
    csv_path = os.path.join(RESULTS_DIR, "exp2_%s_back_pair_summary_%s.csv" % (split, slug))
    agg.to_csv(csv_path, index=False, encoding="utf-8")
    print("Saved backtranslation summary -> %s" % csv_path)

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
    json_path = os.path.join(RESULTS_DIR, "exp2_%s_back_pair_summary_%s.json" % (split, slug))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(nested, f, indent=2)
    print("Saved backtranslation summary JSON -> %s" % json_path)


def main():
    args = parse_args()
    slug = model_slug(args.model)
    api_key = args.api_key or OPENROUTER_API_KEY

    print("MT Model:    %s" % args.model)
    print("Classifier:  %s" % CLASSIFIER_MODEL)
    print("Split:       %s" % args.split)

    pairs = discover_pairs(args.split, slug)
    if args.pairs:
        wanted = set()
        for p in args.pairs:
            s, t = p.lower().split("-")
            wanted.add((s, t))
        pairs = [p for p in pairs if (p[0], p[1]) in wanted]

    if not pairs:
        print("No forward per-pair CSVs found in %s for slug=%s split=%s" % (RESULTS_DIR, slug, args.split))
        return

    print("Discovered %d forward pair(s):" % len(pairs))
    for s, t, f in pairs:
        print("  %s -> %s  (%s)" % (s.upper(), t.upper(), os.path.basename(f)))

    client = create_client(api_key)
    tokenizer, model = load_classifier(CLASSIFIER_MODEL, hf_token=args.hf_token)

    all_rows = []
    for src, tgt, fwd_path in pairs:
        rows = run_backtrans_pair(src, tgt, fwd_path, args, client, tokenizer, model, slug)
        all_rows.extend(rows)

    if all_rows:
        df_all = pd.DataFrame(all_rows)
        out_all = os.path.join(RESULTS_DIR, "exp2_%s_back_pairs_all_%s.csv" % (args.split, slug))
        df_all.to_csv(out_all, index=False, encoding="utf-8")
        print("\nSaved combined backtranslation -> %s" % out_all)
        write_backtrans_summary(df_all, args.split, slug)


if __name__ == "__main__":
    main()
