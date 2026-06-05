#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ComplexityMT -- Task 1 (Robustness)
===================================
Reproduces the Robustness results in Section 5.1 / Figures 2-4 of the paper
"Does Machine Translation Preserve Text Complexity?".

Task 1 asks RQ1: how does the CEFR level of a source text correlate with
machine-translation quality? For each MT system and granularity, we compute
the Spearman correlation between the source text's gold CEFR level and a
reference-free translation-quality score (COMET and GEMBA-DA), per
source -> target language pair. A significant negative correlation means
higher-CEFR (more complex) source texts receive lower-quality translations.

Input data (bundled, no network needed)
----------------------------------------
data/task1/task1_<system>_<granularity>.json, each of the form:
    {
      "contributor": ..., "mt_system": ..., "granularity": "sentence"|"document",
      "results": [
        {"src_lang": "en", "src_cefr": "C1", "tgt_lang": "fr",
         "src_text": ..., "tgt_text": ...,
         "score": {"comet": 0.83, "gemba": 97}},
        ...
      ]
    }

CEFR levels are mapped to an ordinal scale A1=1, A2=2, B1=3, B2=4, C1=5, C2=6
(Spearman uses ranks only, so the exact offset is irrelevant). GEMBA is stored
on a 0-100 scale; this does not affect the correlation.

The paper reports the bidirectional pairs among {ar,en,fr,hi,ru} at the
sentence level and {en,fr,nl} at the document level; the raw submissions
contain additional target languages. By default we report the paper grid;
pass --all-pairs to include every pair present in the data.

Usage
-----
    python task1_robustness.py                    # both metrics, both granularities
    python task1_robustness.py --metric comet     # one metric
    python task1_robustness.py --all-pairs        # do not filter to the paper grid
    python task1_robustness.py --data-dir data/task1

Requires: numpy, scipy (standard scientific Python).
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy.stats import spearmanr

CEFR = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6}

# Language sets used in the paper's heatmaps (Section 5.1, Figures 2-4).
PAPER_LANGS = {"sentence": {"ar", "en", "fr", "hi", "ru"},
               "document": {"en", "fr", "nl"}}

# Pretty system order for printing.
SYSTEM_ORDER = ["tower-instruct-7b", "google-translate", "translategemma-4b-it",
                "translategemma-12b-it", "gpt-5.4"]
ALPHA = 0.05  # significance threshold for the * marker


def load_task1(data_dir):
    """Return {(system, granularity): list_of_result_rows}."""
    out = {}
    for path in sorted(glob.glob(os.path.join(data_dir, "task1_*.json"))):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        out[(d["mt_system"], d["granularity"])] = d["results"]
    if not out:
        raise SystemExit("No task1_*.json files found in %s" % data_dir)
    return out


def pair_correlations(rows, metric, langs=None):
    """Spearman(src_cefr_ordinal, quality) for each (src,tgt) pair.

    Returns {(src,tgt): (rho, p, n)} skipping rows with missing CEFR or score.
    """
    buckets = {}
    for r in rows:
        src, tgt = r["src_lang"], r["tgt_lang"]
        if langs is not None and (src not in langs or tgt not in langs):
            continue
        if src == tgt or r.get("src_cefr") not in CEFR:
            continue
        score = (r.get("score") or {}).get(metric)
        if score is None:
            continue
        buckets.setdefault((src, tgt), []).append((CEFR[r["src_cefr"]], float(score)))
    res = {}
    for pair, vals in buckets.items():
        if len(vals) < 3:
            continue
        x = [a for a, _ in vals]
        y = [b for _, b in vals]
        rs = spearmanr(x, y)
        res[pair] = (float(rs.statistic), float(rs.pvalue), len(vals))
    return res


def summarize(res):
    """Aggregate stats over a dict of {pair: (rho,p,n)}."""
    if not res:
        return None
    rhos = np.array([v[0] for v in res.values()])
    sig_neg = sum(1 for rho, p, _ in res.values() if rho < 0 and p < ALPHA)
    sig_pos = sum(1 for rho, p, _ in res.values() if rho > 0 and p < ALPHA)
    return {"n_pairs": len(res), "mean_rho": float(rhos.mean()),
            "min_rho": float(rhos.min()), "max_rho": float(rhos.max()),
            "sig_neg": sig_neg, "sig_pos": sig_pos}


def print_block(system, gran, metric, res):
    print("  %-22s %-8s metric=%-5s" % (system, gran, metric))
    for pair in sorted(res):
        rho, p, n = res[pair]
        star = "*" if p < ALPHA else " "
        print("      %s->%s  rho=%+0.3f%s  (p=%.3f, n=%d)"
              % (pair[0], pair[1], rho, star, p, n))
    s = summarize(res)
    if s:
        print("      -> %d pairs | mean rho=%+0.3f | range [%+0.3f, %+0.3f] | "
              "sig neg=%d, sig pos=%d"
              % (s["n_pairs"], s["mean_rho"], s["min_rho"], s["max_rho"],
                 s["sig_neg"], s["sig_pos"]))
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "data", "task1"))
    ap.add_argument("--metric", choices=["comet", "gemba", "both"], default="both")
    ap.add_argument("--all-pairs", action="store_true",
                    help="include language pairs outside the paper grid")
    args = ap.parse_args()

    data = load_task1(args.data_dir)
    metrics = ["comet", "gemba"] if args.metric == "both" else [args.metric]

    print("=" * 70)
    print("ComplexityMT  Task 1 (Robustness): source CEFR level vs MT quality")
    print("Spearman correlation per language pair. '*' = p < %.2f" % ALPHA)
    print("Paper grid: sentence={ar,en,fr,hi,ru}, document={en,fr,nl} "
          "(use --all-pairs to disable)")
    print("=" * 70)

    systems = [s for s in SYSTEM_ORDER if any(k[0] == s for k in data)]
    systems += [s for s in sorted({k[0] for k in data}) if s not in systems]

    for gran in ("sentence", "document"):
        langs = None if args.all_pairs else PAPER_LANGS[gran]
        for system in systems:
            rows = data.get((system, gran))
            if rows is None:
                continue
            for metric in metrics:
                res = pair_correlations(rows, metric, langs)
                if res:
                    print_block(system, gran, metric, res)

    print("Interpretation: negative rho => higher-CEFR source texts get lower")
    print("translation quality (the Robustness finding, Section 5.1).")


if __name__ == "__main__":
    main()
