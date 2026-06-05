#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ComplexityMT -- Task 2 (Preservation)
=====================================
Reproduces the Preservation results in Section 5.2 (and the anchor analysis in
Section 5.3) of the paper "Does Machine Translation Preserve Text Complexity?".

Task 2 asks RQ2: does machine translation preserve the CEFR level of the source
text? We measure this with a back-translation round trip and the same CEFR
classifier on both legs:

    x (L_src, gold ell_x)  --MT-->  y_fwd (L_piv)  --MT-->  y_back (L_src)

    ell_hat_fwd  = f(y_fwd)    (stored as "tgt_cefr")
    ell_hat_back = f(y_back)   (stored as "back_cefr")

    model-anchored shift :  d_model = ell_hat_back - ell_hat_fwd
    gold-anchored  shift :  d_gold  = ell_hat_back - ell_x

The model-anchored shift is the primary metric (Section 5.2). The gold-anchored
shift is reported for the anchor analysis (Section 5.3); the two differ by
exactly the forward gap  (ell_hat_fwd - ell_x).

A negative mean shift means the back-translation is read at a LOWER CEFR level
than its anchor. The paper's headline result: the document-level model-anchored
shift is consistently downward across systems (95% CI excludes zero), while the
sentence-level shift is near zero on aggregate.

Input data (bundled, no network needed)
----------------------------------------
data/task2/task2_*_<system>_<granularity>.json, each of the form:
    {
      "contributor": ..., "mt_system": ..., "granularity": ...,
      "cefr_classifier": "UniversalCEFR/xlm-roberta-base-cefr-all-classifier",
      "results": [
        {"src_lang": "en", "src_cefr": "C1",      # gold
         "src_pred_cefr": "C1",                    # f(x), not used in the shift
         "tgt_lang": "fr", "tgt_cefr": "B2",       # f(y_fwd) = ell_hat_fwd
         "back_cefr": "B1",                        # f(y_back) = ell_hat_back
         "src_text": ..., "tgt_text": ..., "back_text": ...},
        ...
      ]
    }

CEFR levels map to A1=1, A2=2, B1=3, B2=4, C1=5, C2=6. Shifts are differences,
so the offset cancels.

Usage
-----
    python task2_preservation.py                 # per-system summary, both granularities
    python task2_preservation.py --per-pair      # also show per source->pivot pair
                                                 # (reveals the Russian asymmetry)
    python task2_preservation.py --data-dir data/task2

Requires: numpy (standard scientific Python).
"""
import argparse
import glob
import json
import os

import numpy as np

CEFR = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6}

SYSTEM_ORDER = ["tower-7b", "google-translate", "translategemma-4b",
                "translategemma-12b", "gpt-5.4"]


def load_task2(data_dir):
    """Return {(system, granularity): list_of_result_rows}."""
    out = {}
    for path in sorted(glob.glob(os.path.join(data_dir, "task2_*.json"))):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        out[(d["mt_system"], d["granularity"])] = d["results"]
    if not out:
        raise SystemExit("No task2_*.json files found in %s" % data_dir)
    return out


def shifts(rows):
    """Return arrays (d_model, d_gold) and a per-pair dict over valid rows.

    d_model = back_cefr - tgt_cefr ;  d_gold = back_cefr - src_cefr(gold)
    per_pair maps (src_lang, pivot_lang) -> list of d_model values.
    """
    d_model, d_gold, per_pair = [], [], {}
    for r in rows:
        fwd, back, gold = r.get("tgt_cefr"), r.get("back_cefr"), r.get("src_cefr")
        if fwd not in CEFR or back not in CEFR:
            continue
        dm = CEFR[back] - CEFR[fwd]
        d_model.append(dm)
        if gold in CEFR:
            d_gold.append(CEFR[back] - CEFR[gold])
        per_pair.setdefault((r["src_lang"], r["tgt_lang"]), []).append(dm)
    return np.array(d_model, float), np.array(d_gold, float), per_pair


def mean_ci(a):
    """Mean and 95% normal-approximation CI half-width; (mean, lo, hi, n)."""
    n = len(a)
    if n == 0:
        return (float("nan"),) * 3 + (0,)
    m = a.mean()
    half = 1.96 * a.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    return m, m - half, m + half, n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "data", "task2"))
    ap.add_argument("--per-pair", action="store_true",
                    help="print per source->pivot pair means (shows Russian asymmetry)")
    args = ap.parse_args()

    data = load_task2(args.data_dir)
    systems = [s for s in SYSTEM_ORDER if any(k[0] == s for k in data)]
    systems += [s for s in sorted({k[0] for k in data}) if s not in systems]

    print("=" * 74)
    print("ComplexityMT  Task 2 (Preservation): model-anchored CEFR level shift")
    print("d_model = f(y_back) - f(y_fwd).  Negative => back-translation read lower.")
    print("95% CI by normal approximation; 'excl 0' = CI excludes zero.")
    print("=" * 74)

    for gran in ("sentence", "document"):
        print("\n--- %s level ---" % gran.upper())
        print("  %-20s %8s %8s %8s   %s" %
              ("system", "d_model", "95%CI_lo", "95%CI_hi", "(n) [d_gold]"))
        pooled = []
        for system in systems:
            rows = data.get((system, gran))
            if rows is None:
                continue
            dm, dg, per_pair = shifts(rows)
            m, lo, hi, n = mean_ci(dm)
            gold_m = dg.mean() if len(dg) else float("nan")
            excl = "excl 0" if (lo > 0 or hi < 0) else "incl 0"
            print("  %-20s %+8.3f %+8.3f %+8.3f   (n=%d) [%+.3f] %s" %
                  (system, m, lo, hi, n, gold_m, excl))
            pooled.append(m)
            if args.per_pair:
                for pair in sorted(per_pair, key=lambda p: np.mean(per_pair[p])):
                    vals = np.array(per_pair[pair], float)
                    print("        %s->%s  mean d_model=%+0.3f (n=%d)" %
                          (pair[0], pair[1], vals.mean(), len(vals)))
        if pooled:
            print("  %-20s %+8.3f   (mean over systems)" %
                  ("ALL", float(np.mean(pooled))))

    print("\nInterpretation (Section 5.2): the document-level model-anchored shift")
    print("is consistently negative across systems (CI excludes zero), while the")
    print("sentence-level shift is near zero. Per-pair (--per-pair) shows the")
    print("Russian asymmetry discussed as a per-language classifier offset.")


if __name__ == "__main__":
    main()
