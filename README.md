# ComplexityMT

**Benchmarking the Interaction Between Text Complexity and Machine Translation**

[![arXiv](https://img.shields.io/badge/arXiv-2606.05421-b31b1b.svg)](https://arxiv.org/abs/2606.05421)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)

Official code and data release for the paper *"Does Machine Translation Preserve Text Complexity?"* (Imperial et al., 2026). **ComplexityMT** is a benchmark for studying how text complexity — operationalised through the **Common European Framework of Reference (CEFR)** — interacts with machine translation across six languages and five MT systems.

The benchmark is organised around two complementary tasks:

| Task | Question | Section in paper |
|------|----------|------------------|
| **Task 1 — Robustness**   | Does source-text complexity affect translation quality?           | §5.1 / Figs. 2–4 |
| **Task 2 — Preservation** | Does machine translation preserve the CEFR level of the source?   | §5.2–5.3 / Figs. 5–6 |

## Key findings

- **Robustness.** Higher-CEFR source texts receive systematically lower-quality translations. The negative correlation between source CEFR level and reference-free MT-quality scores (COMET, GEMBA-DA) is significant for the majority of language pairs and consistent across all five MT systems.
- **Preservation.** At **document level**, back-translations are read at a *lower* CEFR level than the forward translation across every system evaluated (95 % CIs exclude zero, mean shift ≈ −0.16 to −0.31). At **sentence level**, the shift is near zero on aggregate. A per-pair view exposes a strong Russian asymmetry consistent with a per-language classifier offset (discussed in §5.3).

## Languages and systems

**Languages.** Arabic (ar), Dutch (nl), English (en), French (fr), Hindi (hi), Russian (ru). The paper analyses bidirectional pairs among `{ar, en, fr, hi, ru}` at the **sentence level** and `{en, fr, nl}` at the **document level**; the raw submissions in this repository contain some additional target languages that the scripts filter out by default (pass `--all-pairs` to keep them).

**MT systems evaluated:**

| System              | Type                 | Granularities       |
|---------------------|----------------------|---------------------|
| GPT-5.4             | Closed-weight LLM    | sentence, document  |
| Google Translate    | Commercial system    | sentence, document  |
| TranslateGemma 4B   | Open-weight LLM      | sentence, document  |
| TranslateGemma 12B  | Open-weight LLM      | sentence, document  |
| Tower-Instruct 7B   | Open-weight MT LLM   | sentence, document  |

## Repository layout

```
.
├── README.md
├── requirements.txt
├── task1_robustness.py        # Task 1 — reproduces §5.1 / Figs. 2–4
├── task2_preservation.py      # Task 2 — reproduces §5.2–5.3 / Figs. 5–6
└── data/
    ├── task1/                 # 5 systems × {sentence, document} = 10 JSON files
    └── task2/                 # 5 systems × {sentence, document} = 10 JSON files
```

## Quick start

This repository provides the **released JSON submissions** for both tasks along with two analysis scripts that reproduce the headline numbers from them. The scripts are lightweight and need only `numpy` and `scipy`.

```bash
git clone https://github.com/UniversalCEFR/ComplexityMT.git
cd ComplexityMT
pip install -r requirements.txt   # numpy, scipy

python task1_robustness.py        # Task 1 — both metrics, both granularities
python task2_preservation.py      # Task 2 — per-system summaries
```

Requires Python 3.9+. `task1_robustness.py` uses `numpy` and `scipy`; `task2_preservation.py` uses only `numpy`.

## Task 1 — Robustness

For each MT system, granularity, and source→target language pair, the script computes the **Spearman correlation** between the source text's gold CEFR level and a reference-free translation-quality score (COMET or GEMBA-DA). A significant negative correlation indicates that higher-CEFR source texts receive lower-quality translations.

```bash
python task1_robustness.py                      # default: both metrics, both granularities
python task1_robustness.py --metric comet       # COMET only
python task1_robustness.py --metric gemba       # GEMBA only
python task1_robustness.py --all-pairs          # include language pairs outside the paper grid
```

By default the script restricts to the bidirectional pairs reported in Figures 2–4 (sentence: `{ar, en, fr, hi, ru}`; document: `{en, fr, nl}`). Per-pair `rho` values match the paper's heatmaps (e.g., GPT-5.4 en→fr sentence COMET ≈ −0.17; Google Translate ar→en sentence COMET ≈ −0.39). Significant correlations are marked with `*` (p < 0.05).

## Task 2 — Preservation

The script measures the **model-anchored CEFR level shift** over a back-translation round trip, scoring both legs with the same CEFR classifier (`UniversalCEFR/xlm-roberta-base-cefr-all-classifier`):

```
x  (L_src, gold ℓ_x)  ─MT→  y_fwd  (L_piv)  ─MT→  y_back  (L_src)

  ℓ̂_fwd  = f(y_fwd)                       (stored as "tgt_cefr")
  ℓ̂_back = f(y_back)                      (stored as "back_cefr")

  d_model = ℓ̂_back − ℓ̂_fwd                (§5.2, primary metric)
  d_gold  = ℓ̂_back − ℓ_x                   (§5.3, anchor analysis)
```

A negative mean shift means the back-translation is read at a lower CEFR level than its anchor.

```bash
python task2_preservation.py                    # per-system summary, both granularities
python task2_preservation.py --per-pair         # also break down by source→pivot pair
                                                # (reveals the Russian asymmetry)
```

The script prints means with 95 % normal-approximation confidence intervals and a flag indicating whether each CI excludes zero, alongside the gold-anchored mean for comparison.

## Data format

CEFR levels use the ordinal scale `A1=1, A2=2, B1=3, B2=4, C1=5, C2=6`. Shifts are computed as differences, so the offset cancels.

### Task 1 — `data/task1/task1_<system>_<granularity>.json`

```json
{
  "mt_system": "gpt-5.4",
  "granularity": "sentence",
  "results": [
    {
      "src_lang": "en", "src_cefr": "C1",
      "tgt_lang": "fr",
      "src_text": "...",
      "tgt_text": "...",
      "score": {"comet": 0.835, "gemba": 97}
    }
  ]
}
```

### Task 2 — `data/task2/task2_<system>_<granularity>.json`

```json
{
  "mt_system": "gpt-5.4",
  "granularity": "document",
  "cefr_classifier": "UniversalCEFR/xlm-roberta-base-cefr-all-classifier",
  "results": [
    {
      "src_lang": "en", "src_cefr": "C1", "src_pred_cefr": "C1",
      "tgt_lang": "fr", "tgt_cefr": "B2",
      "back_cefr": "B1",
      "src_text": "...", "tgt_text": "...", "back_text": "..."
    }
  ]
}
```

- `tgt_cefr`       — classifier reading of the forward translation, `f(y_fwd)`
- `back_cefr`      — classifier reading of the back-translation,    `f(y_back)`
- `src_pred_cefr`  — classifier reading of the source, not used in the shift
- `*_text` fields  — the actual model outputs, included so others can inspect translations and back-translations

## Notes and caveats

- **GEMBA scale.** GEMBA scores are stored on the 0–100 scale (not 0–1). Spearman correlation is rank-based, so this does not affect the Task 1 results.
- **Classifier robustness.** The paper's three-classifier robustness check (Table 2) compares XLM-R, ModernBERT, and a feature-based Random Forest on a 25 % sample. Only the XLM-R outputs are required to verify the two core results and are released here; the additional classifier outputs are omitted for size.
- **Per-language offsets.** The CEFR classifier is applied to texts in two different languages on the two legs of the round trip, so per-language reading differences in the classifier are not cancelled in `d_model`. This is addressed by cross-classifier replication in the paper.

## Citation

If you use ComplexityMT in your research, please cite:

```bibtex
@article{imperial2026complexitymt,
  title         = {ComplexityMT: Benchmarking the Interaction Between Text
                   Complexity and Machine Translation},
  author        = {Imperial, Joseph Marvin and Liang, Junhong and Shoer, Belal and
                   Barayan, Abdullah and Wilkens, Rodrigo and Mussa, Omar and
                   Knight, Dawn and Ribeiro, Eug{\'e}nio and Kochmar, Ekaterina and
                   Vajjala, Sowmya and Alva-Manchego, Fernando and
                   Tayyar Madabushi, Harish},
  year          = {2026},
  eprint        = {2606.05421},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2606.05421}
}
```

## License

Code and data in this repository are released under the
[**Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)**](https://creativecommons.org/licenses/by-nc-sa/4.0/).

In short: you may share and adapt the material for non-commercial purposes,
provided you give appropriate credit and distribute your contributions under
the same license. See the [LICENSE](LICENSE) file for the full legal text.

## Contact

For questions about the benchmark or this release, please open an issue on this repository.
