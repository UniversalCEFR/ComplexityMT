import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── Data ──────────────────────────────────────────────────────────────────────
# Only keep bidirectional language pairs for en/fr/ru/ar/hi
# Format: (src, tgt): {model: (value, significance)}  significance: 1 = p<0.05, 0 = not significant
# Google Translate correlations updated from new data (2024-05-07)

data = {
    ("ar","en"): {"tower":(-0.29,1), "google":(-0.385,1), "gemma4b":(-0.358,1), "gemma12b":(-0.399,1), "gpt":(-0.365,1)},
    ("ar","fr"): {"tower":(-0.32,1), "google":(-0.344,1), "gemma4b":(-0.356,1), "gemma12b":(-0.364,1), "gpt":(-0.404,1)},
    ("ar","hi"): {"tower":(-0.11,0), "google":(-0.349,1), "gemma4b":(-0.339,1), "gemma12b":(-0.364,1), "gpt":(-0.377,1)},
    ("ar","ru"): {"tower":(-0.28,1), "google":(-0.341,1), "gemma4b":(-0.342,1), "gemma12b":(-0.366,1), "gpt":(-0.371,1)},
    ("en","ar"): {"tower":(-0.23,1), "google":(-0.302,1), "gemma4b":(-0.313,1), "gemma12b":(-0.296,1), "gpt":(-0.396,1)},
    ("en","fr"): {"tower":(-0.19,1), "google":(-0.194,1), "gemma4b":(-0.264,1), "gemma12b":(-0.257,1), "gpt":(-0.303,1)},
    ("en","hi"): {"tower":(-0.16,1), "google":(-0.267,1), "gemma4b":(-0.279,1), "gemma12b":(-0.330,1), "gpt":(-0.345,1)},
    ("en","ru"): {"tower":(-0.20,1), "google":(-0.182,1), "gemma4b":(-0.295,1), "gemma12b":(-0.302,1), "gpt":(-0.321,1)},
    ("fr","ar"): {"tower":( 0.24,1), "google":(-0.012,0), "gemma4b":(-0.143,1), "gemma12b":(-0.105,1), "gpt":(-0.147,1)},
    ("fr","en"): {"tower":(-0.02,0), "google":(-0.052,0), "gemma4b":(-0.129,1), "gemma12b":(-0.113,1), "gpt":(-0.165,1)},
    ("fr","hi"): {"tower":( 0.30,1), "google":(-0.313,1), "gemma4b":(-0.240,1), "gemma12b":(-0.196,1), "gpt":(-0.197,1)},
    ("fr","ru"): {"tower":( 0.02,0), "google":( 0.026,0), "gemma4b":(-0.086,0), "gemma12b":(-0.127,1), "gpt":(-0.114,1)},
    ("hi","ar"): {"tower":( 0.13,0), "google":(-0.132,0), "gemma4b":(-0.058,0), "gemma12b":(-0.038,0), "gpt":(-0.079,0)},
    ("hi","en"): {"tower":(-0.08,0), "google":(-0.098,0), "gemma4b":(-0.114,0), "gemma12b":(-0.138,0), "gpt":(-0.048,0)},
    ("hi","fr"): {"tower":(-0.09,0), "google":(-0.098,0), "gemma4b":(-0.077,0), "gemma12b":(-0.036,0), "gpt":(-0.063,0)},
    ("hi","ru"): {"tower":(-0.06,0), "google":(-0.142,1), "gemma4b":(-0.108,0), "gemma12b":(-0.051,0), "gpt":(-0.074,0)},
    ("ru","ar"): {"tower":( 0.35,1), "google":( 0.157,1), "gemma4b":( 0.137,0), "gemma12b":( 0.130,0), "gpt":( 0.174,1)},
    ("ru","en"): {"tower":( 0.03,0), "google":( 0.048,0), "gemma4b":( 0.092,0), "gemma12b":( 0.088,0), "gpt":( 0.065,0)},
    ("ru","fr"): {"tower":( 0.11,0), "google":( 0.132,0), "gemma4b":( 0.171,1), "gemma12b":( 0.157,1), "gpt":( 0.142,1)},
    ("ru","hi"): {"tower":( 0.28,1), "google":(-0.164,1), "gemma4b":(-0.025,0), "gemma12b":(-0.058,0), "gpt":(-0.054,0)},
}

models = ["tower", "google", "gemma4b", "gemma12b", "gpt"]
model_labels = ["Tower\nInstruct", "Google\nTranslate", "Gemma\n4B", "Gemma\n12B", "GPT-5.4"]
pairs = list(data.keys())
pair_labels = [f"{s}→{t}" for s, t in pairs]

# ── Build Matrix ──────────────────────────────────────────────────────────────────
matrix = np.array([[data[p][m][0] for m in models] for p in pairs])
significance = np.array([[data[p][m][1] for m in models] for p in pairs])

# ── Plotting (Horizontal Version) ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 3))

vmax = 0.45
norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
im = ax.imshow(matrix.T, aspect="auto", cmap="RdBu", norm=norm)  # Transpose for horizontal

# Axis labels - translation pairs on x-axis, models on y-axis
ax.set_xticks(range(len(pairs)))
ax.set_xticklabels(pair_labels, fontsize=9, rotation=45, ha="right")
ax.set_yticks(range(len(models)))
ax.set_yticklabels(model_labels, fontsize=10)

# Value annotations (with significance markers)
for i in range(len(models)):
    for j in range(len(pairs)):
        val = matrix[j, i]
        sig = significance[j, i]
        color = "white" if abs(val) > 0.28 else "black"
        # Add * to indicate significance p<0.05
        text = f"{val:.2f}*" if sig == 1 else f"{val:.2f}"
        ax.text(j, i, text, ha="center", va="center", fontsize=7.5, color=color)

# Group separator lines (4 columns per group, by source language)
group_sizes = [4, 4, 4, 4, 4]
cumsum = 0
for g in group_sizes[:-1]:
    cumsum += g
    ax.axvline(cumsum - 0.5, color="gray", linewidth=1.2, linestyle="--")

# Source language labels (top)
src_langs = ["ar", "en", "fr", "hi", "ru"]
positions = [1.5, 5.5, 9.5, 13.5, 17.5]
for lang, pos in zip(src_langs, positions):
    ax.text(pos, -0.25, lang, ha="center", va="center", fontsize=10,
            fontweight="bold", transform=ax.get_xaxis_transform())

cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("CEFR–COMET Correlation", fontsize=9)

plt.tight_layout()
plt.savefig("cefr_comet_heatmap_horizontal.png", dpi=150, bbox_inches="tight")
plt.show()
