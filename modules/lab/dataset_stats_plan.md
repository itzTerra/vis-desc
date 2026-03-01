# Dataset Statistics Display Plan

## Table 1 — Dataset Overview

Grouped by dataset family, Large subdatasets indented under the Overall row.

```
| Dataset            | Size | Tokens | Sentences | Vocab (unique tokens) |
| ------------------ | ---- | ------ | --------- | --------------------- |
| Small (unbalanced) | 753  | 44,079 | 2,303     | 5,737                 |
| Small (balanced)   | 753  | ...    | ...       | ...                   |
─── Large ─────────────────────────────────────────────────────────────────────────
Overall              |  100,000 |  4,344,439 |     222,025 |                93,077
  artif_5            |   15,000 |  1,210,568 |      68,745 |                14,712
  artif_4            |   15,000 |    405,539 |      15,025 |                10,676
  flickr30k          |   15,000 |    264,617 |      15,011 |                 6,884
  coco               |    5,000 |     73,023 |       5,013 |                 3,144
  sbu                |    5,000 |     91,058 |       6,751 |                11,893
  movie_summaries    |    5,000 |    206,881 |       9,164 |                18,312
  book_summaries     |    5,000 |    220,259 |       8,675 |                21,369
  book_dialogs       |    5,000 |    342,173 |      18,436 |                12,231
  wiki               |   10,000 |    565,028 |      22,360 |                34,992
  news               |    5,000 |    223,127 |       9,634 |                17,397
  hotels             |    2,000 |    104,443 |       2,221 |                 9,234
  yelp               |    3,000 |    178,874 |      13,189 |                10,532
  arxiv              |    5,000 |    211,730 |       8,694 |                11,493
  amazon_reviews     |    5,000 |    247,119 |      19,107 |                10,984
```

## Table 2 — Per-document statistics

Multi-level column header via `pd.MultiIndex`. Each metric uses 4 columns; `Mean ± 2σ` is a single cell.

```
                   ┌── Char Length ──────────────┐┌── Token Count ──────────────┐┌── Sent Count ───────────────┐
Dataset            │ Min   Max    Med   Mean±2σ   ││ Min   Max   Med   Mean±2σ   ││ Min  Max   Med   Mean±2σ    │
───────────────────────────────────────────────────────────────────────────────────────────────────────────────
Small (unbalanced) │  12  2,234   215   265±209   ││   2   ...   ...    59±45    ││   1   ...   ...   3.1±3.8   │
Small (balanced)   │ ...   ...    ...     ...     ││ ...   ...   ...     ...     ││ ...   ...   ...    ...      │
Overall (large)    │ ...   ...    ...     ...     ││ ...   ...   ...     ...     ││ ...   ...   ...    ...      │
  artif_5          │ ...   ...    ...     ...     ││ ...   ...   ...     ...     ││ ...   ...   ...    ...      │
  ...
```

**Implementation:**

```python
metrics = ["Char Length", "Token Count", "Sent Count"]
stats   = ["Min", "Max", "Median", "Mean ± 2σ"]
columns = pd.MultiIndex.from_product([metrics, stats])
```

Render with `df.style` + `display(df)`.

### What needs to be added

- **Median** is not yet computed in the per-dataset loop — add `np.median(...)` alongside existing mean/std calls.
- **Min/Max** are only computed in the visualisation cells, not in the stats loop — add `min(...)` / `max(...)` there too.

### LaTeX export

**Table 1** — use `df_to_latex()` from `evaluation/nli/helpers.py`. The Large subdataset rows can be indented by prepending `\quad` to their Dataset cell values before passing to the function.

**Table 2** — use `df_to_latex_multirow_header()` from `evaluation/nli/helpers.py`. It already handles `\multirow`/`\multicolumn` headers for grouped column structures, which maps directly onto the `[metric, stat]` MultiIndex. Pass `column_decimals` to control precision per stat column (e.g. 0 for Min/Max/Median, 1 for mean/std).

Supporting utilities already available in `evaluation/core.py`: `latex_escape()` for any string cells, `format_number()` for consistent numeric formatting.

## Histograms — token count only

17 histograms total (2 small + Large overall + 14 subdatasets). Each panel shows the token count distribution with mean and ±2σ lines.

| Figure | Layout               | Panels                                               |
| ------ | -------------------- | ---------------------------------------------------- |
| 1      | 2×1                  | Small (unbalanced), Small (balanced)                 |
| 2      | 2×2 (top row merged) | Large overall *(spans full width)*, artif_5, artif_4 |
| 3      | 2×2                  | flickr30k, coco, sbu, movie_summaries                |
| 4      | 2×2                  | book_summaries, book_dialogs, wiki, news             |
| 5      | 2×2                  | hotels, yelp, arxiv, amazon_reviews                  |

Figure 2 uses `subplot_mosaic([["overall", "overall"], ["artif_5", "artif_4"]])` to span Large overall across the top row.
