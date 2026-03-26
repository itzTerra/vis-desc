# Neighbor Relaxation Strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `class_mode="neighbor"` relaxation strategy that keeps all 6 classes but counts predictions within ±1 as correct, alongside the existing `"relaxed"` (3-class grouping) strategy.

**Architecture:** A new `_neighbor_correct_cm` function folds ±1 off-diagonal entries into the diagonal of the 6×6 confusion matrix. A corresponding `collapse_dataset_metrics_neighbor` function derives `DatasetMetrics` from this corrected matrix. All existing visualization functions gain `"neighbor"` as a new `class_mode` value, routing to the new function.

**Tech Stack:** Python, NumPy, the existing `core.py` + `encoder/interface.py` evaluation pipeline.

---

## File Map

| File | Change |
|------|--------|
| `modules/lab/src/evaluation/core.py` | Add `_neighbor_correct_cm`, `collapse_dataset_metrics_neighbor`; update `get_confusion_matrix`, `plot_confusion_matrix`, `vis_all_models_plots`, `vis_all_models_tables` |
| `modules/lab/src/evaluation/encoder/interface.py` | Update `format_encoder_metrics_latex` docstring |

---

### Task 1: Add core neighbor-relaxation helpers

**Files:**
- Modify: `modules/lab/src/evaluation/core.py:70-122`

- [ ] **Step 1: Write the failing tests**

Create a temporary test cell or file to verify the new functions before wiring them in.

```python
import numpy as np
from evaluation.core import _neighbor_correct_cm, collapse_dataset_metrics_neighbor, DatasetMetrics

# --- _neighbor_correct_cm ---

def test_neighbor_correct_cm_identity():
    """Perfect predictions → unchanged diagonal."""
    cm = np.eye(3, dtype=int) * 10
    result = _neighbor_correct_cm(cm)
    expected = np.eye(3, dtype=int) * 10
    assert np.array_equal(result, expected), f"Expected {expected}, got {result}"

def test_neighbor_correct_cm_off_by_one_folded():
    """±1 off-diagonal entries moved to diagonal."""
    cm = np.zeros((4, 4), dtype=int)
    cm[0, 0] = 5   # correct
    cm[0, 1] = 3   # off-by-1 → should go to diagonal[0]
    cm[1, 0] = 2   # off-by-1 → should go to diagonal[1]
    cm[2, 3] = 4   # off-by-1 → should go to diagonal[2]
    result = _neighbor_correct_cm(cm)
    assert result[0, 0] == 8,  f"row 0 diagonal: expected 8, got {result[0,0]}"
    assert result[0, 1] == 0,  f"row 0, col 1: expected 0, got {result[0,1]}"
    assert result[1, 1] == 2,  f"row 1 diagonal: expected 2, got {result[1,1]}"
    assert result[1, 0] == 0,  f"row 1, col 0: expected 0, got {result[1,0]}"
    assert result[2, 2] == 4,  f"row 2 diagonal: expected 4, got {result[2,2]}"
    assert result[2, 3] == 0,  f"row 2, col 3: expected 0, got {result[2,3]}"

def test_neighbor_correct_cm_far_wrong_stays():
    """Predictions off by ≥2 are not moved."""
    cm = np.zeros((4, 4), dtype=int)
    cm[0, 3] = 7   # off-by-3, must stay
    result = _neighbor_correct_cm(cm)
    assert result[0, 3] == 7, f"far wrong: expected 7, got {result[0,3]}"
    assert result[0, 0] == 0, f"diagonal: expected 0, got {result[0,0]}"

def test_neighbor_correct_cm_preserves_total():
    """Total count must be conserved."""
    rng = np.random.default_rng(42)
    cm = rng.integers(0, 20, size=(6, 6))
    result = _neighbor_correct_cm(cm)
    assert cm.sum() == result.sum(), "Total count changed"

# --- collapse_dataset_metrics_neighbor ---

def test_collapse_neighbor_accuracy_above_strict():
    """Neighbor accuracy must be ≥ strict accuracy."""
    cm = np.array([[5, 3, 0], [2, 4, 1], [0, 1, 6]], dtype=int)
    dm = DatasetMetrics(
        mse=0.0, accuracy=0.0, precision=[], recall=[], f1=[], support=[],
        confusion_matrix=cm,
    )
    relaxed = collapse_dataset_metrics_neighbor(dm)
    strict_acc = np.trace(cm) / cm.sum()
    assert relaxed.accuracy >= strict_acc, "Neighbor accuracy must be ≥ strict"

def test_collapse_neighbor_preserves_support():
    """Row support totals must match original."""
    cm = np.array([[5, 3, 0], [2, 4, 1], [0, 1, 6]], dtype=int)
    dm = DatasetMetrics(
        mse=0.0, accuracy=0.0, precision=[], recall=[], f1=[], support=[],
        confusion_matrix=cm,
    )
    relaxed = collapse_dataset_metrics_neighbor(dm)
    original_support = [int(cm[i, :].sum()) for i in range(3)]
    assert relaxed.support == original_support, (
        f"Support changed: {original_support} → {relaxed.support}"
    )
```

Run: execute cells in notebook or `python -c "..."` in the lab `src/` directory.
Expected: `NameError: name '_neighbor_correct_cm' is not defined` (functions don't exist yet).

- [ ] **Step 2: Add `_neighbor_correct_cm` after `_collapse_cm_relaxed` in `core.py`**

In `modules/lab/src/evaluation/core.py`, insert the new function after line 76 (the closing `return out` of `_collapse_cm_relaxed`):

```python
def _neighbor_correct_cm(cm: np.ndarray) -> np.ndarray:
    """Return a copy of cm where ±1 off-diagonal predictions count as correct.

    For each row i and each column j with |i - j| == 1, the count cm[i, j]
    is moved to the diagonal cm[i, i].  Predictions off by ≥ 2 are untouched.
    The total sample count is preserved.
    """
    out = cm.copy()
    n = cm.shape[0]
    for i in range(n):
        for delta in (-1, 1):
            j = i + delta
            if 0 <= j < n:
                out[i, i] += cm[i, j]
                out[i, j] = 0
    return out
```

- [ ] **Step 3: Add `collapse_dataset_metrics_neighbor` after `collapse_dataset_metrics_relaxed` in `core.py`**

Insert after line 122 (end of `collapse_dataset_metrics_relaxed`):

```python
def collapse_dataset_metrics_neighbor(dm: DatasetMetrics) -> DatasetMetrics:
    """Compute DatasetMetrics where predictions within ±1 count as correct.

    The confusion matrix shape stays 6×6; only the off-by-one entries are
    folded into the diagonal before recomputing precision/recall/F1/accuracy.
    MSE is taken from the original DatasetMetrics (distance-weighted, already
    meaningful on its own).
    """
    cm6 = _pad_cm_to_six(dm.confusion_matrix)
    cm_corrected = _neighbor_correct_cm(cm6)
    precision, recall, f1, support, accuracy, _ = _metrics_from_cm(cm_corrected)
    folds = None
    if dm.folds:
        folds = [collapse_dataset_metrics_neighbor(f) for f in dm.folds]
    return DatasetMetrics(
        mse=dm.mse,           # preserve original distance-based MSE
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        support=support,
        confusion_matrix=cm6,  # keep full 6×6 for display
        folds=folds,
    )
```

- [ ] **Step 4: Run the tests and verify they pass**

Run the test functions defined in Step 1. All six assertions should pass with no errors.

- [ ] **Step 5: Commit**

```bash
git add modules/lab/src/evaluation/core.py
git commit -m "feat(lab): add neighbor ±1 relaxation helpers in core.py"
```

---

### Task 2: Update `get_confusion_matrix` to support `"neighbor"` mode

**Files:**
- Modify: `modules/lab/src/evaluation/core.py:149-180`

- [ ] **Step 1: Write the failing test**

```python
from evaluation.core import get_confusion_matrix, AggregatedModelData, DatasetMetrics
import numpy as np

cm6 = np.eye(6, dtype=int) * 5
cm6[0, 1] = 3   # off-by-1
dm = DatasetMetrics(mse=0.0, accuracy=0.0, precision=[], recall=[], f1=[], support=[], confusion_matrix=cm6)
amd = AggregatedModelData(model="test", test=dm)

result = get_confusion_matrix(amd, dataset="test", class_mode="neighbor")
# Should return the ORIGINAL 6×6 cm (not corrected), because the visualization
# shows where predictions actually landed.
assert result.shape == (6, 6), f"Expected (6,6), got {result.shape}"
assert np.array_equal(result, cm6), "Neighbor mode must return original 6×6 cm"
```

Expected failure: `AssertionError` because `"neighbor"` falls through to the default `return cm` — actually it will pass already since the default is return cm.
Verify by running the test to confirm it passes as-is (no code change needed for `get_confusion_matrix`).

> Note: `"neighbor"` mode intentionally returns the raw 6×6 matrix (same as `"full"`) because the confusion matrix visualization should reflect actual prediction placement. Metrics are relaxed separately in `collapse_dataset_metrics_neighbor`.

- [ ] **Step 2: Verify no change needed — add a code comment instead**

In `modules/lab/src/evaluation/core.py` at `get_confusion_matrix` (around line 152), update the docstring to document `"neighbor"`:

```python
def get_confusion_matrix(
    metrics_dict: AggregatedModelData, dataset: str = "test", class_mode: str = "full"
) -> np.ndarray | None:
    """Extract confusion matrix from aggregated model data.

    class_mode:
      - "full": returns a 6x6 matrix, padding if needed
      - "relaxed": merges classes (0,1), (2,3), (4,5) into a 3x3 matrix
      - "neighbor": returns the raw 6x6 matrix (same as "full"); metrics are
        relaxed separately via collapse_dataset_metrics_neighbor
    """
```

- [ ] **Step 3: Commit**

```bash
git add modules/lab/src/evaluation/core.py
git commit -m "docs(lab): document 'neighbor' class_mode in get_confusion_matrix"
```

---

### Task 3: Update `plot_confusion_matrix` for `"neighbor"` mode

**Files:**
- Modify: `modules/lab/src/evaluation/core.py:183-268`

- [ ] **Step 1: Write the failing test**

```python
# Verify that plot_confusion_matrix doesn't crash with class_mode="neighbor"
# and uses 6-class labels (0–5), not the 3-class "0/1" labels.
import matplotlib
matplotlib.use("Agg")  # no display needed in test
import matplotlib.pyplot as plt
from evaluation.core import plot_confusion_matrix, AggregatedModelData, DatasetMetrics
import numpy as np

cm6 = np.eye(6, dtype=int) * 5
dm = DatasetMetrics(mse=0.0, accuracy=1.0, precision=[1.0]*6, recall=[1.0]*6, f1=[1.0]*6, support=[5]*6, confusion_matrix=cm6)
amd = AggregatedModelData(model="test", test=dm)

# Should not raise; must use labels ["0","1","2","3","4","5"]
try:
    plot_confusion_matrix(amd, dataset="test", show_proportional=False, show_title=False, class_mode="neighbor")
    plt.close("all")
    print("OK")
except Exception as e:
    print(f"FAIL: {e}")
```

Expected: `"OK"` — actually it already works because `"neighbor"` falls into the `else` branch that uses `[f"{i}" for i in range(6)]` labels. Verify this is true by running the test.

- [ ] **Step 2: Add explicit `"neighbor"` branch to the labels logic and colormap selection**

In `plot_confusion_matrix`, find the labels block (around lines 225–228):

```python
    if class_mode == "relaxed":
        labels = ["0/1", "2/3", "4/5"]
    else:
        labels = [f"{i}" for i in range(6)]
```

Replace with:

```python
    if class_mode == "relaxed":
        labels = ["0/1", "2/3", "4/5"]
    elif class_mode == "neighbor":
        labels = [f"{i}" for i in range(6)]
    else:
        labels = [f"{i}" for i in range(6)]
```

Also update the colormap selection (around lines 202–206):

```python
    cmap = (
        CMAP_SEQUENTIAL_PRIMARY
        if class_mode not in ("relaxed", "neighbor")
        else CMAP_SEQUENTIAL_SECONDARY
    )
```

- [ ] **Step 3: Run the test again — must still print `"OK"`**

- [ ] **Step 4: Commit**

```bash
git add modules/lab/src/evaluation/core.py
git commit -m "feat(lab): support class_mode='neighbor' in plot_confusion_matrix"
```

---

### Task 4: Update `vis_all_models_tables` to support `"neighbor"` mode

**Files:**
- Modify: `modules/lab/src/evaluation/core.py:993-1070`

- [ ] **Step 1: Write the failing test**

```python
from evaluation.core import vis_all_models_tables, AggregatedModelData, DatasetMetrics, collapse_dataset_metrics_neighbor
import numpy as np

cm6 = np.zeros((6, 6), dtype=int)
cm6[0, 1] = 5   # off-by-1 for class 0
cm6[1, 1] = 10  # correct for class 1
dm = DatasetMetrics(mse=0.5, accuracy=0.5, precision=[0.5]*6, recall=[0.5]*6, f1=[0.5]*6, support=[5]*6, confusion_matrix=cm6)
amd = AggregatedModelData(model="test_model", test=dm)

df = vis_all_models_tables([amd], metrics=["Acc"], splits=["Test"], class_mode="neighbor")
# Neighbor accuracy on this cm: cm[0,1] (off-by-1) counts as correct for row 0
# Strict accuracy: 10/15 ≈ 0.667; Neighbor accuracy: 15/15 = 1.0
neighbor = collapse_dataset_metrics_neighbor(dm)
expected_acc = neighbor.accuracy
test_acc_in_df = df[df["Model"] == "test_model"]["Test Acc"].values[0]
assert abs(test_acc_in_df - expected_acc) < 1e-6, (
    f"Expected neighbor accuracy {expected_acc:.4f}, got {test_acc_in_df:.4f}"
)
print("OK")
```

Expected failure: The table returns strict accuracy because line 1029 only checks `class_mode == "relaxed"`.

- [ ] **Step 2: Update `_get_ds` inside `vis_all_models_tables` (line 1029)**

Find this inner function (around line 1025–1029):

```python
        def _get_ds(dataset: str) -> Optional[DatasetMetrics]:
            d = getattr(m, dataset, None)
            if d is None:
                return None
            return collapse_dataset_metrics_relaxed(d) if class_mode == "relaxed" else d
```

Replace with:

```python
        def _get_ds(dataset: str) -> Optional[DatasetMetrics]:
            d = getattr(m, dataset, None)
            if d is None:
                return None
            if class_mode == "relaxed":
                return collapse_dataset_metrics_relaxed(d)
            if class_mode == "neighbor":
                return collapse_dataset_metrics_neighbor(d)
            return d
```

- [ ] **Step 3: Run the test — must print `"OK"`**

- [ ] **Step 4: Commit**

```bash
git add modules/lab/src/evaluation/core.py
git commit -m "feat(lab): support class_mode='neighbor' in vis_all_models_tables"
```

---

### Task 5: Update `vis_all_models_plots` to support `"neighbor"` mode

**Files:**
- Modify: `modules/lab/src/evaluation/core.py:423-600`

- [ ] **Step 1: Write the failing test**

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from evaluation.core import vis_all_models_plots, AggregatedModelData, DatasetMetrics
import numpy as np

cm6 = np.eye(6, dtype=int) * 5
dm = DatasetMetrics(mse=0.0, accuracy=1.0, precision=[1.0]*6, recall=[1.0]*6, f1=[1.0]*6, support=[5]*6, confusion_matrix=cm6)
amd = AggregatedModelData(model="catboost_minilm", train=dm, test=dm)

try:
    vis_all_models_plots([amd], dataset="test", class_mode="neighbor")
    plt.close("all")
    print("OK")
except Exception as e:
    print(f"FAIL: {e}")
```

Expected: Run to check whether it crashes. The `"neighbor"` mode falls through to the `else` branch in the existing code, which does not call `collapse_dataset_metrics_relaxed`, so it likely already works — but the xtick labels and metrics will not be relaxed.

- [ ] **Step 2: Update the `"models"` code path (lines ~570–601)**

In `vis_all_models_plots`, the section that processes `lg_models_data` and `models_data` has this pattern (lines ~575–601):

```python
            else:
                md = getattr(m, dataset)
                original_mse = md.mse
                md = (
                    collapse_dataset_metrics_relaxed(md)
                    if class_mode == "relaxed"
                    else md
                )
                md.mse = original_mse
                models_data.append({"model": m.model, "metrics": md})
```

Replace the relaxed-only check in BOTH the `lg_models_data` branch (lines ~575-583) and the `models_data` branch (lines ~584-601). For the `lg_models_data` branch:

```python
                md = getattr(m, dataset)
                original_mse = md.mse
                if class_mode == "relaxed":
                    md = collapse_dataset_metrics_relaxed(md)
                elif class_mode == "neighbor":
                    md = collapse_dataset_metrics_neighbor(md)
                md.mse = original_mse
                lg_models_data.append({"model": m.model, "metrics": md})
```

For the `models_data` `else` branch (non-`combined` case):

```python
                    if class_mode == "relaxed":
                        md = collapse_dataset_metrics_relaxed(md)
                    elif class_mode == "neighbor":
                        md = collapse_dataset_metrics_neighbor(md)
                    md.mse = original_mse
                    models_data.append({"model": m.model, "metrics": md})
```

- [ ] **Step 3: Update xtick labels for `"neighbor"` mode**

Find the xtick logic (around lines 549–554):

```python
            if class_mode == "relaxed":
                xticks = ["0/1", "2/3", "4/5"][:n_labels]
            elif class_mode == "combined":
                xticks = [str(i) for i in range(n_labels)]
            else:
                xticks = [f"Label {i}" for i in range(n_labels)]
```

Replace with:

```python
            if class_mode == "relaxed":
                xticks = ["0/1", "2/3", "4/5"][:n_labels]
            elif class_mode in ("combined", "neighbor"):
                xticks = [str(i) for i in range(n_labels)]
            else:
                xticks = [f"Label {i}" for i in range(n_labels)]
```

There is a second xtick block in the `train_test` mode path (lines ~549-554). Apply the same change there.

- [ ] **Step 4: Run the test again — must print `"OK"`**

- [ ] **Step 5: Commit**

```bash
git add modules/lab/src/evaluation/core.py
git commit -m "feat(lab): support class_mode='neighbor' in vis_all_models_plots"
```

---

### Task 6: Update `encoder/interface.py` docstring

**Files:**
- Modify: `modules/lab/src/evaluation/encoder/interface.py:111-141`

- [ ] **Step 1: Update the docstring of `format_encoder_metrics_latex`**

Find the `class_mode` argument documentation (around line 122):

```python
        class_mode: "full" for 6-class or "relaxed" for merged 0/1, 2/3, 4/5.
```

Replace with:

```python
        class_mode: "full" for 6-class, "relaxed" for merged 0/1, 2/3, 4/5,
            or "neighbor" for 6-class with ±1 predictions counted as correct.
```

- [ ] **Step 2: Commit**

```bash
git add modules/lab/src/evaluation/encoder/interface.py
git commit -m "docs(lab): document 'neighbor' class_mode in format_encoder_metrics_latex"
```

---

### Task 7: Smoke-test the full pipeline in the notebook

**Files:**
- Modify: `modules/lab/src/evaluation/nli/nli.ipynb` (add a temporary cell, do not commit)

- [ ] **Step 1: Add a smoke-test cell at the end of the notebook**

After the existing relaxed-mode cells, add a new cell:

```python
# Smoke test: neighbor relaxation
from evaluation.core import vis_all_models_tables, vis_all_models_plots

# Reuse existing nli_models list (already computed above in the notebook)
df_neighbor = vis_all_models_tables(
    nli_models,
    metrics=["RMSE", "Acc", "F1w"],
    splits=["Train", "Test"],
    class_mode="neighbor",
)
display(df_neighbor)

vis_all_models_plots(nli_models, dataset="test", class_mode="neighbor")
```

- [ ] **Step 2: Run the cell — verify**

Expected outputs:
- A DataFrame with 6-class metrics where the neighbor relaxation is applied (Acc column values should be ≥ the "full" Acc values).
- Bar-chart plots with x-axis labels `0 1 2 3 4 5`.
- No exceptions.

- [ ] **Step 3: Remove the smoke-test cell (do not commit notebook changes)**

The cell was for verification only.

---

## Self-Review

### Spec coverage

| Requirement | Task |
|---|---|
| ±1 counts as correct | Task 1 (`_neighbor_correct_cm`, `collapse_dataset_metrics_neighbor`) |
| Keep 6 classes (no collapse) | Task 1 (cm stays 6×6), Task 2 (get_confusion_matrix returns full 6×6) |
| Modular — easy to switch back | All functions keep `class_mode` param; `"relaxed"` path unchanged |
| Tables show relaxed metrics | Task 4 (`vis_all_models_tables`) |
| Plots show relaxed metrics | Task 5 (`vis_all_models_plots`) |
| Confusion matrix unchanged | Task 2 (returns raw 6×6 in "neighbor" mode) |
| MSE preserved from original | Task 1 (`collapse_dataset_metrics_neighbor` copies `dm.mse`) |

### Placeholder scan

No TBDs or "implement later" phrases present.

### Type consistency

- `_neighbor_correct_cm(cm: np.ndarray) -> np.ndarray` — used in `collapse_dataset_metrics_neighbor` ✓
- `collapse_dataset_metrics_neighbor(dm: DatasetMetrics) -> DatasetMetrics` — used in `vis_all_models_tables` and `vis_all_models_plots` ✓
- All call sites pass `DatasetMetrics` and expect `DatasetMetrics` back ✓
