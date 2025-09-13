from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence
import argparse
import math
import csv

import label_studio_models as lsm


Category = int  # 0..5 inclusive


@dataclass(slots=True)
class TaskRatings:
    task_id: int
    ratings: dict[int, Category]  # annotator_id -> rating
    lead_times: dict[int, float]  # annotator_id -> lead_time (seconds)


def _filter_outliers_iqr(values: List[float]) -> List[float]:
    """Return values with outliers (per 1.5*IQR rule) removed.

    If fewer than 4 points, returns values unchanged.
    """
    if len(values) < 4:
        return values
    xs = sorted(values)
    n = len(xs)

    def _percentile(p: float) -> float:
        if n == 1:
            return xs[0]
        idx = (n - 1) * p
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        return xs[lo] * (1 - frac) + xs[hi] * frac

    q1 = _percentile(0.25)
    q3 = _percentile(0.75)
    iqr = q3 - q1
    if iqr <= 0:
        return values  # all identical
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return [v for v in values if lower <= v <= upper]


def _extract_rating_from_annotation(
    ann: lsm.Annotation, label_name: str = "rating"
) -> Category | None:
    """Return the integer rating (0..5) for a single annotation or None."""
    choices = ann.choices_by_from_name().get(label_name)
    if not choices:
        return None
    # take first, ignore extras
    choice = choices[0]
    try:
        val = int(choice)
    except Exception:
        return None
    return val


def collect_task_ratings(
    tasks: Sequence[lsm.Task], label_name: str = "rating"
) -> List[TaskRatings]:
    """Build per-task rating maps. Only tasks with >=1 rating annotation are included."""
    out: List[TaskRatings] = []
    for t in tasks:
        rating_map: Dict[int, Category] = {}
        lead_time_map: Dict[int, float] = {}
        for ann in t.annotations:
            if ann.completed_by is None:
                continue
            r = _extract_rating_from_annotation(ann, label_name)
            if r is None:
                continue
            rating_map[ann.completed_by] = r
            if ann.lead_time is not None:
                lead_time_map[ann.completed_by] = float(ann.lead_time)
        if rating_map:
            out.append(
                TaskRatings(
                    task_id=t.id,
                    ratings=rating_map,
                    lead_times=lead_time_map,
                )
            )
    return out


def quadratic_weights(k: int) -> List[List[float]]:
    """Pre-compute quadratic weights matrix w_ij for categories 0..k-1."""
    denom = (k - 1) ** 2 if k > 1 else 1
    w: List[List[float]] = []
    for i in range(k):
        row = []
        for j in range(k):
            row.append(1.0 - ((i - j) ** 2) / denom)
        w.append(row)
    return w


def weighted_cohen_kappa(
    rater_a: List[Category],
    rater_b: List[Category],
    k: int = 6,
    return_details: bool = False,
) -> float | tuple[float, dict]:
    """Compute quadratic weighted Cohen's kappa.

    Weight definition (quadratic):
    w_ij = 1 - ( (i - j)^2 / (k - 1)^2 )   where k = number of categories

    Parameters
    ----------
    rater_a, rater_b: parallel lists of category indices (0..k-1).
    k: number of categories (default 6 for ratings 0-5).
    return_details: if True, also return a dict with confusion matrix and stats.
    """
    if len(rater_a) != len(rater_b):
        raise ValueError("Rater label lists must be the same length")
    n = len(rater_a)
    if n == 0:
        return float("nan") if not return_details else (float("nan"), {})

    # Confusion matrix counts n_ij (rows=rater_a, cols=rater_b)
    counts = [[0 for _ in range(k)] for _ in range(k)]
    for a, b in zip(rater_a, rater_b):
        if not (0 <= a < k and 0 <= b < k):
            raise ValueError(f"Rating out of bounds (a={a}, b={b}, k={k})")
        counts[a][b] += 1

    weights = quadratic_weights(k)

    # Row and column marginals
    row_marg = [sum(row) for row in counts]
    col_marg = [sum(counts[i][j] for i in range(k)) for j in range(k)]

    # Observed weighted agreement
    obs = 0.0
    for i in range(k):
        for j in range(k):
            obs += weights[i][j] * counts[i][j]
    obs /= n

    # Expected weighted agreement
    exp = 0.0
    for i in range(k):
        for j in range(k):
            exp += weights[i][j] * (row_marg[i] * col_marg[j] / n)
    exp /= n

    if math.isclose(1 - exp, 0.0):
        kappa = float("nan")
    else:
        kappa = (obs - exp) / (1 - exp)

    if not return_details:
        return kappa

    # Per-category F1 treating rater A as predictions, rater B as reference.
    per_category_f1: List[float] = []
    for c in range(k):
        tp = counts[c][c]
        fp = row_marg[c] - tp  # predicted c but ref not c
        fn = col_marg[c] - tp  # ref c but predicted not c
        if tp == 0 and (fp > 0 or fn > 0):
            per_category_f1.append(0.0)
            continue
        denom = 2 * tp + fp + fn
        if denom == 0:
            per_category_f1.append(float("nan"))
        else:
            per_category_f1.append(2 * tp / denom)
    details = {
        "confusion": counts,
        "row_marg": row_marg,
        "col_marg": col_marg,
        "per_category_f1": per_category_f1,
        "n": n,
    }
    return kappa, details


def pairwise_kappas(
    task_ratings: Sequence[TaskRatings], k: int = 6, with_details: bool = False
):
    """Yield (rater_i, rater_j, kappa, n_pairs[, details]) for all annotator pairs."""
    annotator_ids = sorted({aid for tr in task_ratings for aid in tr.ratings})
    results = []
    for idx_i in range(len(annotator_ids)):
        for idx_j in range(idx_i + 1, len(annotator_ids)):
            a_id = annotator_ids[idx_i]
            b_id = annotator_ids[idx_j]
            a_vals: List[Category] = []
            b_vals: List[Category] = []
            lead_a: List[float] = []
            lead_b: List[float] = []

            for tr in task_ratings:
                if a_id in tr.ratings and b_id in tr.ratings:
                    a_vals.append(tr.ratings[a_id])
                    b_vals.append(tr.ratings[b_id])
                    # Collect lead times if present
                    if a_id in tr.lead_times:
                        lead_a.append(tr.lead_times[a_id])
                    if b_id in tr.lead_times:
                        lead_b.append(tr.lead_times[b_id])

            if not a_vals:
                continue

            if with_details:
                kappa, details = weighted_cohen_kappa(
                    a_vals, b_vals, k=k, return_details=True
                )
                # Compute average lead times (seconds)
                filtered_a = _filter_outliers_iqr(lead_a)
                filtered_b = _filter_outliers_iqr(lead_b)
                combined_list = filtered_a + filtered_b
                avg_a = (
                    sum(filtered_a) / len(filtered_a) if filtered_a else float("nan")
                )
                avg_b = (
                    sum(filtered_b) / len(filtered_b) if filtered_b else float("nan")
                )
                avg_combined = (
                    sum(combined_list) / len(combined_list)
                    if combined_list
                    else float("nan")
                )
                details["avg_lead_time_a"] = avg_a
                details["avg_lead_time_b"] = avg_b
                details["avg_lead_time_combined"] = avg_combined
                results.append((a_id, b_id, kappa, len(a_vals), details))
            else:
                kappa = weighted_cohen_kappa(a_vals, b_vals, k=k)
                results.append((a_id, b_id, kappa, len(a_vals)))
    return results


def load_any(paths: Iterable[str | Path]) -> List[lsm.Task]:
    collected: List[lsm.Task] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            for f in sorted(path.iterdir()):
                if f.suffix == ".json":
                    collected.extend(lsm.load_tasks_from_file(f))
        else:
            collected.extend(lsm.load_tasks_from_file(path))
    return collected


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Quadratic weighted Cohen's kappa for 'rating' label"
    )
    parser.add_argument(
        "paths", nargs="+", help="One or more JSON export files or directories"
    )
    parser.add_argument(
        "--label", default="rating", help="Label (from_name) to use (default: rating)"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=6,
        help="Number of ordinal categories (default 6 for 0-5)",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Print confusion matrix and per-category F1 scores for each pair (A=pred, B=ref)",
    )
    parser.add_argument(
        "--csv",
        dest="csv_path",
        help="Write CSV of per-task ratings (columns: segment_id, text, per-annotator ratings, agreed_rating)",
    )
    parser.add_argument(
        "--from-csv",
        dest="from_csv",
        help=(
            "Path to an existing ratings CSV whose non-empty agreed_rating values should be reused when annotator ratings are not unanimous. "
            "If ratings become unanimous, the unanimous value overwrites any previous agreed_rating."
        ),
    )
    args = parser.parse_args(argv)

    tasks = load_any(args.paths)
    tr = collect_task_ratings(tasks, label_name=args.label)
    if not tr:
        print("No ratings found for label", args.label)
        return 1

    if args.csv_path:
        # Load any existing agreed_rating values if --from-csv was provided.
        existing_agreed: dict[int, str] = {}
        if args.from_csv:
            try:
                with open(args.from_csv, "r", encoding="utf-8", newline="") as f_in:
                    reader = csv.reader(f_in, delimiter="\t")
                    header = next(reader, None)
                    # Expect at minimum: segment_id ... agreed_rating
                    if header and len(header) >= 2 and header[0] == "segment_id":
                        for row in reader:
                            if not row:
                                continue
                            try:
                                segment_id = int(row[0])
                            except ValueError:
                                continue
                            if len(row) >= 1:
                                agreed_val = (
                                    row[-1].strip() if row[-1] is not None else ""
                                )
                                if agreed_val != "":
                                    existing_agreed[segment_id] = agreed_val
                    else:
                        print(
                            f"Warning: from-csv file '{args.from_csv}' missing expected header; ignoring."
                        )
            except FileNotFoundError:
                print(f"Warning: from-csv file '{args.from_csv}' not found; ignoring.")
            except Exception as e:  # pragma: no cover
                print(f"Warning: failed to read from-csv '{args.from_csv}': {e}")
        task_map: dict[int, lsm.Task] = {t.id: t for t in tasks}
        # Exclude annotator id 0 from CSV (treated as system / do-not-export per requirement)
        annotator_ids = sorted({aid for rec in tr for aid in rec.ratings if aid != 0})
        # Sort rows by segment_id as requested
        sorted_tr = sorted(tr, key=lambda rec: rec.task_id)
        try:
            with open(args.csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f, delimiter="\t")
                header = (
                    ["segment_id", "text"]
                    + [f"annotator_{i + 1}" for i in range(len(annotator_ids))]
                    + ["agreed_rating"]
                )
                writer.writerow(header)
                for rec in sorted_tr:
                    task = task_map.get(rec.task_id)
                    text = (
                        task.data.text if task and task.data and task.data.text else ""
                    )
                    row: List[str | int] = [rec.task_id, text]
                    per_ann_ratings: List[int | str] = []
                    complete_same: bool = True
                    first_val: int | None = None
                    for aid in annotator_ids:
                        val = rec.ratings.get(aid)
                        per_ann_ratings.append(val if val is not None else "")
                        if val is None:
                            complete_same = False
                        else:
                            if first_val is None:
                                first_val = val
                            elif val != first_val:
                                complete_same = False
                    row.extend(per_ann_ratings)
                    # Determine agreed rating output.
                    # 1. If unanimous now, override with unanimous value (even if different from existing agreed rating).
                    # 2. Else, if existing CSV provided a non-empty agreed rating for this task, carry it forward.
                    # 3. Else, leave blank.
                    if complete_same and first_val is not None:
                        row.append(first_val)
                    else:
                        prev = existing_agreed.get(rec.task_id)
                        row.append(prev if prev is not None else "")
                    writer.writerow(row)
            print(f"CSV written: {args.csv_path}")
        except Exception as e:  # pragma: no cover
            print(f"Failed to write CSV '{args.csv_path}': {e}")

    results = pairwise_kappas(tr, k=args.k, with_details=args.details)
    if not results:
        print("Not enough overlapping annotations for pairwise kappa.")
        return 1
    print(
        f"Pairwise quadratic weighted Cohen's kappa (label='{args.label}', k={args.k})"
    )
    for item in results:
        if args.details:
            a_id, b_id, kappa, n_pairs, details = item
        else:
            a_id, b_id, kappa, n_pairs = item  # type: ignore
        print(f"Annotators {a_id} vs {b_id}: kappa={kappa:.4f} (n={n_pairs})")
        if args.details:
            conf = details["confusion"]
            row_marg = details["row_marg"]
            col_marg = details["col_marg"]
            per_cat = details["per_category_f1"]
            n = details["n"]
            avg_a = details.get("avg_lead_time_a")
            avg_b = details.get("avg_lead_time_b")
            avg_c = details.get("avg_lead_time_combined")
            width = 4  # column width for counts
            # Confusion matrix header
            print(" Confusion matrix (rows=A, cols=B)")
            header = (
                "  B→   | "
                + "".join(f"{c:>{width}}" for c in range(args.k))
                + f" | {'RowΣ':>{width}}"
            )
            print(header)
            print("  " + "-" * (len(header) - 2))
            for i, row in enumerate(conf):
                row_str = "".join(f"{v:>{width}}" for v in row)
                print(f"  A {i:>2} | {row_str} | {row_marg[i]:>{width}}")
            col_str = "".join(f"{v:>{width}}" for v in col_marg)
            print(f"  ColΣ | {col_str} | {n:>{width}}")

            print("  Per-category F1:")
            for cat in reversed(range(args.k)):
                val = per_cat[cat]
                disp = f"{val:.3f}" if not math.isnan(val) else "-"
                print(f"    {cat}: {disp}")

            # Average lead times
            def _fmt(x: float | None) -> str:
                return (
                    f"{x:.2f}s"
                    if isinstance(x, (int, float)) and not math.isnan(x)
                    else "-"
                )

            print(
                f"  Avg lead time seconds: A={_fmt(avg_a)} | B={_fmt(avg_b)} | combined={_fmt(avg_c)}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
