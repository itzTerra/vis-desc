import { ref, computed, watch, onBeforeUnmount, isRef, unref } from "vue";
import type { Ref } from "vue";
import type { Highlight } from "~/types/common";

// Use `Highlight` directly; no alias needed

/*
Auto-illustration selection algorithm

This composable implements an in-order, lookahead selection algorithm used to
automatically choose highlights for which `ImageEditor` instances should open.

Key points:
- Input: a document-ordered array of `Highlight` objects (the `highlights` arg).
- Output: an ordered array of selected highlight ids. The composable itself
  adds those ids into the provided `selectedHighlights` set (union semantics)
  when `enabled` is true, and leaves manually-selected ids untouched.
- Gap mapping: 1 line ≈ 0.01 normalized page height. `minGapLines` and
  `maxGapLines` are provided in lines and converted to normalized units by
  multiplying by 0.01. This mapping is used to compute the lookahead window
  (`maxGap`) and the skip distance after choosing an item (`minGap`).
- Algorithm: iterate the highlights in document order. For each position i,
  consider the lookahead window of segments whose vertical center is within
  `maxGap` of highlight i. From that window choose the segment with the
  highest score (respecting `minScore` if set). After selecting, advance the
  scan to the first segment whose center lies beyond the chosen segment's
  center plus `minGap` to enforce spacing.

This comment block documents the WHY of the implementation; the code below
implements the behavior described and exposes helpers for UI integration.
*/
export function useAutoIllustration(opts: {
  highlights: Ref<Highlight[]> | Highlight[];
  selectedHighlights: Ref<Set<number>> | Set<number>;
  // optional page sizing helpers from PdfViewer
  pageAspectRatio?: Ref<number> | number | null;
}) {
  const { highlights, selectedHighlights } = opts;
  const pageAspectRatioRef = (opts.pageAspectRatio ?? null) as Ref<number> | number | null;

  const pageAspectRatioValue = computed(() => {
    if (isRef(pageAspectRatioRef) && typeof (pageAspectRatioRef as Ref<number>).value === "number") return (pageAspectRatioRef as Ref<number>).value;
    if (typeof pageAspectRatioRef === "number") return pageAspectRatioRef as number;
    return 1;
  });

  const enabled = ref(false);
  const minGapLines = ref(2); // lines
  const maxGapLines = ref(8);
  const minScore = ref<number | null>(null);

  const autoAddedIds = ref(new Set<number>());
  // ids that the composable asks the UI to open editors for
  const pendingOpenIds = ref<number[]>([]);

  // simple counters for downstream UI (enhance/generate flows will increment)
  const enhancedCount = ref(0);
  const generatedCount = ref(0);

  const minGapNorm = computed(() => minGapLines.value * 0.01);
  const maxGapNorm = computed(() => maxGapLines.value * 0.01);

  function getYCenterNorm(h: Highlight) {
    // simplified: look at first polygon's y values if available
    // Accept both array-shaped polygons (array of polygons)
    // and object-shaped polygons (map page -> polygon array)
    let firstPoly: any = null;
    if (Array.isArray(h.polygons) && Array.isArray(h.polygons[0])) {
      firstPoly = h.polygons[0];
    } else if (h.polygons && typeof h.polygons === "object" && !Array.isArray(h.polygons)) {
      const vals = Object.values(h.polygons);
      if (Array.isArray(vals) && vals.length > 0) {
        // vals[0] may be an array-of-polygons (e.g. [[pt,pt], [pt,pt]]) or a polygon directly
        if (Array.isArray(vals[0]) && Array.isArray((vals[0] as any)[0])) {
          firstPoly = (vals[0] as any)[0];
        } else if (Array.isArray(vals[0])) {
          firstPoly = vals[0];
        }
      }
    }
    // compute a y in 0..1 if possible (don't early return from polygon branch)
    let y: number | null = null;
    if (Array.isArray(firstPoly)) {
      const ys = firstPoly.map((p: any) => p[1]).filter((v: any) => typeof v === "number");
      if (ys.length) y = ys.reduce((a: number, b: number) => a + b, 0) / ys.length;
    }
    if (y === null) {
      if (typeof h.yCenterNorm === "number") y = h.yCenterNorm;
      else if (typeof h.centerY === "number") y = h.centerY;
    }

    // determine page index (0-based) for this highlight if available
    // Support multiple shapes: polygons map, explicit page/pageIndex fields, or default 0
    let pageIndex = 0;
    if (h.polygons && !Array.isArray(h.polygons) && typeof h.polygons === "object") {
      const keys = Object.keys(h.polygons || {});
      if (keys.length > 0) {
        const k = Number(keys[0]);
        if (Number.isFinite(k)) pageIndex = k;
      }
    }
    if ((h as any).pageIndex !== undefined && Number.isFinite((h as any).pageIndex)) pageIndex = (h as any).pageIndex;
    if ((h as any).page !== undefined && Number.isFinite((h as any).page)) pageIndex = (h as any).page;

    // Compute a document-level normalized Y so comparisons across pages are meaningful.
    // Use the unified `pageAspectRatioValue` computed above so both y-centers and
    // gap thresholds share the same scale. If we cannot compute a reliable center
    // return NaN so callers can skip this highlight instead of treating it as 0.
    const pageUnit = pageAspectRatioValue.value ?? 1;
    if (y === null || !Number.isFinite(y)) return NaN;
    return (pageIndex + y) * pageUnit;
  }

  const resolveHighlights = () => unref(highlights) || [];

  const resolveSelected = (): Set<number> => unref(selectedHighlights) as Set<number>;

  function runSelectionOnce(): number[] {
    const list = resolveHighlights() || [];
    const selected: number[] = [];
    const n = list.length;

    // scale the min/max gap by the page unit used in getYCenterNorm
    const pageUnit = pageAspectRatioValue.value;
    const minGap = minGapNorm.value * pageUnit;
    const maxGap = maxGapNorm.value * pageUnit;
    const minScoreVal = minScore.value;

    // Precompute y centers and scores to avoid repeated work inside inner loops
    const yCenters: number[] = new Array(n);
    const scores: number[] = new Array(n);
    for (let i = 0; i < n; i++) {
      const it = list[i];
      const yc = it ? getYCenterNorm(it) : NaN;
      yCenters[i] = Number.isFinite(yc) ? yc : NaN;
      scores[i] = it && typeof it.score === "number" ? it.score : -Infinity;
    }

    for (let i = 0; i < n; i++) {
      const base = list[i];
      if (!base) continue;
      const baseY = yCenters[i];
      if (!Number.isFinite(baseY)) continue;
      // build lookahead window up to maxGap
      let bestIdx = -1;
      let bestScore = -Infinity;
      for (let j = i; j < n; j++) {
        const candY = yCenters[j];
        if (!Number.isFinite(candY)) continue;
        if (candY - baseY > maxGap) break; // lookahead only forward in document order
        const s = scores[j];
        if (minScoreVal !== null && (s === -Infinity || s < minScoreVal)) continue;
        if (s > bestScore) {
          bestScore = s;
          bestIdx = j;
        }
      }

      if (bestIdx === -1) continue;
      const chosen = list[bestIdx];
      if (chosen && typeof chosen.id === "number") {
        selected.push(chosen.id);
        // advance i to index beyond chosen + minGap
        const chosenY = yCenters[bestIdx];
        let k = bestIdx + 1;
        while (k < n && Number.isFinite(yCenters[k]) && (yCenters[k] - chosenY) <= minGap) k++;
        i = k - 1;
      }
    }

    return selected;
  }

  function runPass() {
    const ids = runSelectionOnce();
    const newlyAdded: number[] = [];
    const selSet = resolveSelected();
    for (const id of ids) {
      if (!selSet.has(id)) {
        selSet.add(id);
        autoAddedIds.value.add(id);
        newlyAdded.push(id);
      }
    }
    // request UI to open editors for newly added ids
    if (newlyAdded.length) pendingOpenIds.value = newlyAdded.slice();
    return newlyAdded;
  }

  function clearAutoSelections() {
    const selSet = resolveSelected();
    for (const id of autoAddedIds.value) selSet.delete(id);
    // clear the tracking set in-place to keep the same ref object
    autoAddedIds.value.clear();
    pendingOpenIds.value = [];
  }

  function consumePendingOpenIds(): number[] {
    const list = pendingOpenIds.value.slice();
    pendingOpenIds.value = [];
    return list;
  }

  // reactively run when highlights change, but only if enabled
  let debounceTimer: ReturnType<typeof setTimeout> | null = null;
  const scheduleRun = () => {
    if (typeof window === "undefined") return; // client-only scheduling
    if (debounceTimer !== null) clearTimeout(debounceTimer);
    debounceTimer = globalThis.setTimeout(() => {
      runPass();
      debounceTimer = null;
    }, 120);
  };

  watch(() => resolveHighlights(), () => {
    if (!enabled.value) return;
    scheduleRun();
  }, { deep: true, flush: "post" });

  onBeforeUnmount(() => {
    if (debounceTimer !== null) clearTimeout(debounceTimer);
  });

  function getMaxPossible() {
    return runSelectionOnce().length;
  }

  return {
    enabled,
    minGapLines,
    maxGapLines,
    minScore,
    runPass,
    clearAutoSelections,
    // instrumentation & UI helpers
    pendingOpenIds,
    consumePendingOpenIds,
    enhancedCount,
    generatedCount,
    progress: computed(() => {
      const selSet = resolveSelected();
      return {
        selectedCount: selSet.size,
        enhancedCount: enhancedCount.value,
        generatedCount: generatedCount.value,
        maxPossible: getMaxPossible(),
      };
    }),
    getMaxPossible,
    isActive: computed(() => enhancedCount.value > 0 || generatedCount.value > 0 || enabled.value),
  };
}

