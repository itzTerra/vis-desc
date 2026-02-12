import { ref, computed, watch, onBeforeUnmount, isRef } from "vue";
import type { Ref } from "vue";
import type { Highlight } from "~/types/common";

type HighlightLike = Highlight;

export function useAutoIllustration(opts: {
  highlights: Ref<HighlightLike[]> | HighlightLike[];
  selectedHighlights: Ref<Set<number>> | Set<number>;
  // optional page sizing helpers from PdfViewer
  pageAspectRatio?: Ref<number> | number | null;
}) {
  const { highlights, selectedHighlights } = opts;
  const pageAspectRatioRef = (opts.pageAspectRatio ?? null) as Ref<number> | number | null;

  const pageAspectRatioValue = computed(() => {
    if (pageAspectRatioRef && typeof (pageAspectRatioRef as any).value === "number") return (pageAspectRatioRef as Ref<number>).value;
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

  function getYCenterNorm(h: HighlightLike) {
    // simplified: look at first polygon's y values if available
    // Accept both array-shaped polygons (array of polygons)
    // and object-shaped polygons (map page -> polygon array)
    let firstPoly: any = null;
    if (Array.isArray(h.polygons) && Array.isArray(h.polygons[0])) {
      firstPoly = h.polygons[0];
    } else if (h.polygons && typeof h.polygons === "object" && !Array.isArray(h.polygons)) {
      const vals = Object.values(h.polygons);
      if (Array.isArray(vals) && vals.length > 0 && Array.isArray(vals[0])) {
        firstPoly = vals[0];
      }
    }
    if (Array.isArray(firstPoly)) {
      const ys = firstPoly.map((p: any) => p[1]).filter((v: any) => typeof v === "number");
      if (ys.length) return ys.reduce((a: number, b: number) => a + b, 0) / ys.length;
    }
    let y: number = 0;
    if (typeof h.yCenterNorm === "number") y = h.yCenterNorm;
    else if (typeof h.centerY === "number") y = h.centerY;

    // determine page index (0-based) for this highlight if available
    let pageIndex = 0;
    if (h.polygons && !Array.isArray(h.polygons) && typeof h.polygons === "object") {
      const keys = Object.keys(h.polygons || {});
      if (keys.length > 0) {
        const k = Number(keys[0]);
        if (Number.isFinite(k)) pageIndex = k;
      }
    }

    // Compute a document-level normalized Y so comparisons across pages are meaningful.
    // Use pageAspectRatio when available to scale a page unit; otherwise each page is 1 unit.
    const pageUnit = (pageAspectRatioRef && typeof (pageAspectRatioRef as any).value === "number")
      ? (pageAspectRatioRef as Ref<number>).value
      : (typeof pageAspectRatioRef === "number" ? pageAspectRatioRef : 1);

    return (pageIndex + y) * pageUnit;
  }

  const resolveHighlights = () => isRef(highlights as any) ? (highlights as any).value : (highlights as any);

  const resolveSelected = (): Set<number> =>
    isRef(selectedHighlights as any) ? (selectedHighlights as any).value as Set<number> : (selectedHighlights as any as Set<number>);

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
      yCenters[i] = it ? getYCenterNorm(it) : 0;
      scores[i] = it && typeof it.score === "number" ? it.score : -Infinity;
    }

    for (let i = 0; i < n; i++) {
      const base = list[i];
      if (!base) continue;
      const baseY = yCenters[i];
      // build lookahead window up to maxGap
      let bestIdx = i;
      let bestScore = (typeof list[i].score === "number") ? list[i].score : -Infinity;
      for (let j = i; j < n; j++) {
        const candY = yCenters[j];
        if (Math.abs(candY - baseY) > maxGap) break;
        const s = scores[j];
        if (minScoreVal !== null && (s === -Infinity || s < minScoreVal)) continue;
        if (s > bestScore) {
          bestScore = s;
          bestIdx = j;
        }
      }

      const chosen = list[bestIdx];
      if (chosen && typeof chosen.id === "number") {
        selected.push(chosen.id);
        // advance i to index beyond chosen + minGap
        const chosenY = yCenters[bestIdx];
        let k = bestIdx + 1;
        while (k < n && Math.abs(yCenters[k] - chosenY) <= minGap) k++;
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
  let debounceTimer: number | undefined;
  const scheduleRun = () => {
    if (debounceTimer) clearTimeout(debounceTimer);
    debounceTimer = window.setTimeout(() => {
      runPass();
      debounceTimer = undefined;
    }, 120);
  };

  watch(() => resolveHighlights(), () => {
    if (!enabled.value) return;
    scheduleRun();
  }, { deep: true, flush: "post" });

  onBeforeUnmount(() => {
    if (debounceTimer) clearTimeout(debounceTimer);
  });

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
      // compute max possible by running selection with current settings
      const maxPossible = runSelectionOnce().length;
      return {
        selectedCount: selSet.size,
        enhancedCount: enhancedCount.value,
        generatedCount: generatedCount.value,
        maxPossible,
      };
    }),
  };
}

export default useAutoIllustration;
