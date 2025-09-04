<template>
  <div
    v-for="(page, pageNum) in pageMap"
    :key="pageNum"
    class="polygon-highlight-layer pointer-events-none"
    :style="getPageLayerStyle(pageNum)"
  >
    <svg :viewBox="`0 0 ${page.width} ${page.height}`" class="absolute inset-0 w-full h-full">
      <template v-for="(poly, pIndex) in page.polygons" :key="pIndex">
        <polygon
          :points="poly.map(p => `${p[0]} ${p[1]}`).join(' ')"
          class="poly-shape"
          :data-page="pageNum"
        />
      </template>
    </svg>
  </div>
</template>

<script setup lang="ts">
import type { AlignedSegment } from "~/types/common";

const props = defineProps<{
  segments: AlignedSegment[];
  pageRefs: Element[];
}>();

const pageMap = computed(() => {
  const map: Record<number, { width: number; height: number; polygons: number[][][] }> = {};
  // Build temporary structure of normalized polygons per page
  const norm: Record<number, number[][][]> = {};
  for (const seg of props.segments) {
    for (const span of seg.page_spans) {
      if (!span.polygons?.length) continue;
      norm[span.page] = norm[span.page] || [];
      norm[span.page].push(...span.polygons);
    }
  }
  // Convert normalized to actual after reading page element size
  for (const [pageStr, polys] of Object.entries(norm)) {
    const page = Number(pageStr);
    const el = props.pageRefs[page - 1] as HTMLElement | undefined; // page numbers start at 1
    if (!el) continue;
    const rect = el.getBoundingClientRect();
    const width = rect.width;
    const height = rect.height;
    const scaledPolys = polys.map(poly => poly.map(([x, y]) => [x * width, y * height]));
    map[page] = { width, height, polygons: scaledPolys };
  }
  return map;
});

function getPageLayerStyle(pageNum: number | string): Record<string, string> {
  const el = props.pageRefs[Number(pageNum) - 1] as HTMLElement | undefined;
  if (!el) return {};
  return {
    position: "absolute",
    top: `${el.offsetTop}px`,
    left: `${el.offsetLeft}px`,
    width: `${el.clientWidth}px`,
    height: `${el.clientHeight}px`,
  };
}
</script>

<style scoped>
.polygon-highlight-layer {
  position: absolute;
  inset: 0;
  z-index: 5; /* above text layer but below UI */
}
.poly-shape {
  stroke: hsl(var(--s));
  stroke-width: 1.5;
  vector-effect: non-scaling-stroke;
  mix-blend-mode: multiply;
  rx: 2px;
}
</style>
