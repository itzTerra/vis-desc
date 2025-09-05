<template>
  <div
    v-for="(page, pageNum) in pageMap"
    :key="pageNum"
    class="polygon-highlight-layer"
    :style="getPageLayerStyle(pageNum)"
  >
    <div
      v-for="(highlight, index) in page.highlights"
      :key="index"
      ref="highlightRefs"
      class="dropdown dropdown-right dropdown-center dropdown-hover highlight-dropdown"
      :style="getHighlightStyle(highlight)"
    >
      <div class="highlight-dropdown-content" :class="dropdownContentClasses[index]" @click="$emit('select', index)">
        <svg :viewBox="`0 0 ${highlight.bbox.width} ${highlight.bbox.height}`" class="absolute inset-0 w-full h-full pointer-events-none">
          <polygon
            :points="highlight.relPolygon.map(p => `${p[0]} ${p[1]}`).join(' ')"
            class="poly-shape"
            :title="highlight.text"
            :data-page="pageNum"
          />
        </svg>
      </div>
      <div class="dropdown-content bg-base-100 hover:bg-base-200 rounded-box z-1 w-64 p-2 shadow-sm">
        <div class="hidden">
          {{ highlight.text }}
        </div>
        <div v-if="highlight.score" class="flex">
          <div class="stat pa-2">
            <div class="stat-value">
              {{ (highlight.score * 100).toFixed(0) }}
            </div>
            <div class="stat-figure text-secondary">
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" class="inline-block h-8 w-8 stroke-current">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
            <button class="btn btn-sm btn-primary ms-auto" :disabled="!highlight.text" @click="genImage(highlight)">
              Generate image <Icon name="lucide:chevron-right" size="24" />
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import type { Highlight } from "~/types/common";

const props = defineProps<{
  highlights: Highlight[];
  selectedHighlights: Highlight[];
  pageRefs: Element[];
}>();

defineEmits<{
  select: [index: number]
}>();

const highlightRefs = ref<HTMLElement[]>([]);
defineExpose({
  highlightRefs,
});

interface HighlightRenderData {
  // Absolute (page-scaled) polygon coordinates
  polygon: number[][];
  // Polygon points translated relative to bounding box (0,0 origin)
  relPolygon: number[][];
  bbox: { x: number; y: number; width: number; height: number };
  text: string;
  score?: number;
  imageLoading?: boolean;
  imageUrl?: string;
}

type PageMap = Record<number, { width: number; height: number; highlights: HighlightRenderData[] }>;

const pageMap = computed<PageMap>(() => {
  const map: PageMap = {};
  // Build temporary structure of normalized polygons per page
  const norm: Record<number, {
  polygon: number[][];
  text: string;
  score?: number;
  imageLoading?: boolean;
  imageUrl?: string;
}[]> = {};
  for (const seg of props.highlights) {
    for (const [page, poly] of Object.entries(seg.polygons)) {
      if (!poly.length) continue;
      const pageNum = Number(page);
      norm[pageNum] = norm[pageNum] || [];
      norm[pageNum].push({
        polygon: poly,
        text: seg.text,
        score: seg.score,
        imageLoading: seg.imageLoading,
        imageUrl: seg.imageUrl,
      });
    }
  }
  // Convert normalized to actual after reading page element size
  for (const [pageStr, polys] of Object.entries(norm)) {
    const page = Number(pageStr);
    const el = props.pageRefs[page] as HTMLElement | undefined; // page numbers start at 1
    if (!el) continue;
    const rect = el.getBoundingClientRect();
    const width = rect.width;
    const height = rect.height;
    map[page] = { width, height, highlights: [] };
    for (const p of polys) {
      const scaledPoly = p.polygon.map(([x, y]) => [x * width, y * height]);
      const xs = scaledPoly.map(pt => pt[0]);
      const ys = scaledPoly.map(pt => pt[1]);
      const minX = Math.min(...xs);
      const maxX = Math.max(...xs);
      const minY = Math.min(...ys);
      const maxY = Math.max(...ys);
      const bboxWidth = Math.max(1, maxX - minX); // avoid zero width
      const bboxHeight = Math.max(1, maxY - minY); // avoid zero height
      const relPolygon = scaledPoly.map(([x, y]) => [x - minX, y - minY]);
      map[page].highlights.push({
        polygon: scaledPoly,
        relPolygon,
        bbox: { x: minX, y: minY, width: bboxWidth, height: bboxHeight },
        text: p.text,
        score: p.score,
        imageLoading: p.imageLoading,
        imageUrl: p.imageUrl,
      });
    }
  }
  return map;
});

function getHighlightStyle(h: HighlightRenderData): Record<string, string> {
  return {
    position: "absolute",
    top: `${h.bbox.y}px`,
    left: `${h.bbox.x}px`,
    width: `${h.bbox.width}px`,
    height: `${h.bbox.height}px`,
  };
}

function getPageLayerStyle(pageNum: number | string): Record<string, string> {
  const el = props.pageRefs[Number(pageNum)] as HTMLElement | undefined;
  if (!el) return {};
  return {
    position: "absolute",
    top: `${el.offsetTop}px`,
    left: `${el.offsetLeft}px`,
    width: `${el.clientWidth}px`,
    height: `${el.clientHeight}px`,
  };
}

const dropdownContentClasses = computed(() =>
  props.highlights.map((highlight) => ({
    // "highlight-dropdown-content__hovered": hoveredHighlightIndex.value === index,
    "highlight-selected": props.selectedHighlights.includes(highlight),
  }))
);

// const hoveredHighlightIndex = ref<number | null>(null);
// function onHighlightHover(index: number) {
//   hoveredHighlightIndex.value = index;
//   console.log(`Hovered over highlight at index: ${index}`);
// }
// function onHighlightLeave() {
//   hoveredHighlightIndex.value = null;
//   console.log("Mouse left highlight");
// }

const { $api } = useNuxtApp();

async function genImage(highlight: Omit<Highlight, "score" | "polygons">) {
  if (!highlight.text) return;

  highlight.imageLoading = true;
  const res = await $api("/api/gen-image-bytes", {
    method: "POST",
    body: { text: highlight.text }
  });
  console.log(res);
  const blob = new Blob([res as any], { type: "image/png" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.target = "_blank";
  link.click();
  highlight.imageUrl = url;
  highlight.imageLoading = false;
  return { blob, url };
}
</script>

<style scoped>
.polygon-highlight-layer {
  position: absolute;
  inset: 0;
  z-index: 5; /* above text layer but below UI */
}
.poly-shape {
  rx: 2px;
  opacity: 0.1;
}
</style>
