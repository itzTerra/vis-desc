<template>
  <div class="polygon-highlight-layer">
    <div
      v-for="[pageNum, page] in pageMap.entries().filter(([pageNum]) => (pageNum + 1) in pageVisibility && pageVisibility[pageNum + 1])"
      :key="pageNum"
      :data-page="pageNum + 1"
      style="position: absolute; z-index: 5;"
      :style="getHighlightPageStyle(pageNum)"
    >
      <div
        v-for="(highlight, index) in page.highlights"
        v-show="highlight.score !== undefined"
        :key="index"
        class="dropdown dropdown-right dropdown-center dropdown-hover highlight-dropdown"
        :style="getHighlightStyle(highlight)"
      >
        <div
          ref="highlightRefs"
          class="highlight-dropdown-activator"
          :class="{'highlight-selected': selectedHighlights.has(highlight.id)}"
          :data-segment-id="highlight.id"
          :style="{
            clipPath: `polygon(${highlight.relPolygon.map(p => `${p[0]}px ${p[1]}px`).join(',')})`
          }"
          v-on="highlight.hasSiblings ? {
            mouseenter: (e: any) => onMouseEnter(e.currentTarget, highlight.id),
            mouseleave: (e: any) => onMouseLeave(e.currentTarget, highlight.id),
            click: () => $emit('select', highlight.id)
          } : {
            click: () => $emit('select', highlight.id)
          }"
        >
          <svg :viewBox="`0 0 ${highlight.bbox.width} ${highlight.bbox.height}`" class="absolute inset-0 w-full h-full pointer-events-none">
            <polygon
              :points="highlight.relPolygon.map(p => `${p[0]} ${p[1]}`).join(' ')"
              class="poly-shape"
            />
          </svg>
        </div>
        <div
          class="dropdown-content highlight-dropdown-content bg-base-300 hover:bg-base-200 rounded-box z-1 p-2 shadow-sm"
          :class="{'w-64': !highlight.imageLoading, 'w-72': highlight.imageLoading}"
          v-on="highlight.hasSiblings ? {
            mouseenter: (e: any) => onMouseEnter(e.currentTarget, highlight.id),
            mouseleave: (e: any) => onMouseLeave(e.currentTarget, highlight.id),
          } : {}"
        >
          <div class="hidden">
            {{ highlight.text }}
          </div>
          <div v-if="highlight.score" class="flex">
            <div class="stat p-2">
              <div class="stat-value">
                {{ (highlight.score * 100).toFixed(0) }}
              </div>
              <div class="stat-figure text-secondary">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" class="inline-block h-8 w-8 stroke-current">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              </div>
              <button class="btn btn-sm btn-primary ms-auto" :disabled="!highlight.text" @click="$emit('genImage', highlight.id)">
                <span v-if="highlight.imageLoading" class="loading loading-spinner loading-sm" />
                Generate image <Icon name="lucide:chevron-right" size="24" />
              </button>
            </div>
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
  selectedHighlights: Set<number>;
  pageRefs: Element[];
  pageVisibility: Record<number, boolean>;
  defaultPageSize: { width: number; height: number };
}>();

defineEmits<{
  select: [index: number],
  genImage: [highlightId: number],
}>();

const highlightRefs = ref<HTMLElement[]>([]);

interface HighlightRenderData {
  id: number;
  // Absolute (page-scaled) polygon coordinates
  polygon: number[][];
  // Polygon points translated relative to bounding box (0,0 origin)
  relPolygon: number[][];
  bbox: { x: number; y: number; width: number; height: number };
  hasSiblings: boolean;
  text: string;
  score?: number;
  imageLoading?: boolean;
  imageUrl?: string;
}

type PageMap = Map<number, { width: number; height: number; highlights: HighlightRenderData[] }>;

const pageMap = computed<PageMap>(() => {
  const map: PageMap = new Map();
  // Build temporary structure of normalized polygons per page
  const norm: Record<number, {
    id: number;
    polygon: number[][];
    hasSiblings: boolean;
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
        id: seg.id,
        polygon: poly,
        hasSiblings: Object.keys(seg.polygons).length > 1,
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
    map.set(page, { width, height, highlights: [] });
    for (const p of polys) {
      const scaledPoly = p.polygon.map(([x, y]) => [x * width, y * height]);
      const xs = scaledPoly.map(pt => pt[0]);
      const ys = scaledPoly.map(pt => pt[1]);
      const minX = Math.min(...xs);
      const maxX = Math.max(...xs);
      const minY = Math.min(...ys);
      const maxY = Math.max(...ys);
      const bboxWidth = Math.max(1, maxX - minX) + 1; // avoid zero width
      const bboxHeight = Math.max(1, maxY - minY) + 1; // avoid zero height
      const relPolygon = scaledPoly.map(([x, y]) => [x - minX, y - minY]);
      map.get(page)?.highlights.push({
        id: p.id,
        polygon: scaledPoly,
        relPolygon,
        bbox: { x: minX, y: minY, width: bboxWidth, height: bboxHeight },
        hasSiblings: p.hasSiblings,
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

let lastRenderedPageStyle: {
  top: string;
  left: string;
  width: string;
  height: string;
} = {
  top: "0px",
  left: "0px",
  width: `${props.defaultPageSize.width}px`,
  height: `${props.defaultPageSize.height}px`,
};

function getHighlightPageStyle(pageNum: number): Record<string, string> {
  const el = props.pageRefs[pageNum] as HTMLElement | undefined;
  if (el && (el.offsetWidth !== 300 || el.offsetHeight !== 150)) {
    lastRenderedPageStyle = {
      top: `${el.offsetTop}px`,
      left: `${el.offsetLeft}px`,
      width: `${el.clientWidth}px`,
      height: `${el.clientHeight}px`,
    };
  } else {
    console.log("!!!!!", pageNum);
    console.log(el, JSON.stringify(el), el?.offsetTop, el?.offsetLeft, el?.clientWidth, el?.clientHeight);
  }
  return {
    ...lastRenderedPageStyle,
    top: `${el?.offsetTop ?? 0}px`
  };
}

function onMouseEnter(eventTarget: any, id: number) {
  const otherHighlightsOfSegment = highlightRefs.value.filter(el => el !== eventTarget && Number(el.dataset.segmentId) === id);
  for (const el of otherHighlightsOfSegment) {
    el.classList.add("highlight-selected-hover");
  }
}

function onMouseLeave(eventTarget: any, id: number) {
  const otherHighlightsOfSegment = highlightRefs.value.filter(el => el !== eventTarget && Number(el.dataset.segmentId) === id);
  for (const el of otherHighlightsOfSegment) {
    el.classList.remove("highlight-selected-hover");
  }
}

function spawnMarker(highlightId: number) {
  const els = highlightRefs.value.filter(el => Number(el.dataset.segmentId) === highlightId);
  for (const el of els) {
    el.classList.add("highlight-marker");
    setTimeout(() => {
      el.classList.remove("highlight-marker");
    }, 2000);
  }
}

defineExpose({
  highlightRefs,
  spawnMarker,
});
</script>
