<template>
  <div class="auto-illustration-status" :title="tooltipText" :aria-label="tooltipText" role="group">
    <div class="bar" aria-hidden="true">
      <div class="segment selected" :style="{ width: normalized.selected + '%' }"></div>
      <div class="segment enhanced" :class="{ active: isActive && enhancedCount>0 }" :style="{ width: normalized.enhanced + '%' }"></div>
      <div class="segment generated" :class="{ active: isActive && generatedCount>0 }" :style="{ width: normalized.generated + '%' }"></div>
    </div>
    <div class="counts text-xs ml-2" aria-live="polite" aria-atomic="true">
      <span class="text-muted">Sel:</span> <strong>{{ selectedCount }}</strong>
      <span class="pl-2 text-muted">Enh:</span> <strong>{{ enhancedCount }}</strong>
      <span class="pl-2 text-muted">Gen:</span> <strong>{{ generatedCount }}</strong>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from "vue";
// Name the component for devtools and consistent conventions
defineOptions({ name: "AutoIllustrationStatusBar" });
interface Progress { selectedCount: number; enhancedCount: number; generatedCount: number; maxPossible: number }
const props = defineProps<{ progress: Progress; isActive?: boolean }>();
const isActive = computed(() => !!props.isActive);

const selectedCount = computed(() => props.progress?.selectedCount ?? 0);
const enhancedCount = computed(() => props.progress?.enhancedCount ?? 0);
const generatedCount = computed(() => props.progress?.generatedCount ?? 0);
const maxPossible = computed(() => Math.max(1, props.progress?.maxPossible ?? 1));

// Normalize percentages so the three bars never overflow the container
const normalized = computed(() => {
  const raw = {
    selected: (selectedCount.value / maxPossible.value) * 100,
    enhanced: (enhancedCount.value / maxPossible.value) * 100,
    generated: (generatedCount.value / maxPossible.value) * 100,
  };
  const total = raw.selected + raw.enhanced + raw.generated;
  const scale = total > 100 ? 100 / total : 1;
  const sel = Math.round(raw.selected * scale);
  const enh = Math.round(raw.enhanced * scale);
  // ensure the three values sum to 100 to avoid tiny gaps/overflow
  const gen = Math.max(0, 100 - (sel + enh));
  return { selected: sel, enhanced: enh, generated: gen };
});

const tooltipText = computed(() => {
  return `Selected: ${selectedCount.value} / ${maxPossible.value}, Enhanced: ${enhancedCount.value}, Generated: ${generatedCount.value}, Active: ${isActive.value ? "Yes" : "No"}`;
});
</script>

<style scoped>
.auto-illustration-status {
  /* local theme tokens (fallbacks provided) */
  --ai-selected-start: #60a5fa;
  --ai-selected-end: #3b82f6;
  --ai-enhanced-start: #34d399;
  --ai-enhanced-end: #10b981;
  --ai-generated-start: #f97316;
  --ai-generated-end: #fb923c;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}
.bar { display:flex; width:96px; height:10px; background:var(--p-200); border-radius:6px; overflow:hidden; }
.segment { height:100%; position:relative; border-radius: inherit; }
.segment.selected { background: linear-gradient(90deg, var(--ai-selected-start, #60a5fa), var(--ai-selected-end, #3b82f6)); }
.segment.enhanced { background: linear-gradient(90deg, var(--ai-enhanced-start, #34d399), var(--ai-enhanced-end, #10b981)); }
.segment.generated { background: linear-gradient(90deg, var(--ai-generated-start, #f97316), var(--ai-generated-end, #fb923c)); }
.segment.active { box-shadow: 0 0 6px rgba(0,0,0,0.08) inset; }
.counts { color:var(--p-600); }

/* Shared overlay base to avoid repetition */
.segment::after {
  content: '';
  position: absolute;
  pointer-events: none;
  border-radius: inherit;
  opacity: 0.9;
  will-change: transform, background-position;
}

/* Enhanced shimmer (keeps only the differing properties) */
.segment.enhanced.active::after {
  top: 0; left: 0; bottom: 0;
  width: 40%;
  background: linear-gradient(90deg, rgba(255,255,255,0) 0%, rgba(255,255,255,0.35) 50%, rgba(255,255,255,0) 100%);
  transform: translateX(-120%);
  animation: shimmer-translate 2.4s ease-in-out infinite;
}

/* Generated diagonal stripes */
.segment.generated.active::after {
  inset: 0;
  background-image: repeating-linear-gradient(135deg, rgba(255,255,255,0.06) 0 6px, transparent 6px 12px);
  background-size: 24px 24px;
  animation: stripes-move 3s linear infinite;
}

@keyframes shimmer-translate {
  0% { transform: translateX(-120%); }
  50% { transform: translateX(20%); }
  100% { transform: translateX(120%); }
}

@keyframes stripes-move {
  from { background-position: 0 0; }
  to { background-position: 24px 24px; }
}

/* Respect user preference for reduced motion */
@media (prefers-reduced-motion: reduce) {
  .segment.enhanced.active::after,
  .segment.generated.active::after {
    /* stop motion but keep a subtle, non-animated overlay for visual affordance */
    animation: none;
    transform: none;
    background-position: initial;
    opacity: 0.35;
  }
}

</style>
