<template>
  <Teleport to="body">
    <div class="help-overlay" @click="cancel">
      <!-- Default: box-shadow scrim — border-radius-aware, reads radius from the element -->
      <div
        v-if="currentStep && currentStep.clipPath === undefined"
        class="highlight"
        :style="{
          top: `${highlightTop}px`,
          left: `${highlightLeft}px`,
          width: `${highlightWidth}px`,
          height: `${highlightHeight}px`,
          borderRadius: highlightBorderRadius,
        }"
      />
      <!-- Custom clipPath override: full-screen polygon scrim -->
      <div
        v-else-if="currentStep?.clipPath !== undefined"
        class="scrim"
        :style="{ clipPath: clipPath }"
      />
      <div class="message-box bg-base-100" :style="messageBoxStyle" @click.stop>
        <p class="font-bold">
          {{ helpSteps[Math.max(currentHelpStep, 0)].title }}
        </p>
        <p>{{ helpSteps[Math.max(currentHelpStep, 0)].message }}</p>
        <div class="flex justify-end pt-4">
          <!-- <button class="btn btn-sm btn-error" @click="cancel">
            <Icon name="lucide:x" size="16px" />
            Close
          </button> -->
          <div class="flex items-center space-x-2 bg-base-200">
            <button class="btn btn-sm" :disabled="currentHelpStep <= 0" @click="prev">
              <Icon name="lucide:chevron-left" size="18px" />
            </button>
            <div>{{ currentHelpStep + 1 }} / {{ helpSteps.length }}</div>
            <button class="btn btn-sm btn-primary" @click="next">
              <Icon v-if="currentHelpStep < helpSteps.length - 1" name="lucide:chevron-right" size="18px" />
              <Icon v-else name="lucide:check" size="18px" />
            </button>
          </div>
        </div>
        <button class="btn btn-sm btn-ghost btn-circle absolute top-1 right-1" @click="cancel">
          ✕
        </button>
      </div>
    </div>
  </Teleport>
</template>

<script lang="ts">
export interface Step {
  selector: string;
  title: string;
  message: string;
  position: "top" | "bottom" | "left" | "right";
  /** Override the scrim clip-path. The selector is still used for message-box positioning. */
  clipPath?: string | (() => string);
  /** Pixel offset applied to the highlight rect (and message-box anchor). Useful for fine-tuning the cutout position. */
  transform?: { x?: number; y?: number };
  onEnter?: () => void;
  onLeave?: () => void;
}

export function computeClipPathFromRect(rect: DOMRect): string {
  const w = window.innerWidth;
  const h = window.innerHeight;
  const x1 = (rect.left / w) * 100;
  const y1 = (rect.top / h) * 100;
  const x2 = ((rect.left + rect.width) / w) * 100;
  const y2 = ((rect.top + rect.height) / h) * 100;
  return `polygon(0% 0%, 100% 0%, 100% 100%, 0% 100%, 0% 0%, ${x1}% ${y1}%, ${x1}% ${y2}%, ${x2}% ${y2}%, ${x2}% ${y1}%, ${x1}% ${y1}%)`;
}

export function computeClipPathFromElement(el: Element): string {
  return computeClipPathFromRect(el.getBoundingClientRect());
}
</script>

<script setup lang="ts">
const emit = defineEmits<{
  cancel: []
}>();

const props = defineProps<{
  helpSteps: Step[];
}>();

const currentHelpStep = ref(-1);
const currentStep = computed(() =>
  currentHelpStep.value >= 0 ? props.helpSteps[currentHelpStep.value] : undefined
);

onMounted(() => {
  next();
});

// Box-shadow highlight state (default)
const highlightTop = ref(0);
const highlightLeft = ref(0);
const highlightWidth = ref(0);
const highlightHeight = ref(0);
const highlightBorderRadius = ref("0px");

// Clip-path scrim state (custom clipPath override)
const clipPath = ref("");

const messageBoxStyle = ref<Record<string, string>>({});
const messageWidth = 400;
const messageHeight = 150;

function positionMessageBox(rect: DOMRect, position: Step["position"]) {
  const windowWidth = window.innerWidth;
  const windowHeight = window.innerHeight;
  let top: number | null = null, left: number | null = null;
  let bottom: number | null = null, right: number | null = null;
  const HOLE_MARGIN = 10;
  const EDGE_MARGIN = 20;
  switch (position) {
  case "top":
    bottom = windowHeight - rect.top + HOLE_MARGIN;
    left = rect.left + rect.width / 2 - messageWidth / 2;
    break;
  case "bottom":
    top = rect.bottom + HOLE_MARGIN;
    left = rect.left + rect.width / 2 - messageWidth / 2;
    break;
  case "left":
    top = rect.top + rect.height / 2 - messageHeight / 2;
    right = windowWidth - rect.left + HOLE_MARGIN;
    break;
  case "right":
    top = rect.top + rect.height / 2 - messageHeight / 2;
    left = rect.right + HOLE_MARGIN;
    break;
  }
  if (left !== null) {
    if (left < EDGE_MARGIN) left = EDGE_MARGIN;
    if (left + messageWidth > windowWidth - EDGE_MARGIN) left = windowWidth - messageWidth - EDGE_MARGIN;
  }
  if (right !== null) {
    if (right < EDGE_MARGIN) right = EDGE_MARGIN;
    if (right + messageWidth > windowWidth - EDGE_MARGIN) right = windowWidth - messageWidth - EDGE_MARGIN;
  }
  if (top !== null) {
    if (top < EDGE_MARGIN) top = EDGE_MARGIN;
    if (top + messageHeight > windowHeight - EDGE_MARGIN) top = windowHeight - messageHeight - EDGE_MARGIN;
  }
  if (bottom !== null) {
    if (bottom < EDGE_MARGIN) bottom = EDGE_MARGIN;
    if (bottom + messageHeight > windowHeight - EDGE_MARGIN) bottom = windowHeight - messageHeight - EDGE_MARGIN;
  }
  const style: Record<string, string> = {};
  if (top !== null) style.top = top + "px";
  if (left !== null) style.left = left + "px";
  if (bottom !== null) style.bottom = bottom + "px";
  if (right !== null) style.right = right + "px";
  messageBoxStyle.value = style;
}

watch(currentHelpStep, async () => {
  await nextTick();
  await wait(500);
  nextTick(() => {
    const step = props.helpSteps[currentHelpStep.value];
    const el = document.querySelector(step.selector) as HTMLElement;
    if (!el) return;

    const elRect = getSubtreeBoundingRect(el);
    const tx = step.transform?.x ?? 0;
    const ty = step.transform?.y ?? 0;
    const rect = (tx || ty) ? new DOMRect(elRect.left + tx, elRect.top + ty, elRect.width, elRect.height) : elRect;

    if (step.clipPath !== undefined) {
      // Custom clip-path override: polygon scrim, message box from subtree bounds (+ transform).
      clipPath.value = typeof step.clipPath === "function" ? step.clipPath() : step.clipPath;
      positionMessageBox(rect, step.position);
    } else {
      // Default: box-shadow highlight from the element's own rect + border-radius (+ transform).
      highlightTop.value = rect.top;
      highlightLeft.value = rect.left;
      highlightWidth.value = rect.width;
      highlightHeight.value = rect.height;
      highlightBorderRadius.value = getComputedStyle(el).borderRadius;
      positionMessageBox(rect, step.position);
    }
  });
});

function prev() {
  if (currentHelpStep.value > 0) {
    const step = props.helpSteps[currentHelpStep.value];
    step.onLeave?.();
    props.helpSteps[currentHelpStep.value - 1].onEnter?.();
    currentHelpStep.value--;
  }
}

function next() {
  if (currentHelpStep.value < props.helpSteps.length - 1) {
    if (currentHelpStep.value >= 0) {
      props.helpSteps[currentHelpStep.value].onLeave?.();
    }
    props.helpSteps[currentHelpStep.value + 1].onEnter?.();
    currentHelpStep.value++;
  } else {
    cancel();
  }
}

function cancel() {
  const step = props.helpSteps[currentHelpStep.value];
  step.onLeave?.();
  emit("cancel");
  currentHelpStep.value = 0;
}

function getSubtreeBoundingRect(element: HTMLElement): DOMRect {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;

  function traverse(el: Element) {
    if (!(el instanceof HTMLElement)) return;
    const style = getComputedStyle(el);
    if (style.display === "none" || style.visibility === "hidden" || el.offsetWidth === 0 || el.offsetHeight === 0) {
      return;
    }
    const rect = el.getBoundingClientRect();
    minX = Math.min(minX, rect.left);
    minY = Math.min(minY, rect.top);
    maxX = Math.max(maxX, rect.right);
    maxY = Math.max(maxY, rect.bottom);
    for (const child of el.children) {
      traverse(child);
    }
  }

  traverse(element);
  return new DOMRect(minX, minY, maxX - minX, maxY - minY);
}
</script>

<style scoped>
.help-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  z-index: 1000;
}

.highlight {
  position: absolute;
  pointer-events: none;
  box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.7);
  transition: top 0.3s ease, left 0.3s ease, width 0.3s ease, height 0.3s ease, border-radius 0.3s ease;
}

.scrim {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0, 0, 0, 0.7);
}

.message-box {
  position: absolute;
  width: 400px;
  border-radius: 8px;
  padding: 16px;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
  z-index: 20;
  transition: top 0.3s ease, left 0.3s ease, bottom 0.3s ease, right 0.3s ease;
}
</style>
