<template>
  <Teleport to="body">
    <div class="help-overlay" @click="cancel">
      <div class="scrim" :style="{ clipPath: clipPath }" />
      <div class="message-box bg-base-100" :style="{ top: messageTop + 'px', left: messageLeft + 'px' }" @click.stop>
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

<script setup lang="ts">
export interface Step {
  selector: string;
  title: string;
  message: string;
  position: "top" | "bottom" | "left" | "right";
  onEnter?: () => void;
  onLeave?: () => void;
}

const emit = defineEmits<{
  cancel: []
}>();

const props = defineProps<{
  helpSteps: Step[];
}>();

const currentHelpStep = ref(-1);
onMounted(() => {
  next();
});

const clipPath = ref("");
const messageTop = ref(0);
const messageLeft = ref(0);
const messageWidth = 400;
const messageHeight = 150;

watch(currentHelpStep, () => {
  nextTick(() => {
    const step = props.helpSteps[currentHelpStep.value];
    const el = document.querySelector(step.selector) as HTMLElement;
    if (!el) return;
    const rect = getSubtreeBoundingRect(el);
    const windowWidth = window.innerWidth;
    const windowHeight = window.innerHeight;
    // Compute clip-path for hole
    const x1 = (rect.left / windowWidth) * 100;
    const y1 = (rect.top / windowHeight) * 100;
    const x2 = ((rect.left + rect.width) / windowWidth) * 100;
    const y2 = ((rect.top + rect.height) / windowHeight) * 100;
    clipPath.value = `polygon(0% 0%, 100% 0%, 100% 100%, 0% 100%, 0% 0%, ${x1}% ${y1}%, ${x1}% ${y2}%, ${x2}% ${y2}%, ${x2}% ${y1}%, ${x1}% ${y1}%)`;
    // Position message box
    let top = 0, left = 0;
    const HOLE_MARGIN = 10;
    const EDGE_MARGIN = 20;
    switch (step.position) {
    case "top":
      top = rect.top - messageHeight - HOLE_MARGIN;
      left = rect.left + rect.width / 2 - messageWidth / 2;
      break;
    case "bottom":
      top = rect.bottom + HOLE_MARGIN;
      left = rect.left + rect.width / 2 - messageWidth / 2;
      break;
    case "left":
      top = rect.top + rect.height / 2 - messageHeight / 2;
      left = rect.left - messageWidth - HOLE_MARGIN;
      break;
    case "right":
      top = rect.top + rect.height / 2 - messageHeight / 2;
      left = rect.right + HOLE_MARGIN;
      break;
    }
    // Clamp to screen
    if (left < EDGE_MARGIN) left = EDGE_MARGIN;
    if (left + messageWidth > windowWidth - EDGE_MARGIN) left = windowWidth - messageWidth - EDGE_MARGIN;
    if (top < EDGE_MARGIN) top = EDGE_MARGIN;
    if (top + messageHeight > windowHeight - EDGE_MARGIN) top = windowHeight - messageHeight - EDGE_MARGIN;
    messageTop.value = top;
    messageLeft.value = left;
  });
});

function prev() {
  if (currentHelpStep.value > 0) {
    const step = props.helpSteps[currentHelpStep.value];
    step.onLeave?.();
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
  transition: top 0.3s ease, left 0.3s ease;
}
</style>
