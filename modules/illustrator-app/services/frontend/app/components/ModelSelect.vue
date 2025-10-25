<template>
  <div class="dropdown bg-base-100 border border-base-content/25 rounded">
    <div
      tabindex="0"
      role="button"
      class="btn btn-ghost btn-sm w-auto text-nowrap ps-2"
    >
      <Icon name="lucide:component" />
      {{ MODELS.find(m => m.value === modelValue)?.label ?? "??" }}
    </div>
    <ul tabindex="0" class="dropdown-content z-[1] menu p-2 shadow bg-base-300 rounded-box w-96 ml-[-1px]">
      <li class="menu-title">
        <div class="grid grid-cols-[1fr_50px_50px] gap-4 text-sm font-semibold">
          <span>Model</span>
          <span>Speed</span>
          <span>Quality</span>
        </div>
      </li>
      <li v-for="model in MODELS" :key="model.value" class="border-t border-base-content/10" :class="{ 'menu-disabled': model.disabled }">
        <a
          :title="model.description"
          class="block"
          @click="() => selectModel(model)"
        >
          <div class="grid grid-cols-[1fr_50px_50px] gap-4 text-sm items-center">
            <span>{{ model.label }}</span>
            <div class="grid grid-cols-5 border border-base-content/20 gap-x-[1px] h-4">
              <div v-for="i in model.speed" :key="i" class="bg-secondary" :class="{ 'opacity-50': model.disabled }" />
            </div>
            <div class="grid grid-cols-5 border border-base-content/20 gap-x-[1px] h-4">
              <div v-for="i in model.quality" :key="i" class="bg-primary" :class="{ 'opacity-50': model.disabled }" />
            </div>
          </div>
        </a>
      </li>
      <li class="menu-title border-t border-base-content/10">
        <small class="text-muted italic font-normal">Speed and quality are just orientational and not to scale.</small>
      </li>
    </ul>
  </div>
</template>

<script setup lang="ts">
const modelValue = defineModel<ModelValue>({ required: true });

function selectModel(model: Model) {
  if (!model.disabled) {
    modelValue.value = model.value;
  }
  (document.activeElement as HTMLElement)?.blur();
}
</script>
