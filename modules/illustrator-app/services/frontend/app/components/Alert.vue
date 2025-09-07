<template>
  <div v-if="isVisible" class="toast toast-top toast-right z-[9999]" @click="closeAlert">
    <div class="alert" :class="computedClass">
      <Icon
        :name="{
          info: 'lucide:info',
          success: 'lucide:check',
          warning: 'lucide:alert-triangle',
          error: 'lucide:alert-circle',
        }[alertType]"
        size="24"
        class="shrink-0"
      />
      <span>{{ alertMessage }}</span>
    </div>
  </div>
</template>

<script setup lang="ts">
export interface AlertOptions {
  type: "info" | "success" | "warning" | "error"; // default to "error"
  message: string;
  duration?: number; // in milliseconds, default to 3000ms
}

const nuxtApp = useNuxtApp();
nuxtApp.hook("custom:alert", (options: AlertOptions) => {
  showAlert(options);
});

let alertTimeout: ReturnType<typeof setTimeout> | null = null;
const isVisible = ref(false);
const alertMessage = ref("");
const alertType = ref<AlertOptions["type"]>("error");

const computedClass = computed(() => {
  switch (alertType.value) {
  case "info":
    return "alert-info";
  case "success":
    return "alert-success";
  case "warning":
    return "alert-warning";
  case "error":
  default:
    return "alert-error";
  }
});

function showAlert({ type = "error", message, duration = 3000 }: AlertOptions) {
  alertMessage.value = message;
  alertType.value = type;
  isVisible.value = true;
  if (alertTimeout) {
    clearTimeout(alertTimeout);
  }
  alertTimeout = setTimeout(closeAlert, duration);
}

function closeAlert() {
  isVisible.value = false;
  if (alertTimeout) {
    clearTimeout(alertTimeout);
    alertTimeout = null;
  }
}
</script>
