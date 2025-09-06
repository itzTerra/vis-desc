<template>
  <button
    v-show="isVisible"
    class="fixed bottom-12 right-2 btn btn-primary"
    aria-label="Back to top"
    @click="scrollToTop"
  >
    <Icon name="lucide:chevron-up" size="24" />
  </button>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from "vue";

const isVisible = ref(false);

const handleScroll = () => {
  isVisible.value = window.scrollY > 200;
};

const scrollToTop = () => {
  window.scrollTo({ top: 0, behavior: "smooth" });
};

onMounted(() => {
  window.addEventListener("scroll", handleScroll);
});

onUnmounted(() => {
  window.removeEventListener("scroll", handleScroll);
});
</script>

<style scoped>
button {
  z-index: 50;
}
</style>
