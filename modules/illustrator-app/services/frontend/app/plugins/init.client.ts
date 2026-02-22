import type { AppReadyState } from "~/types/common";
import { setScorerWorker } from "~/utils/scorerWorker";
import ScorerWorker from "~/workers/scorer.worker.ts?worker";

export default defineNuxtPlugin(async (nuxtApp) => {
  const appReady: AppReadyState = reactive({
    apiReady: false,
    apiError: null,
    scorerWorkerReady: false,
  });
  nuxtApp.provide("appReadyState", appReady);

  if (nuxtApp.$debugPanel) {
    (nuxtApp.$debugPanel as any).track("appReady", appReady);
  }

  (nuxtApp.$api as any)("/api/ping", { method: "GET" }).then(() => {
    appReady.apiReady = true;
  }).catch((error: any) => {
    console.error("API ping failed:", error);
    useNotifier().error("Failed to connect to API. Please check your connection and try again.");
    appReady.apiError = "Failed to connect to API";
  });

  if (!appReady.scorerWorkerReady) {
    const scorerWorker = new ScorerWorker();
    setScorerWorker(scorerWorker);
    appReady.scorerWorkerReady = true;
  }
});

declare module "#app" {
	// Nuxt 3 injection augmentation
	interface NuxtApp { $appReadyState: AppReadyState }
}
declare module "vue" {
	interface ComponentCustomProperties { $appReadyState: AppReadyState }
}

export {}; // ensure module scope
