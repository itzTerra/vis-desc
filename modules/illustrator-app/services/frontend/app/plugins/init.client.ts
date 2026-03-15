import type { AppReadyState } from "~/types/common";
import { setScorerWorker } from "~/utils/scorerWorker";
import ScorerWorker from "~/workers/scorer.worker.ts?worker";

const API_SLOW_THRESHOLD_MS = 10_000;

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

  function handleApiResponse(success: boolean) {
    if (success) {
      appReady.apiReady = true;
      appReady.apiError = null;
    } else {
      appReady.apiReady = false;
      appReady.apiError = "Failed to connect to API";
    }
  }

  // Monitor all API requests: if a call doesn't return within API_SLOW_THRESHOLD_MS,
  // mark the API as not ready. When the response eventually arrives, restore state
  // using the same logic as the initial ping.
  const nativeFetch = globalThis.fetch;
  globalThis.fetch = async function (...args: Parameters<typeof fetch>) {
    const [input] = args;
    const url = typeof input === "string" ? input
      : input instanceof URL ? input.href
        : (input as Request).url;

    if (!url.includes("/api/")) {
      return nativeFetch.apply(this, args);
    }

    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      appReady.apiReady = false;
    }, API_SLOW_THRESHOLD_MS);

    try {
      const response = await nativeFetch.apply(this, args);
      clearTimeout(timer);
      if (timedOut) {
        handleApiResponse(response.ok);
      }
      return response;
    } catch (error) {
      clearTimeout(timer);
      if (timedOut) {
        handleApiResponse(false);
      }
      throw error;
    }
  };

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
