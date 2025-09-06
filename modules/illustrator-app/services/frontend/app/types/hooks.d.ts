import type { HookResult } from "@nuxt/schema";
import type { AlertOptions } from "~/components/Alert.vue";

declare module "#app" {
  interface RuntimeNuxtHooks {
    "custom:alert": (options: AlertOptions) => HookResult
  }
  // interface NuxtHooks {
  // }
}
