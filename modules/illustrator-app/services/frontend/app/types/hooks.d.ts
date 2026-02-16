import type { HookResult } from "@nuxt/schema";
import type { AlertOptions } from "~/components/Alert.vue";
import type { Highlight } from "~/types/common";

declare module "#app" {
  interface RuntimeNuxtHooks {
    "custom:alert": (options: AlertOptions) => HookResult,
    "custom:goToHighlight": (highlight: Highlight) => HookResult,
    "custom:downloadNeeded": (groupId: string) => HookResult,
  }
  // interface NuxtHooks {
  // }
}
