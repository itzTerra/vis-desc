import type { AlertOptions } from "~/types/common";

export default function useNotifier() {
  const notify = (options: AlertOptions) => {
    const nuxtApp = useNuxtApp();
    nuxtApp.callHook("custom:alert", options);
  };

  const error = (message: string, opts?: Partial<AlertOptions>) => {
    notify({
      type: "error",
      message,
      ...opts,
    });
  };

  const success = (message: string, opts?: Partial<AlertOptions>) => {
    notify({
      type: "success",
      message,
      ...opts,
    });
  };

  const warning = (message: string, opts?: Partial<AlertOptions>) => {
    notify({
      type: "warning",
      message,
      ...opts,
    });
  };

  const info = (message: string, opts?: Partial<AlertOptions>) => {
    notify({
      type: "info",
      message,
      ...opts,
    });
  };

  return {
    notify,
    error,
    success,
    warning,
    info,
  };
}
