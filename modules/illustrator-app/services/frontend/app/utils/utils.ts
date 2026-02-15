export const DEFAULT_PAGE_ASPECT_RATIO = 1.4142; // A4 fallback (height / width)

export const IS_WEBGPU_AVAILABLE = typeof navigator !== "undefined" && "gpu" in navigator;

function easeInExpo(x: number): number {
  return x === 0 ? 0 : Math.pow(2, 10 * x - 10);
}

function assertIsDefined<T>(val: T): asserts val is NonNullable<T> {
  if (val === undefined || val === null) {
    throw new Error(
      `Expected 'val' to be defined, but received ${val}`
    );
  }
}

/**
 * Repeatedly attempts to scrollIntoView the referenced element until a timeout elapses.
 *
 * Returns true if successful, else false on timeout / abort.
 */
async function scrollIntoView(
  elRef: MaybeRefOrGetter<HTMLElement | null | undefined | string>,
  options: {
    timeout?: number; // ms before giving up (default 2000)
    interval?: number; // ms between attempts (default 120)
    behavior?: ScrollBehavior; // scroll behavior (default "smooth")
    block?: ScrollLogicalPosition; // block alignment (default "center")
    inline?: ScrollLogicalPosition; // inline alignment (default "nearest")
    abortSignal?: AbortSignal; // optional abort signal
    /**
     * If true, keep trying until element is at least fully visible (default true)
     */
    ensureVisible?: boolean;
    /**
     * If true, additionally try to center the element vertically (default true)
     */
    center?: boolean;
    /**
     * Allowed distance in px between element center and viewport center (default 30)
     */
    tolerance?: number;
    /**
     * Positive px to shift final vertical position upward (e.g. fixed header) (default 0)
     */
    topOffset?: number;
  } = {}
): Promise<boolean> {
  const {
    timeout = 3000,
    interval = 120,
    behavior = "smooth",
    block = "center",
    inline = "center",
    abortSignal,
    ensureVisible = true,
    center = true,
    tolerance = 30,
    topOffset = 0,
  } = options;

  const started = performance.now();

  function fullyVisible(rect: DOMRect) {
    return (
      rect.top >= 0 &&
      rect.left >= 0 &&
      rect.bottom <= window.innerHeight &&
      rect.right <= window.innerWidth
    );
  }

  function nearCentered(rect: DOMRect) {
    const elCenter = (rect.top + rect.bottom) / 2;
    const targetCenter = window.innerHeight / 2 + topOffset / 2;
    return Math.abs(elCenter - targetCenter) <= tolerance;
  }

  function adjustForTopOffset(rect: DOMRect) {
    if (topOffset !== 0) {
      // After a smooth scroll finishes, apply a corrective jump if needed.
      const overshoot = rect.top - topOffset;
      if (overshoot < 0 || rect.top !== topOffset) {
        window.scrollBy({ top: rect.top - topOffset, behavior: "instant" as ScrollBehavior });
      }
    }
  }

  function resolveElement(val: HTMLElement | null | undefined | string): HTMLElement | null {
    if (typeof val === "string") {
      const found = document.querySelector(val);
      return (found instanceof HTMLElement) ? found : null;
    }
    return val ?? null;
  }

  return new Promise<boolean>((resolve) => {
    const attempt = async () => {
      if (abortSignal?.aborted) {
        resolve(false);
        return;
      }

      const raw = toValue(elRef);
      const el = resolveElement(raw);
      if (!el) {
        if (performance.now() - started >= timeout) {
          resolve(false);
          return;
        }
        setTimeout(attempt, interval);
        return;
      }

      const rect = el.getBoundingClientRect();

      if (
        (!ensureVisible || fullyVisible(rect)) &&
        (!center || nearCentered(rect))
      ) {
        adjustForTopOffset(rect);
        resolve(true);
        return;
      }

      // Perform scroll attempt
      try {
        el.scrollIntoView({ behavior, block, inline });
      } catch {
        // ignore
      }

      await nextTick();
      if (performance.now() - started >= timeout) {
        resolve(false);
        return;
      }
      setTimeout(attempt, interval);
    };

    attempt();
  });
}

function clamp(v: number, min: number, max: number) {
  return v < min ? min : (v > max ? max : v);
}

function lerp(start: number, end: number, t: number): number {
  return start + (end - start) * t;
}

/**
 * Unicode-aware casefold helper using NFKC normalization and small mappings.
 * This is a conservative casefold implementation suitable for feature extraction.
 */
function casefold(s: string): string {
  if (!s) return s;
  // Normalize to NFKC to combine compatibility characters
  let v = s.normalize("NFKC");
  // Basic mappings (extend if needed)
  v = v.replace(/\u00DF/g, "ss"); // German sharp s
  v = v.replace(/\u017F/g, "s"); // long s
  // Final lowercase (locale-insensitive)
  return v.toLowerCase();
}

export { easeInExpo, assertIsDefined, scrollIntoView, clamp, lerp, casefold };
