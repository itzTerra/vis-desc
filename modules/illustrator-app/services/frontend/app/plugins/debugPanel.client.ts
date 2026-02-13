// Lightweight in-app debug panel for inspecting live variables during development.
// Usage examples (anywhere in app after plugin is loaded):
// const { $debugPanel } = useNuxtApp();
// $debugPanel.add("now", () => new Date().toISOString());
// const count = ref(0); $debugPanel.track("count", count);
// Press Ctrl+Shift+D to toggle the panel.

import type { Ref, ComputedRef } from "vue";

interface DebugPanelItem {
	name: string;
	getter: () => any;
	last?: any;
}

interface DebugPanelAction {
  label: string;
  action: () => void;
  title?: string;
}

interface DebugPanelApi {
	add: (name: string, getter: () => any) => void;
	track: (name: string, source: Ref<any> | ComputedRef<any> | Record<string, any>) => void;
	remove: (name: string) => void;
	clear: () => void;
	list: () => string[];
  addAction: (label: string, action: () => void, title?: string) => void;
  removeAction: (label: string) => void;
	show: () => void;
	hide: () => void;
	toggle: () => void;
	isVisible: () => boolean;
}

const KEYBIND = { ctrl: true, shift: true, code: "KeyD" }; // Ctrl+Shift+D
const REFRESH_INTERVAL_MS = 500;

function createPanel(): { api: DebugPanelApi } {
  const items: DebugPanelItem[] = [];
  const actions: DebugPanelAction[] = [];
  let container: HTMLDivElement | null = null;
  let tableBody: HTMLTableSectionElement | null = null;
  let actionsContainer: HTMLDivElement | null = null;
  let visible = false;
  let timer: number | null = null;

  const ensureDom = () => {
    if (container) return;
    container = document.createElement("div");
    container.id = "nuxt-debug-panel";
    container.style.cssText = [
      "position:fixed",
      "top:0",
      "right:0",
      "z-index:99999",
      "font:12px/1.4 ui-monospace,monospace",
      "background:rgba(18,18,20,0.92)",
      "color:#eee",
      "padding:8px 8px 10px",
      "max-width:340px",
      "max-height:60vh",
      "overflow:auto",
      "border:1px solid #444",
      "border-bottom-left-radius:6px",
      "box-shadow:0 4px 18px -2px rgba(0,0,0,.6)",
    ].join(";");

    const header = document.createElement("div");
    header.style.cssText = "display:flex;align-items:center;gap:6px;margin-bottom:6px;";
    const title = document.createElement("strong");
    title.textContent = "Debug Panel";
    title.style.cssText = "flex:1;cursor:move;";

    const btnRefresh = document.createElement("button");
    btnRefresh.textContent = "⟳";
    btnRefresh.title = "Force refresh";
    btnRefresh.style.cssText = "background:#333;border:1px solid #555;color:#ddd;padding:2px 6px;cursor:pointer;border-radius:3px;";
    btnRefresh.onclick = () => refreshNow(true);

    const btnHide = document.createElement("button");
    btnHide.textContent = "×";
    btnHide.title = "Close (Ctrl+Shift+D)";
    btnHide.style.cssText = btnRefresh.style.cssText;
    btnHide.onclick = () => api.hide();

    header.appendChild(title);
    header.appendChild(btnRefresh);
    header.appendChild(btnHide);
    container.appendChild(header);

    actionsContainer = document.createElement("div");
    actionsContainer.style.cssText = "display:flex;gap:4px;margin-bottom:6px;flex-wrap:wrap;";
    container.appendChild(actionsContainer);

    const table = document.createElement("table");
    table.style.cssText = "width:100%;border-collapse:collapse;";
    const thead = document.createElement("thead");
    thead.innerHTML = "<tr><th style=\"text-align:left;border-bottom:1px solid #444;padding:2px 4px;\">Name</th><th style=\"text-align:left;border-bottom:1px solid #444;padding:2px 4px;\">Value</th></tr>";
    tableBody = document.createElement("tbody");
    table.appendChild(thead);
    table.appendChild(tableBody);
    container.appendChild(table);

    // Basic drag (vertical only) so panel can be moved down.
    let dragStartY = 0;
    let startTop = 0;
    let dragging = false;
    title.addEventListener("mousedown", (e) => {
      dragging = true;
      dragStartY = e.clientY;
      startTop = container!.offsetTop;
      e.preventDefault();
    });
    window.addEventListener("mousemove", (e) => {
      if (!dragging) return;
      const delta = e.clientY - dragStartY;
			container!.style.top = Math.max(0, startTop + delta) + "px";
    });
    window.addEventListener("mouseup", () => {
      dragging = false;
    });

    document.body.appendChild(container);
  };

  const renderActions = () => {
    if (!actionsContainer) return;
    actionsContainer.innerHTML = "";
    for (const actionItem of actions) {
      const btn = document.createElement("button");
      btn.textContent = actionItem.label;
      if (actionItem.title) btn.title = actionItem.title;
      btn.style.cssText = "background:#2a5a8a;border:1px solid #3a7ac0;color:#fff;padding:3px 8px;cursor:pointer;border-radius:3px;font-size:11px;";
      btn.onclick = () => {
        try {
          actionItem.action();
        } catch (e: any) {
          console.error("Debug panel action error:", e);
        }
      };
      actionsContainer.appendChild(btn);
    }
  };

  const render = () => {
    if (!visible || !tableBody) return;
    const rows: string[] = [];
    for (const item of items) {
      let val: any;
      try {
        val = item.getter();
      } catch (e: any) {
        val = `!ERR: ${e?.message || e}`;
      }
      item.last = val;
      const display = formatVal(val);
      rows.push(`<tr><td style="vertical-align:top;padding:2px 4px;border-bottom:1px solid #333;white-space:nowrap;">${escapeHtml(item.name)}</td><td style="padding:2px 4px;border-bottom:1px solid #333;max-width:240px;word-break:break-word;">${display}</td></tr>`);
    }
    tableBody.innerHTML = rows.join("");
  };

  const refreshNow = (force?: boolean) => {
    if (!visible && !force) return;
    render();
  };

  const startTimer = () => {
    if (timer != null) return;
    timer = window.setInterval(refreshNow, REFRESH_INTERVAL_MS);
  };
  const stopTimer = () => {
    if (timer != null) {
      clearInterval(timer);
      timer = null;
    }
  };

  const formatVal = (v: any): string => {
    if (v === null) return "<em style=\"opacity:.6\">null</em>";
    if (v === undefined) return "<em style=\"opacity:.6\">undefined</em>";
    if (typeof v === "string") {
      const trimmed = v.length > 180 ? v.slice(0, 177) + "…" : v;
      return `<span style="color:#9cdcfe">"${escapeHtml(trimmed)}"</span>`;
    }
    if (typeof v === "number") return `<span style="color:#b5cea8">${v}</span>`;
    if (typeof v === "boolean") return `<span style="color:#569cd6">${v}</span>`;
    if (v instanceof Date) return `<span>${v.toISOString()}</span>`;
    if (v instanceof Map) {
      const obj: Record<string, any> = {};
      v.forEach((val, key) => { obj[String(key)] = val; });
      v = obj;
    }
    try {
      return `<code>${escapeHtml(JSON.stringify(v, (_, val) => typeof val === "bigint" ? val.toString() + "n" : val, 2))}</code>`;
    } catch {
      return `<code>${escapeHtml(String(v))}</code>`;
    }
  };

  const escapeHtml = (s: string) =>
    s
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");

  const api: DebugPanelApi = {
    add(name, getter) {
      if (items.find(i => i.name === name)) {
        const idx = items.findIndex(i => i.name === name);
        items[idx] = { name, getter };
      } else {
        items.push({ name, getter });
      }
      if (visible) refreshNow(true);
    },
    track(name, source) {
      if (isRefLike(source)) {
        this.add(name, () => (source as any).value);
      } else {
        this.add(name, () => source);
      }
    },
    remove(name) {
      const i = items.findIndex(it => it.name === name);
      if (i >= 0) items.splice(i, 1);
      if (visible) refreshNow(true);
    },
    clear() {
      items.splice(0, items.length);
      if (visible) refreshNow(true);
    },
    list() {
      return items.map(i => i.name);
    },
    addAction(label, action, title) {
      if (actions.find(a => a.label === label)) {
        const idx = actions.findIndex(a => a.label === label);
        actions[idx] = { label, action, title };
      } else {
        actions.push({ label, action, title });
      }
      if (visible) renderActions();
    },
    removeAction(label) {
      const i = actions.findIndex(a => a.label === label);
      if (i >= 0) actions.splice(i, 1);
      if (visible) renderActions();
    },
    show() {
      if (visible) return;
      ensureDom();
      visible = true;
			container!.style.display = "block";
			renderActions();
			startTimer();
			refreshNow(true);
    },
    hide() {
      if (!visible) return;
      visible = false;
      if (container) container.style.display = "none";
      stopTimer();
    },
    toggle() {
      if (visible) {
        this.hide();
      } else {
        this.show();
      }
    },
    isVisible() {
      return visible;
    }
  };

  return { api };
}

function isRefLike(obj: any): obj is Ref<any> | ComputedRef<any> {
  return obj && typeof obj === "object" && "value" in obj;
}

export default defineNuxtPlugin((nuxtApp) => {
  const { api } = createPanel();
  nuxtApp.provide("debugPanel", api);

  // Default variables
  api.add("time", () => new Date().toLocaleTimeString());
  api.add("route", () => ({ path: useRoute().path, name: useRoute().name }));

  const listener = (e: KeyboardEvent) => {
    if (e.code === KEYBIND.code && (!!KEYBIND.ctrl === e.ctrlKey) && (!!KEYBIND.shift === e.shiftKey)) {
      api.toggle();
      e.preventDefault();
    }
  };

  window.addEventListener("keydown", listener);

  if ((import.meta as any).hot) {
    (import.meta as any).hot.on("nuxt:before-hmr", () => {
      api.hide();
      api.clear();
      window.removeEventListener("keydown", listener);
    });
  }
});

// Provide type-safe accessor `$debugPanel`
declare module "#app" {
	// Nuxt 3 injection augmentation
	interface NuxtApp { $debugPanel: DebugPanelApi }
}
declare module "vue" {
	interface ComponentCustomProperties { $debugPanel: DebugPanelApi }
}

export {}; // ensure module scope
