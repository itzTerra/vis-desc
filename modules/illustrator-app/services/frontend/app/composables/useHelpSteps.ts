import type { Step } from "~/components/HelpOverlay.vue";
import type { Highlight } from "~/types/common";

export function useHelpSteps(opts: {
  isLoading: Ref<number | null>;
  currentStage: Ref<string | undefined>;
  highlights: Highlight[];
  pdfViewer: Ref<any>;
  cacheManagerRef: Ref<any>;
}) {
  const { isLoading, currentStage, highlights, pdfViewer, cacheManagerRef } = opts;
  const runtimeConfig = useRuntimeConfig();
  const seenHelpOnce = useLocalStorage("seenHelpOnce", false);
  const showHelp = ref(false);
  const highlightsHelpBackup: Highlight[] = [];
  const helpPdfUrl = `${runtimeConfig.app.baseURL}sample.pdf`;

  function toggleHelp(toValue?: boolean) {
    showHelp.value = toValue !== undefined ? toValue : !showHelp.value;
    if (showHelp.value) {
      highlightsHelpBackup.splice(0, highlightsHelpBackup.length, ...highlights);
      highlights.splice(
        0,
        highlights.length,
        {
          id: 1,
          text: "This is a scored segment.",
          score: 0.98,
          score_received_at: Date.now(),
          polygons: [[[0.11, 0.18], [0.83, 0.18], [0.83, 0.4], [0.11, 0.4]]],
        },
        {
          id: 2,
          text: "This is an unscored segment.",
          score: undefined,
          score_received_at: Date.now(),
          polygons: [[[0.11, 0.6], [0.89, 0.6], [0.89, 0.85], [0.11, 0.85]]],
        }
      );
      if (!seenHelpOnce.value) {
        seenHelpOnce.value = true;
      }
    } else {
      highlights.splice(0, highlights.length, ...highlightsHelpBackup);
    }
  }

  const helpSteps: Step[] = [
    // > Open model menu
    {
      selector: "[data-help-target=\"model\"]",
      title: "Model Selection",
      message: "Select the AI model to use for scoring text segments.",
      position: "bottom",
      onEnter: () => {
        const el = document.querySelector("[data-help-target=\"model\"] [role=\"button\"]") as HTMLElement;
        el?.focus();
      },
      onLeave: () => {
        (document.activeElement as HTMLElement)?.blur();
      },
    },
    // > Show the cache manager button
    {
      selector: "[data-help-target=\"cache\"]",
      title: "Model Cache Manager",
      message: "Click 'Manage Downloads' to open the cache manager where you can download AI models for local, browser-based inference — no data leaves your machine during scoring.",
      position: "top",
    },
    // > Open the cache manager dialog (non-modal so the overlay stays on top).
    // The box-shadow highlight reads .modal-box's border-radius automatically.
    {
      selector: "[data-help-target=\"cache\"] .modal-box",
      title: "GPU Acceleration & Downloads",
      message: "Each model has a GPU toggle. Enable it to use WebGPU for faster scoring on devices with a strong GPU. Models are downloaded once and cached — use the Delete button to free up space later.",
      position: "top",
      // transform: { y: -259 },
      onEnter: () => { cacheManagerRef.value?.openNonModal(); },
      onLeave: () => { cacheManagerRef.value?.closeNonModal(); },
    },
    {
      selector: "[data-help-target=\"file\"]",
      title: "File Upload",
      message: "Upload your PDF document here by clicking or dragging.",
      position: "bottom",
    },
    // > Simulate processing
    {
      selector: "[data-help-target=\"progress\"]",
      title: "Scoring Progress",
      message: "Monitor the progress of text segment scoring here.",
      position: "bottom",
      onEnter: () => {
        currentStage.value = "Scoring...";
        isLoading.value = Date.now() - 120000; // Simulate 2 minutes of loading time
      },
      onLeave: () => {
        currentStage.value = undefined;
        isLoading.value = null;
      },
    },
    // > Open the nav panel
    {
      selector: "[data-help-target=\"nav\"]",
      title: "Segment Navigation",
      message: "Navigate through scored segments using this panel. Filter selected segments only or set a minimum and/or maximum score. Sort segments by score or index. Click a segment to jump to its location in the document.",
      position: "bottom",
      onEnter: () => {
        const navEl = document.querySelector("[data-help-target=\"nav\"] summary") as HTMLElement;
        navEl?.click();
      },
      onLeave: () => {
        const navEl = document.querySelector("[data-help-target=\"nav\"] summary") as HTMLElement;
        if (navEl?.parentElement) {
          (navEl.parentElement as HTMLDetailsElement).open = false;
        }
      },
    },
    // > Show a static PDF page with highlights; hover the first segment
    {
      selector: "[data-help-target=\"viewer\"]",
      title: "Document Viewer",
      message: "View your PDF with controls at the bottom and a score heatmap on the left. Hover over a segment to see its score. Left-click a segment to select it for filtering. Hover over a segment and click 'Illustrate' in the tooltip to create its Image Editor panel to the right.",
      position: "right",
      onEnter: () => {
        const firstHighlightEl = document.querySelector(".highlight-dropdown") as HTMLElement;
        firstHighlightEl?.classList.add("dropdown-open");
      },
      onLeave: () => {
        const firstHighlightEl = document.querySelector(".highlight-dropdown") as HTMLElement;
        firstHighlightEl?.classList.remove("dropdown-open");
      },
    },
    // > Open an ImageEditor for the first help highlight
    {
      selector: "[data-help-target=\"images\"]",
      title: "Image Editor",
      message: "Expand the editor to edit the prompt, use Enhance to refine it with AI, then Generate to create an illustration. Use the history arrows to browse previous prompts and images.",
      position: "left",
      onEnter: () => {
        pdfViewer.value?.openImageEditors?.([1]);
      },
      onLeave: () => {
        pdfViewer.value?.closeImageEditors?.([1]);
      },
    },
    // > Open the auto-illustration settings dropdown
    {
      selector: "[data-help-target=\"auto-illustration\"]",
      title: "Auto-Illustration",
      message: "Automatically select high-scoring segments and open their Image Editors on a schedule. Configure the min/max page gap between illustrations and an optional minimum score threshold. The progress bar shows selected (blue), enhanced (green), and generated (orange) counts. Hover over the bar for exact numbers.",
      position: "bottom",
      onEnter: () => {
        const dropdown = document.querySelector("[data-help-target=\"auto-illustration\"] .dropdown") as HTMLElement;
        dropdown?.classList.add("dropdown-open");
      },
      onLeave: () => {
        const dropdown = document.querySelector("[data-help-target=\"auto-illustration\"] .dropdown") as HTMLElement;
        dropdown?.classList.remove("dropdown-open");
      },
    },
    {
      selector: "[data-help-target=\"export\"]",
      title: "Export as HTML",
      message: "Export your PDF and generated images as a single standalone HTML file. The export includes all pages lazy-loaded for performance and a gallery of generated images. The file requires an internet connection to load the PDF viewer library.",
      position: "bottom",
    },
  ];

  return { showHelp, seenHelpOnce, helpSteps, toggleHelp, helpPdfUrl };
}
