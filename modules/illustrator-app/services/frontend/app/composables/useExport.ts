import { ref } from "vue";
import type { Highlight } from "~/types/common";
import {
  createExportSnapshot,
  generateExportHtml,
  downloadExport,
} from "~/utils/export";

interface ExportResult {
  html: string;
}

export function useExport() {
  const showExportDialog = ref(false);
  const isExporting = ref(false);

  async function exportPdf(
    pdfFile: File | null,
    highlights: Highlight[],
    imageBlobs: Record<number, Blob>,
    pageCount: number
  ): Promise<ExportResult> {
    if (!pdfFile) {
      throw new Error("PDF file is required for export");
    }

    isExporting.value = true;

    try {
      const snapshot = await createExportSnapshot(
        pdfFile,
        highlights,
        imageBlobs,
        pageCount
      );

      const html = generateExportHtml(snapshot);

      return { html };
    } finally {
      isExporting.value = false;
    }
  }

  async function confirmExport(
    pdfFile: File | null,
    highlights: Highlight[],
    imageBlobs: Record<number, Blob>,
    pageCount: number,
    filename: string
  ): Promise<void> {
    const result = await exportPdf(
      pdfFile,
      highlights,
      imageBlobs,
      pageCount
    );
    await downloadExport(result.html, filename);
  }

  return {
    showExportDialog,
    isExporting,
    exportPdf,
    confirmExport,
  };
}
