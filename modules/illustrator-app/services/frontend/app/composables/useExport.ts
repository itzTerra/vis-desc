import { ref } from "vue";
import type { ExportImageData, Highlight } from "~/types/common";
import { buildExportPdf } from "~/utils/pdfExport/buildExportPdf";
import { downloadPdf } from "~/utils/pdfExport/download";

interface ExportResult {
  pdfBytes: Uint8Array;
}

export function useExport() {
  const showExportDialog = ref(false);
  const isExporting = ref(false);

  async function exportPdf(
    pdfFile: File | null,
    highlights: Highlight[],
    images: Record<number, ExportImageData>
  ): Promise<ExportResult> {
    if (!pdfFile) {
      throw new Error("PDF file is required for export");
    }

    isExporting.value = true;

    try {
      const pdfBytes = await pdfFile.arrayBuffer();
      const resultBytes = await buildExportPdf({ pdfBytes, highlights, images });
      return { pdfBytes: resultBytes };
    } finally {
      isExporting.value = false;
    }
  }

  async function confirmExport(
    pdfFile: File | null,
    highlights: Highlight[],
    images: Record<number, ExportImageData>,
    filename: string
  ): Promise<void> {
    const result = await exportPdf(pdfFile, highlights, images);
    const finalFilename = filename.toLowerCase().endsWith(".pdf") ? filename : `${filename}.pdf`;
    await downloadPdf(result.pdfBytes, finalFilename);
  }

  return {
    showExportDialog,
    isExporting,
    exportPdf,
    confirmExport,
  };
}
