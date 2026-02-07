import type { Highlight } from "~/types/common";

interface ExportHighlight {
  id: number;
  text: string;
  imageUrl?: string;
  polygons: Record<number, number[][]>;
}

interface ExportSnapshot {
  pdfArrayBuffer: ArrayBuffer;
  imageDataUrls: Record<number, string>;
  highlights: ExportHighlight[];
  pageCount: number;
}

export async function blobToDataUrl(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

export async function createExportSnapshot(
  pdfFile: File | null,
  highlights: Highlight[],
  imageBlobs: Record<number, Blob>,
  pageCount: number
): Promise<ExportSnapshot> {
  if (!pdfFile) {
    throw new Error("PDF file is required for export");
  }

  const pdfArrayBuffer = await pdfFile.arrayBuffer();

  const imageDataUrls: Record<number, string> = {};
  for (const [highlightId, blob] of Object.entries(imageBlobs)) {
    imageDataUrls[highlightId] = await blobToDataUrl(blob);
  }

  const exportHighlights: ExportHighlight[] = highlights.map((h) => ({
    id: h.id,
    text: h.text,
    imageUrl: imageDataUrls[h.id],
    polygons: h.polygons,
  }));

  return {
    pdfArrayBuffer,
    imageDataUrls,
    highlights: exportHighlights,
    pageCount,
  };
}

export function generateExportHtml(snapshot: ExportSnapshot): string {
  const pdfArrayBuffer = snapshot.pdfArrayBuffer;
  const pdfBase64 = btoa(
    String.fromCharCode.apply(null, Array.from(new Uint8Array(pdfArrayBuffer)))
  );

  const html = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PDF Export</title>
  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }

    body {
      font-family: system-ui, -apple-system, sans-serif;
      background: #f5f5f5;
      color: #333;
    }

    #container {
      display: flex;
      height: 100vh;
      gap: 1px;
      background: #ddd;
    }

    #pdf-viewer {
      flex: 1;
      overflow-y: auto;
      background: #fff;
      padding: 20px;
      position: relative;
    }

    .page-wrapper {
      margin-bottom: 20px;
      background: #fff;
      box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }

    canvas {
      display: block;
      width: 100%;
      height: auto;
    }

    .page-placeholder {
      width: 100%;
      height: 600px;
      background: linear-gradient(135deg, #f0f0f0 25%, transparent 25%, transparent 75%, #f0f0f0 75%, #f0f0f0),
                  linear-gradient(135deg, #f0f0f0 25%, transparent 25%, transparent 75%, #f0f0f0 75%, #f0f0f0);
      background-size: 20px 20px;
      background-position: 0 0, 10px 10px;
      background-color: #fff;
    }

    #image-buttons {
      width: 160px;
      overflow-y: auto;
      background: #fafafa;
      padding: 10px;
      border-left: 1px solid #ddd;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    .image-button {
      display: block;
      width: 100%;
      aspect-ratio: 1 / 1;
      border: 2px solid #ddd;
      border-radius: 4px;
      cursor: pointer;
      overflow: hidden;
      background: #fff;
      transition: border-color 0.2s;
    }

    .image-button:hover {
      border-color: #0066cc;
    }

    .image-button img {
      width: 100%;
      height: 100%;
      object-fit: cover;
    }

    /* Modal Overlay */
    .modal-overlay {
      display: none;
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: rgba(0, 0, 0, 0.7);
      z-index: 1000;
      align-items: center;
      justify-content: center;
    }

    .modal-overlay.active {
      display: flex;
    }

    .modal-content {
      background: #fff;
      border-radius: 8px;
      padding: 24px;
      max-width: 90vw;
      max-height: 90vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
    }

    .modal-content img {
      max-width: 512px;
      max-height: 512px;
      width: 100%;
      height: auto;
      object-fit: contain;
    }

    .modal-close {
      position: absolute;
      top: 16px;
      right: 16px;
      width: 32px;
      height: 32px;
      border: none;
      background: rgba(0, 0, 0, 0.6);
      color: #fff;
      border-radius: 4px;
      cursor: pointer;
      font-size: 20px;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: background 0.2s;
    }

    .modal-close:hover {
      background: rgba(0, 0, 0, 0.8);
    }

    @media (max-width: 768px) {
      #container {
        flex-direction: column;
      }

      #image-buttons {
        width: 100%;
        height: 150px;
        flex-direction: row;
        border-left: none;
        border-top: 1px solid #ddd;
      }

      .image-button {
        width: 140px;
        height: 140px;
        flex-shrink: 0;
      }

      .modal-content img {
        max-width: 100%;
        max-height: 100%;
      }
    }
  </style>
</head>
<body>
  <div id="container">
    <div id="pdf-viewer"></div>
    <div id="image-buttons"></div>
  </div>

  <div class="modal-overlay" id="imageModal">
    <button class="modal-close" onclick="closeImageModal()">✕</button>
    <div class="modal-content">
      <img id="modalImage" src="" alt="Image preview">
    </div>
  </div>

  <script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.11.0/pdf.min.js"></script>
  <script>
    const PRELOAD_PAGES = 3;
    const PAGE_THRESHOLDS = [0, 0.1, 0.25, 0.5, 0.75, 1.0];

    const pdf = \`data:application/pdf;base64,${pdfBase64}\`;
    const highlights = ${JSON.stringify(snapshot.highlights)};

    let pdfDoc = null;
    let pageRendering = {};
    let pageQueue = new Set();
    let renderTimeout = null;

    async function initPdf() {
      pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.11.0/pdf.worker.min.js';

      try {
        pdfDoc = await pdfjsLib.getDocument(pdf).promise;
        const pdfViewer = document.getElementById('pdf-viewer');

        for (let i = 1; i <= pdfDoc.numPages; i++) {
          const pageWrapper = document.createElement('div');
          pageWrapper.className = 'page-wrapper';
          pageWrapper.setAttribute('data-page', i);

          const placeholder = document.createElement('div');
          placeholder.className = 'page-placeholder';
          pageWrapper.appendChild(placeholder);

          pdfViewer.appendChild(pageWrapper);
        }

        setupLazyLoading();
        renderVisiblePages();
      } catch (error) {
        console.error('Failed to load PDF:', error);
        document.getElementById('pdf-viewer').innerHTML = '<p>Failed to load PDF</p>';
      }
    }

    function setupLazyLoading() {
      const pdfViewer = document.getElementById('pdf-viewer');
      const pageWrappers = pdfViewer.querySelectorAll('.page-wrapper');

      const observer = new IntersectionObserver(
        (entries) => {
          for (const entry of entries) {
            const pageNum = parseInt(entry.target.getAttribute('data-page'), 10);
            if (entry.isIntersecting) {
              pageQueue.add(pageNum);
            }
          }
          clearTimeout(renderTimeout);
          renderTimeout = setTimeout(renderVisiblePages, 100);
        },
        { threshold: PAGE_THRESHOLDS }
      );

      pageWrappers.forEach((wrapper) => observer.observe(wrapper));

      pdfViewer.addEventListener('scroll', () => {
        const scrollTop = pdfViewer.scrollTop;
        const scrollHeight = pdfViewer.scrollHeight;
        const clientHeight = pdfViewer.clientHeight;

        for (let i = 1; i <= pdfDoc.numPages; i++) {
          if (Math.abs(i - Math.max(1, Math.ceil(scrollTop / 600))) <= PRELOAD_PAGES) {
            pageQueue.add(i);
          }
        }

        clearTimeout(renderTimeout);
        renderTimeout = setTimeout(renderVisiblePages, 100);
      });
    }

    async function renderVisiblePages() {
      const pagesToRender = Array.from(pageQueue)
        .filter((pageNum) => !pageRendering[pageNum])
        .sort((a, b) => a - b);

      for (const pageNum of pagesToRender) {
        if (!pageRendering[pageNum]) {
          pageRendering[pageNum] = true;
          await renderPage(pageNum);
        }
      }

      pageQueue.forEach((pageNum) => {
        if (pageRendering[pageNum]) {
          pageQueue.delete(pageNum);
        }
      });
    }

    async function renderPage(pageNum) {
      try {
        const page = await pdfDoc.getPage(pageNum);
        const viewport = page.getViewport({ scale: 1.5 });

        const canvas = document.createElement('canvas');
        canvas.width = viewport.width;
        canvas.height = viewport.height;

        const context = canvas.getContext('2d');
        await page.render({
          canvasContext: context,
          viewport: viewport
        }).promise;

        const pageWrapper = document.querySelector(\`[data-page="\${pageNum}"]\`);
        if (pageWrapper) {
          pageWrapper.innerHTML = '';
          pageWrapper.appendChild(canvas);
        }
      } catch (error) {
        console.error(\`Failed to render page \${pageNum}:\`, error);
      }
    }

    function initImageButtons() {
      const imageButtons = document.getElementById('image-buttons');

      if (highlights.length === 0) {
        imageButtons.innerHTML = '<p style="color: #999; font-size: 12px; padding: 10px;">No images</p>';
        return;
      }

      const highlightsWithImages = highlights.filter(h => h.imageUrl);

      if (highlightsWithImages.length === 0) {
        imageButtons.innerHTML = '<p style="color: #999; font-size: 12px; padding: 10px;">No images</p>';
        return;
      }

      highlightsWithImages.forEach((highlight) => {
        const button = document.createElement('button');
        button.className = 'image-button';
        button.title = highlight.text.substring(0, 50);
        button.onclick = () => openImageModal(highlight.imageUrl);

        const img = document.createElement('img');
        img.src = highlight.imageUrl;
        img.alt = highlight.text;

        button.appendChild(img);
        imageButtons.appendChild(button);
      });
    }

    function openImageModal(imageUrl) {
      const modal = document.getElementById('imageModal');
      const modalImage = document.getElementById('modalImage');
      modalImage.src = imageUrl;
      modal.classList.add('active');
    }

    function closeImageModal() {
      const modal = document.getElementById('imageModal');
      modal.classList.remove('active');
    }

    document.getElementById('imageModal').addEventListener('click', (e) => {
      if (e.target.id === 'imageModal') {
        closeImageModal();
      }
    });

    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        closeImageModal();
      }
    });

    initPdf();
    initImageButtons();
  </script>
</body>
</html>`;

  return html;
}

export async function downloadExport(
  html: string,
  filename: string = "export.html"
): Promise<void> {
  const blob = new Blob([html], { type: "text/html" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}
