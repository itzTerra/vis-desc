/** Decodes the base64 payload of a data URI into raw bytes. */
export function dataUriToBytes(dataUri: string): Uint8Array {
  const commaIndex = dataUri.indexOf(",");
  const base64 = commaIndex >= 0 ? dataUri.slice(commaIndex + 1) : dataUri;
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}

/**
 * Re-encodes any browser-decodable image data URI (PNG today, from
 * ImageEditor's applyGenerateResult) to JPEG bytes suitable for
 * PDFDocument#embedJpg. Requires a browser environment (Image + canvas);
 * not covered by node:test — see Task 12's manual QA pass.
 */
export async function reencodeToJpeg(dataUri: string, quality = 0.85): Promise<Uint8Array> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        reject(new Error("Canvas 2D context unavailable"));
        return;
      }
      ctx.drawImage(img, 0, 0);
      resolve(dataUriToBytes(canvas.toDataURL("image/jpeg", quality)));
    };
    img.onerror = () => reject(new Error("Failed to load image for re-encoding"));
    img.src = dataUri;
  });
}
