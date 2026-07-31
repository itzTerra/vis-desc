export interface AppendixFontBytes {
  regular: ArrayBuffer;
  bold: ArrayBuffer;
}

/** Fetches the appendix body font (Source Serif 4) from the app's static assets. */
export async function loadAppendixFonts(): Promise<AppendixFontBytes> {
  const base = useRuntimeConfig().app.baseURL.replace(/\/$/, "");
  const [regular, bold] = await Promise.all([
    fetch(`${base}/fonts/SourceSerif4-Regular.otf`).then((r) => r.arrayBuffer()),
    fetch(`${base}/fonts/SourceSerif4-Bold.otf`).then((r) => r.arrayBuffer()),
  ]);
  return { regular, bold };
}
