export interface AppendixFontBytes {
  regular: ArrayBuffer;
  bold: ArrayBuffer;
}

/** Fetches the appendix body font (Source Serif 4) from the app's static assets. */
export async function loadAppendixFonts(): Promise<AppendixFontBytes> {
  const [regular, bold] = await Promise.all([
    fetch("/fonts/SourceSerif4-Regular.otf").then((r) => r.arrayBuffer()),
    fetch("/fonts/SourceSerif4-Bold.otf").then((r) => r.arrayBuffer()),
  ]);
  return { regular, bold };
}
