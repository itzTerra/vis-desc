/**
 * Greedy word-wrap: appends words to the current line until measureWidth
 * would exceed maxWidth, then starts a new line. An unbreakable overlong
 * word is placed on its own line rather than truncated.
 */
export function wrapText(
  text: string,
  maxWidth: number,
  measureWidth: (s: string) => number,
): string[] {
  const words = text.split(/\s+/).filter(Boolean);
  const lines: string[] = [];
  let current = "";

  for (const word of words) {
    const candidate = current ? `${current} ${word}` : word;
    if (!current || measureWidth(candidate) <= maxWidth) {
      current = candidate;
    } else {
      lines.push(current);
      current = word;
    }
  }

  if (current) lines.push(current);
  return lines;
}
