/**
 * Trie-based data structure for efficient multi-word expression matching.
 * Supports both original text and lemma lookups with longest match preference.
 */
export class MultiWordExpressionTrie {
  private root: TrieNode = {};

  constructor() {
    this.root = {};
  }

  /**
   * Add a multi-word expression to the trie.
   * @param expression - Whitespace-separated words
   * @param score - Associated score/value
   * @param nWords - Number of words in expression
   */
  addExpression(expression: string, score: number, nWords: number): void {
    const words = expression.split(/\s+/);
    this._addToTrie(this.root, words, score, nWords);
  }

  /**
   * Recursively add words to the trie structure.
   */
  private _addToTrie(
    node: TrieNode,
    words: string[],
    score: number,
    nWords: number
  ): void {
    if (words.length === 0) {
      node.$score = score;
      node.$nWords = nWords;
      return;
    }

    const [firstWord, ...rest] = words;
    if (!(firstWord in node)) {
      node[firstWord] = {};
    }
    this._addToTrie(node[firstWord] as TrieNode, rest, score, nWords);
  }

  /**
   * Find all longest matches in the token list.
   * Returns list of [startIdx, endIdx, score] tuples.
   */
  findLongestMatches(
    tokens: Array<{ text: string; lemma?: string }>
  ): Array<[number, number, number]> {
    const matches: Array<[number, number, number]> = [];
    const covered = new Set<number>();

    for (let i = 0; i < tokens.length; i++) {
      if (covered.has(i)) continue;

      const match = this._findMatchFromPosition(tokens, i);
      if (match) {
        const [endIdx, score] = match;
        // Mark all tokens in this match as covered
        for (let j = i; j <= endIdx; j++) {
          covered.add(j);
        }
        matches.push([i, endIdx, score]);
      }
    }

    return matches;
  }

  /**
   * Try to find a match starting at the given position.
   * Returns [endIdx, score] if match found, null otherwise.
   */
  private _findMatchFromPosition(
    tokens: Array<{ text: string; lemma?: string }>,
    startIdx: number
  ): [number, number] | null {
    let currentNode = this.root;
    let lastMatch: [number, number] | null = null;
    let idx = startIdx;

    while (idx < tokens.length) {
      const token = tokens[idx];
      const text = token.text.toLowerCase();
      const lemma = (token.lemma || text).toLowerCase();

      // Try both text and lemma
      let nextNode: TrieNode | undefined;
      if (text in currentNode) {
        nextNode = currentNode[text] as TrieNode | undefined;
      } else if (lemma in currentNode) {
        nextNode = currentNode[lemma] as TrieNode | undefined;
      }

      if (!nextNode) break;

      currentNode = nextNode;
      idx++;

      // Check if this position is a valid match
      if ("$score" in currentNode && currentNode.$score !== undefined) {
        lastMatch = [idx - 1, currentNode.$score];
      }
    }

    return lastMatch;
  }
}

interface TrieNode {
  [key: string]: unknown;
  $score?: number;
  $nWords?: number;
}
