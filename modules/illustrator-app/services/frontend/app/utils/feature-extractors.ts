import type {
  TokenData,
  EntityData,
  SentenceData,
  BookNLPContext,
} from "~/types/text2features-worker";

export const FEATURE_COUNT = 3671;

export const EXTRACTOR_FEATURE_COUNTS: Record<string, number> = {
  extract_quote_ratio: 1,
  extract_char_ngrams: 1000,
  extract_word_length_by_char: 15,
  extract_ngram_word_length_by_char: 15,
  extract_sentence_length_by_word: 26,
  extract_sentence_length_by_word_avg: 1,
  extract_numeric_word_ratio: 1,
  extract_ttr: 1,
  extract_lexical_density: 1,
  extract_syllable_count_avg: 1,
  extract_3plus_syllable_count_ratio: 1,
  extract_stopwords: 1,
  extract_articles: 2,
  extract_punctuation: 16,
  extract_contractions: 1,
  extract_casing: 9,
  extract_casing_bigrams: 9,
  extract_pos_frequency: 50,
  extract_pos_ngrams: 995,
  extract_dependency_tree_structure: 92,
  extract_dependency_tree_relations: 1319,
  extract_noun_phrase_lengths: 14,
  extract_entity_categories: 6,
  extract_events: 1,
  extract_supersense: 41,
  extract_tense: 4,
  extract_polysemy: 15,
  extract_word_concreteness: 20,
  extract_word_concreteness_avg: 1,
  extract_preposition_imageability: 10,
  extract_preposition_imageability_avg: 1,
  extract_places: 1,
};

const PREPROCESS_TRANSLATION_TABLE: Record<string, string | null> = {
  "\r": null,
  "'": "'",
  "\u201c": "\"",
  "\u201d": "\"",
  "？": "?",
  "！": "!",
};

const CONTENT_UD_POS_TAGS = new Set(["NOUN", "PROPN", "VERB", "ADJ", "ADV"]);

const PUNCTUATION_SYMBOL_MAP: Record<string, number> = {
  ".": 0,
  ",": 1,
  ":": 2,
  ";": 3,
  "!": 4,
  "?": 5,
  "\"": 6,
  "'": 7,
  "(": 8,
  ")": 9,
  "-": 10,
  "–": 11,
  "—": 12,
  "…": 13,
  "/": 14,
  "\\": 15,
};

const ENGLISH_STOPWORDS = new Set([
  "a",
  "about",
  "above",
  "after",
  "again",
  "against",
  "all",
  "am",
  "an",
  "and",
  "any",
  "are",
  "aren't",
  "as",
  "at",
  "be",
  "because",
  "been",
  "before",
  "being",
  "below",
  "between",
  "both",
  "but",
  "by",
  "can't",
  "cannot",
  "could",
  "couldn't",
  "did",
  "didn't",
  "do",
  "does",
  "doesn't",
  "doing",
  "don't",
  "down",
  "during",
  "each",
  "few",
  "for",
  "from",
  "further",
  "had",
  "hadn't",
  "has",
  "hasn't",
  "have",
  "haven't",
  "having",
  "he",
  "he'd",
  "he'll",
  "he's",
  "her",
  "here",
  "here's",
  "hers",
  "herself",
  "him",
  "himself",
  "his",
  "how",
  "how's",
  "i",
  "i'd",
  "i'll",
  "i'm",
  "i've",
  "if",
  "in",
  "into",
  "is",
  "isn't",
  "it",
  "it's",
  "its",
  "itself",
  "just",
  "me",
  "my",
  "myself",
  "no",
  "nor",
  "not",
  "of",
  "off",
  "on",
  "once",
  "only",
  "or",
  "other",
  "ought",
  "our",
  "ours",
  "ourselves",
  "out",
  "over",
  "own",
  "same",
  "shan't",
  "she",
  "she'd",
  "she'll",
  "she's",
  "should",
  "shouldn't",
  "so",
  "some",
  "such",
  "than",
  "that",
  "that's",
  "the",
  "their",
  "theirs",
  "them",
  "themselves",
  "then",
  "there",
  "there's",
  "these",
  "they",
  "they'd",
  "they'll",
  "they're",
  "they've",
  "this",
  "those",
  "through",
  "to",
  "too",
  "under",
  "until",
  "up",
  "very",
  "was",
  "wasn't",
  "we",
  "we'd",
  "we'll",
  "we're",
  "we've",
  "were",
  "weren't",
  "what",
  "what's",
  "when",
  "when's",
  "where",
  "where's",
  "which",
  "while",
  "who",
  "who's",
  "whom",
  "why",
  "why's",
  "with",
  "won't",
  "would",
  "wouldn't",
  "you",
  "you'd",
  "you'll",
  "you're",
  "you've",
  "your",
  "yours",
  "yourself",
  "yourselves",
]);

export function preprocess(text: string): string {
  let result = text;
  for (const [char, replacement] of Object.entries(PREPROCESS_TRANSLATION_TABLE)) {
    if (replacement === null) {
      result = result.replaceAll(char, "");
    } else {
      result = result.replaceAll(char, replacement);
    }
  }
  return result;
}

/**
 * Count characters between properly closed quotes only.
 * Time complexity: O(n)
 */
export function extractQuoteRatio(text: string): number {
  if (!text) return 0;

  const totalChars = text.length;
  let quotedChars = 0;
  let inQuote = false;
  let quoteChar: string | null = null;
  let currentSpanChars = 0;

  for (const char of text) {
    if (!inQuote && (char === "\"" || char === "'")) {
      inQuote = true;
      quoteChar = char;
      currentSpanChars = 0;
    } else if (inQuote && char === quoteChar) {
      inQuote = false;
      quotedChars += currentSpanChars;
      currentSpanChars = 0;
      quoteChar = null;
    } else if (inQuote) {
      currentSpanChars++;
    }
  }

  return totalChars > 0 ? quotedChars / totalChars : 0;
}

/**
 * Mock function for getting BookNLP context.
 * Will be replaced with actual implementation using booknlp library.
 */
export async function getBookNLPContext(text: string): Promise<BookNLPContext> {
  // TODO: Replace with actual booknlp integration
  return {
    tokens: [],
    entities: [],
    supersense: [],
    sentences: [],
  };
}

/**
 * Extract word length distribution.
 * Returns 15-dimensional vector where each dimension represents
 * relative frequency of words with that character length.
 */
export function extractWordLengthByChar(words: string[]): number[] {
  const DIM = 15;
  const lengthCounts = new Array(DIM).fill(0);

  if (words.length === 0) {
    return lengthCounts;
  }

  for (const word of words) {
    const len = word.length;
    const binIdx = Math.min(len - 1, DIM - 1);
    lengthCounts[binIdx]++;
  }

  return lengthCounts.map((count) => count / words.length);
}

/**
 * Extract word length by n-grams (3-grams).
 * Returns 15-dimensional vector of normalized lengths.
 */
export function extractNgramWordLengthByChar(words: string[]): number[] {
  const DIM = 15;
  const N = 3;
  const lengthCounts = new Array(DIM).fill(0);

  if (words.length === 0) {
    return lengthCounts;
  }

  const totalNgrams = Math.max(words.length - N + 1, 1);
  for (let i = 0; i < totalNgrams; i++) {
    const ngramLength = words
      .slice(i, i + N)
      .reduce((sum, w) => sum + w.length, 0);
    const binIdx = Math.min(Math.floor(ngramLength / 3), DIM - 1);
    lengthCounts[binIdx]++;
  }

  return lengthCounts.map((count) => count / totalNgrams);
}

/**
 * Extract quote ratio.
 */
export function extractQuoteRatioFeature(text: string): number[] {
  return [extractQuoteRatio(text)];
}

/**
 * Count numeric words in text.
 */
export function extractNumericWordRatio(words: string[]): number {
  if (words.length === 0) return 0;

  const numericCount = words.filter((word) =>
    /^\d+(\.\d+)?$/.test(word)
  ).length;

  return numericCount / words.length;
}

/**
 * Calculate type-token ratio (unique words / total words).
 */
export function extractTTR(words: string[]): number {
  if (words.length === 0) return 0;

  const uniqueWords = new Set(words.map((w) => w.toLowerCase()));
  return uniqueWords.size / words.length;
}

/**
 * Calculate lexical density (simple approximation without POS tagging).
 */
export function extractLexicalDensity(words: string[]): number {
  if (words.length === 0) return 0;

  // Simple heuristic: words longer than 3 chars are content words
  const contentWordCount = words.filter((word) => word.length > 3).length;
  return contentWordCount / words.length;
}

/**
 * Extract syllable count ratios.
 */
export function extractSyllableRatios(words: string[]): number[] {
  if (words.length === 0) {
    return [0, 0];
  }

  const UPPER_SYLLABLE_COUNT = 8;
  let syllableTotal = 0;
  let threeOrMoreCount = 0;

  for (const word of words) {
    const syllables = estimateSyllables(word);
    syllableTotal += syllables;
    if (syllables >= 3) {
      threeOrMoreCount++;
    }
  }

  return [
    Math.min(syllableTotal / words.length / UPPER_SYLLABLE_COUNT, 1.0),
    threeOrMoreCount / words.length,
  ];
}

/**
 * Estimate syllable count in a word.
 */
function estimateSyllables(word: string): number {
  const vowels = (word.match(/[aeiouy]/gi) || []).length;
  let count = 0;

  for (let i = 0; i < word.length; i++) {
    const currentIsVowel = /[aeiouy]/i.test(word[i]);
    const previousIsVowel = i > 0 && /[aeiouy]/i.test(word[i - 1]);

    if (currentIsVowel && !previousIsVowel) {
      count++;
    }
  }

  if (word.endsWith("e")) {
    count--;
  }

  if (word.endsWith("le") && word.length > 2 && !/le$/.test(word.substring(0, word.length - 2))) {
    count++;
  }

  return Math.max(1, count);
}

/**
 * Extract stopword ratio.
 */
export function extractStopwords(words: string[]): number {
  if (words.length === 0) return 0;

  const stopwordCount = words.filter((word) =>
    ENGLISH_STOPWORDS.has(word.toLowerCase())
  ).length;

  return stopwordCount / words.length;
}

/**
 * Extract article ratio (definite and indefinite).
 */
export function extractArticles(words: string[]): number[] {
  if (words.length === 0) {
    return [0, 0];
  }

  let definiteCount = 0;
  let indefiniteCount = 0;

  for (const word of words) {
    const lower = word.toLowerCase();
    if (lower === "the") {
      definiteCount++;
    } else if (lower === "a" || lower === "an") {
      indefiniteCount++;
    }
  }

  return [definiteCount / words.length, indefiniteCount / words.length];
}

/**
 * Extract punctuation frequency.
 */
export function extractPunctuation(text: string): number[] {
  const counts = new Array(Object.keys(PUNCTUATION_SYMBOL_MAP).length).fill(0);

  if (text.length === 0) {
    return counts;
  }

  for (const char of text) {
    if (char in PUNCTUATION_SYMBOL_MAP) {
      counts[PUNCTUATION_SYMBOL_MAP[char]]++;
    }
  }

  const totalPunctuation = counts.reduce((a, b) => a + b, 0);
  return totalPunctuation > 0
    ? counts.map((c) => c / totalPunctuation)
    : counts;
}

/**
 * Extract contraction ratio.
 */
export function extractContractions(words: string[]): number {
  if (words.length === 0) return 0;

  const contractionCount = words.filter((word) => word.includes("'")).length;
  return contractionCount / words.length;
}

/**
 * Extract casing features (position × casing).
 */
export function extractCasing(words: string[]): number[] {
  const counts = new Array(9).fill(0); // 3 positions × 3 casing types

  if (words.length === 0) {
    return counts;
  }

  const getCasingType = (word: string): number => {
    if (word.length === 0) return 0;
    const firstChar = word[0];
    const isAllUpper = word === word.toUpperCase();
    const isTitle = firstChar === firstChar.toUpperCase();

    if (isAllUpper) return 2; // uppercase
    if (isTitle) return 1; // title case
    return 0; // lowercase
  };

  for (let i = 0; i < words.length; i++) {
    const casingType = getCasingType(words[i]);
    let positionOffset = 0;

    if (i === 0) {
      positionOffset = 0;
    } else if (i === words.length - 1) {
      positionOffset = 6;
    } else {
      positionOffset = 3;
    }

    counts[positionOffset + casingType]++;
  }

  return counts.map((c) => c / words.length);
}

/**
 * Extract casing bigrams.
 */
export function extractCasingBigrams(words: string[]): number[] {
  const counts = new Array(9).fill(0);

  if (words.length < 2) {
    return counts;
  }

  const getCasingType = (word: string): number => {
    if (word.length === 0) return 0;
    const isAllUpper = word === word.toUpperCase();
    const isTitle = word[0] === word[0].toUpperCase();

    if (isAllUpper) return 2;
    if (isTitle) return 1;
    return 0;
  };

  for (let i = 0; i < words.length - 1; i++) {
    const casingType1 = getCasingType(words[i]);
    const casingType2 = getCasingType(words[i + 1]);
    const idx = casingType1 * 3 + casingType2;
    counts[idx]++;
  }

  const totalBigrams = words.length - 1;
  return counts.map((c) => c / totalBigrams);
}

/**
 * Stub implementations for BookNLP-dependent features.
 * These return zero-filled arrays and will be properly implemented
 * when BookNLP integration is added.
 */

export function extractPOSFrequency(
  tokens: TokenData[] | null
): number[] {
  // Returns 50-dimensional vector for POS tags
  return new Array(50).fill(0);
}

export function extractPOSNgrams(tokens: TokenData[] | null): number[] {
  // Returns 995-dimensional vector for POS n-grams
  return new Array(995).fill(0);
}

export function extractDependencyTreeStructure(
  sentences: SentenceData[] | null
): number[] {
  // Returns 92-dimensional vector for tree structural features
  return new Array(92).fill(0);
}

export function extractDependencyTreeRelations(
  sentences: SentenceData[] | null
): number[] {
  // Returns 1319-dimensional vector for dependency relations
  return new Array(1319).fill(0);
}

export function extractNounPhraseLengths(words: string[]): number[] {
  // Returns 14-dimensional vector for noun phrase lengths
  // Placeholder: returns simple word length distribution
  return extractWordLengthByChar(words).slice(0, 14);
}

export function extractEntityCategories(
  tokens: TokenData[] | null,
  words: string[]
): number[] {
  // Returns 6-dimensional vector for entity categories
  return new Array(6).fill(0);
}

export function extractEvents(tokens: TokenData[] | null): number {
  // Returns event ratio
  return 0;
}

export function extractSupersense(tokens: TokenData[] | null, words: string[]): number[] {
  // Returns 41-dimensional vector for supersense labels
  return new Array(41).fill(0);
}

export function extractTense(tokens: TokenData[] | null): number[] {
  // Returns [past, present, future, none] - 4 dimensions
  return [0, 0, 0, 0];
}

export function extractPolysemy(words: string[]): number[] {
  // Returns 15-dimensional vector for polysemy bins
  return new Array(15).fill(0);
}

export function extractWordConcreteness(words: string[]): number[] {
  // Returns 20 bins + 1 average = 21 dimensions
  return new Array(21).fill(0);
}

export function extractPrepositionImageability(): number[] {
  // Returns 10 bins + 1 average = 11 dimensions
  return new Array(11).fill(0);
}

export function extractPlaces(words: string[]): number {
  // Returns places ratio
  return 0;
}

export function extractCharNgrams(text: string): number[] {
  // Placeholder: returns 1000-dimensional zero array
  // Full implementation would require pre-computed n-gram features map
  return new Array(1000).fill(0);
}
