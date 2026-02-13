import type {
  TokenData,
  EntityData,
  SentenceData,
  BookNLPContext,
} from "~/types/text2features-worker";

import { getPolysemyCount, downloadAndLoadWordNet } from "~/utils/wordnet";
import { estimate as estimateSyllables } from "~/utils/syllables";
import { WORDNETS } from "~/utils/models";

// dynamic import path for booknlp-ts; allow fallback if not available
let _booknlpPipeline: any | null = null;
let _featureResourcesLoaded: Promise<void> | null = null;
let _charFeaturesMap: Record<string, number> | null = null;
let _posFeaturesMap: Record<string, number> | null = null;
let _depNodeFeaturesMap: Record<string, number> | null = null;
let _depRelationFeaturesMap: Record<string, number> | null = null;
let _depCompleteFeaturesMap: Record<string, number> | null = null;
let _concretenessSingle: Record<string, number> | null = null;
let _concretenessMulti: Record<string, number> | null = null;
let _prepositions: Array<{ pattern: string; score: number }> | null = null;
let _placesList: string[] | null = null;

async function parseCsv(text: string): Promise<string[][]> {
  const lines = text.split(/\r?\n/).filter((l) => l.trim().length > 0);
  if (lines.length === 0) return [];
  const rows: string[][] = [];
  for (const line of lines) {
    // naive CSV split, assumes no embedded commas in fields
    rows.push(line.split(",").map((c) => c.trim()));
  }
  return rows;
}

async function ensureFeatureResourcesLoaded(): Promise<void> {
  if (_featureResourcesLoaded) return _featureResourcesLoaded;
  _featureResourcesLoaded = (async () => {
    try {
      const base = "/assets/data/features";
      // char ngrams
      try {
        const resp = await fetch(`${base}/char_ngrams_features.csv`);
        const txt = await resp.text();
        const rows = await parseCsv(txt);
        const header = rows[0] || [];
        const idx = header.indexOf("ngram");
        _charFeaturesMap = {};
        for (let i = 1; i < rows.length; i++) {
          const row = rows[i];
          const ngram = row[idx] || row[0];
          _charFeaturesMap[ngram] = i - 1;
        }
      } catch (e) {
        _charFeaturesMap = null;
      }

      // pos ngrams
      try {
        const resp = await fetch(`${base}/pos_ngrams_features.csv`);
        const txt = await resp.text();
        const rows = await parseCsv(txt);
        const header = rows[0] || [];
        const idx = header.indexOf("ngram");
        _posFeaturesMap = {};
        for (let i = 1; i < rows.length; i++) {
          const row = rows[i];
          const raw = row[idx] || row[0];
          const key1 = raw;
          const key2 = raw.replace(/[\[\]\(\)\"' ]+/g, "").replace(/,/g, "_");
          _posFeaturesMap[key1] = i - 1;
          _posFeaturesMap[key2] = i - 1;
        }
      } catch (e) {
        _posFeaturesMap = null;
      }

      // dep tree ngrams
      try {
        const resp = await fetch(`${base}/dep_tree_node_ngrams_features.csv`);
        const txt = await resp.text();
        const rows = await parseCsv(txt);
        const header = rows[0] || [];
        const idx = header.indexOf("ngram");
        _depNodeFeaturesMap = {};
        for (let i = 1; i < rows.length; i++) {
          const row = rows[i];
          const raw = row[idx] || row[0];
          _depNodeFeaturesMap[raw] = i - 1;
        }
      } catch (e) {
        _depNodeFeaturesMap = null;
      }

      try {
        const resp = await fetch(`${base}/dep_tree_relation_ngrams_features.csv`);
        const txt = await resp.text();
        const rows = await parseCsv(txt);
        const header = rows[0] || [];
        const idx = header.indexOf("ngram");
        _depRelationFeaturesMap = {};
        for (let i = 1; i < rows.length; i++) {
          const row = rows[i];
          const raw = row[idx] || row[0];
          _depRelationFeaturesMap[raw] = i - 1;
        }
      } catch (e) {
        _depRelationFeaturesMap = null;
      }

      try {
        const resp = await fetch(`${base}/dep_tree_complete_ngrams_features.csv`);
        const txt = await resp.text();
        const rows = await parseCsv(txt);
        const header = rows[0] || [];
        const idx = header.indexOf("ngram");
        _depCompleteFeaturesMap = {};
        for (let i = 1; i < rows.length; i++) {
          const row = rows[i];
          const raw = row[idx] || row[0];
          _depCompleteFeaturesMap[raw] = i - 1;
        }
      } catch (e) {
        _depCompleteFeaturesMap = null;
      }

      // concreteness and related resources
      try {
        const resp = await fetch(`${base}/concreteness/words.csv`);
        const txt = await resp.text();
        const rows = await parseCsv(txt);
        _concretenessSingle = {};
        const header = rows[0] || [];
        const widx = header.indexOf('word');
        const sidx = header.indexOf('score');
        for (let i = 1; i < rows.length; i++) {
          const row = rows[i];
          const w = casefold(row[widx] || row[0] || '');
          const s = parseFloat(row[sidx] || row[1] || '0') || 0;
          _concretenessSingle[w] = s;
        }
      } catch (e) {
        _concretenessSingle = null;
      }

      try {
        const resp = await fetch(`${base}/concreteness/multiword.csv`);
        const txt = await resp.text();
        const rows = await parseCsv(txt);
        _concretenessMulti = {};
        const header = rows[0] || [];
        const widx = header.indexOf('expression');
        const sidx = header.indexOf('score');
        for (let i = 1; i < rows.length; i++) {
          const row = rows[i];
          const w = casefold(row[widx] || row[0] || '');
          const s = parseFloat(row[sidx] || row[1] || '0') || 0;
          _concretenessMulti[w] = s;
        }
      } catch (e) {
        _concretenessMulti = null;
      }

      try {
        const resp = await fetch(`${base}/concreteness/prepositions.csv`);
        const txt = await resp.text();
        const rows = await parseCsv(txt);
        _prepositions = [];
        const header = rows[0] || [];
        const pidx = header.indexOf('pattern');
        const sidx = header.indexOf('score');
        for (let i = 1; i < rows.length; i++) {
          const row = rows[i];
          const p = (row[pidx] || row[0] || '');
          const s = parseFloat(row[sidx] || row[1] || '0') || 0;
          _prepositions.push({ pattern: p, score: s });
        }
      } catch (e) {
        _prepositions = null;
      }

      try {
        const resp = await fetch(`${base}/concreteness/places.txt`);
        const txt = await resp.text();
        _placesList = txt.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
      } catch (e) {
        _placesList = null;
      }
    } catch (err) {
      // swallow errors — feature extraction will degrade gracefully
      console.warn('Failed to load feature resources', err);
    }
  })();
  return _featureResourcesLoaded;
}

const ENG_POS_TAGS = [
  "$",
  "''",
  ",",
  "-LRB-",
  "-RRB-",
  ".",
  ":",
  "ADD",
  "AFX",
  "CC",
  "CD",
  "DT",
  "EX",
  "FW",
  "HYPH",
  "IN",
  "JJ",
  "JJR",
  "JJS",
  "LS",
  "MD",
  "NFP",
  "NN",
  "NNP",
  "NNPS",
  "NNS",
  "PDT",
  "POS",
  "PRP",
  "PRP$",
  "RB",
  "RBR",
  "RBS",
  "RP",
  "SYM",
  "TO",
  "UH",
  "VB",
  "VBD",
  "VBG",
  "VBN",
  "VBP",
  "VBZ",
  "WDT",
  "WP",
  "WP$",
  "WRB",
  "XX",
  "_SP",
  "``",
];

const ENG_POS_TAG_MAP: Record<string, number> = ENG_POS_TAGS.reduce((acc, t, i) => {
  acc[t] = i;
  return acc;
}, {} as Record<string, number>);

const ENTITY_CATEGORY_MAP: Record<string, number> = {
  PER: 0,
  FAC: 1,
  GPE: 2,
  LOC: 3,
  VEH: 4,
  ORG: 5,
};

const SUPERSENSE_LABELS_NOUNS = [
  "act",
  "animal",
  "artifact",
  "attribute",
  "body",
  "cognition",
  "communication",
  "event",
  "feeling",
  "food",
  "group",
  "location",
  "motive",
  "object",
  "quantity",
  "phenomenon",
  "plant",
  "possession",
  "process",
  "person",
  "relation",
  "shape",
  "state",
  "substance",
  "time",
  "Tops",
];

const SUPERSENSE_LABELS_VERBS = [
  "body",
  "change",
  "cognition",
  "communication",
  "competition",
  "consumption",
  "contact",
  "creation",
  "emotion",
  "motion",
  "perception",
  "possession",
  "social",
  "stative",
  "weather",
];

const SUPERSENSE_LABELS = [
  ...SUPERSENSE_LABELS_NOUNS.map((l) => `noun.${l}`),
  ...SUPERSENSE_LABELS_VERBS.map((l) => `verb.${l}`),
];

const SUPERSENSE_LABEL_MAP: Record<string, number> = SUPERSENSE_LABELS.reduce((acc, s, i) => {
  acc[s] = i;
  return acc;
}, {} as Record<string, number>);

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

// Load FEATURE_NAMES via Vite import from assets so bundler includes it.
// The JSON file lives under `assets/data/features/feature-names.json`.
import featureNamesData from "~/assets/data/features/feature-names.json";
export const FEATURE_NAMES: string[] = featureNamesData as string[];

// Runtime sanity checks: ensure FEATURE_COUNT matches extractor counts sum
{
  const total = Object.values(EXTRACTOR_FEATURE_COUNTS).reduce((a, b) => a + b, 0);
  if (total !== FEATURE_COUNT) {
    throw new Error(`FEATURE_COUNT mismatch: ${total} != ${FEATURE_COUNT}`);
  }
}

const PREPROCESS_TRANSLATION_TABLE: Record<string, string | null> = {
  "\r": null,
  "\u2019": "'",
  "\u201c": "\"",
  "\u201d": "\"",
  "？": "?",
  "！": "!",
};

/**
 * Unicode-aware casefold helper using NFKC normalization and small mappings.
 * This is a conservative casefold implementation suitable for feature extraction.
 */
export function casefold(s: string): string {
  if (!s) return s;
  // Normalize to NFKC to combine compatibility characters
  let v = s.normalize("NFKC");
  // Basic mappings (extend if needed)
  v = v.replace(/\u00DF/g, "ss"); // German sharp s
  v = v.replace(/\u017F/g, "s"); // long s
  // Final lowercase (locale-insensitive)
  return v.toLowerCase();
}

// Tokenization: words, contractions, numbers, ellipses, punctuation
const TOKEN_REGEX = /(?:\.\.\.|\p{L}+(?:'\p{L}+)?)|\d+(?:\.\d+)?|[^\s\p{L}\d]/gu;

/**
 * Tokenize text in a way that mirrors the Python tokenization used in extractors.
 * Preserves contractions (don't), ellipses (...) and returns punctuation as tokens.
 */
export function tokenize(text: string): string[] {
  const src = preprocess(text);
  const matches = [...src.matchAll(TOKEN_REGEX)];
  return matches.map((m) => m[0]).filter(Boolean);
}

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

export const PUNCTUATION = PUNCTUATION_SYMBOL_MAP;

// Note: stopword lists are removed from the frontend port. The BookNLP/Python
// pipeline provides `isStop` on tokens; feature extraction should rely on
// that field instead of an embedded stopword set.

export const CONTRACTIONS: Record<string, string> = {
  "aren't": "are not",
  "can't": "cannot",
  "couldn't": "could not",
  "didn't": "did not",
  "doesn't": "does not",
  "don't": "do not",
  "hadn't": "had not",
  "hasn't": "has not",
  "haven't": "have not",
  "he'd": "he would",
  "he'll": "he will",
  "he's": "he is",
  "i'd": "i would",
  "i'll": "i will",
  "i'm": "i am",
  "i've": "i have",
  "isn't": "is not",
  "it's": "it is",
  "she'd": "she would",
  "she'll": "she will",
  "she's": "she is",
  "shouldn't": "should not",
  "that's": "that is",
  "they'd": "they would",
  "they'll": "they will",
  "they're": "they are",
  "they've": "they have",
  "we'd": "we would",
  "we'll": "we will",
  "we're": "we are",
  "we've": "we have",
  "won't": "will not",
  "wouldn't": "would not",
  "you're": "you are",
  "you've": "you have",
};

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
 * Simple whitespace-based tokenizer exposed for worker use.
 */
export function tokenizeSimple(text: string): string[] {
  // preserve old behaviour (lowercased whitespace split) but delegate to new tokenizer
  return tokenize(text).map((t) => casefold(t)).filter((w) => w.length > 0);
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

export async function getBookNLPContext(spaCyContext: any): Promise<BookNLPContext> {
  if (!_booknlpPipeline) {
    // TODO add the package from NPM when uploaded
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const mod = await import('/home/terra/Projects/booknlp/booknlp-ts/src/english-booknlp');
    _booknlpPipeline = await mod.createPipeline({ pipeline: 'entity,supersense,event' });
  }

  const result = await _booknlpPipeline.process(spaCyContext);

  // Map BookNLPResult to our BookNLPContext
  const ctx: BookNLPContext = {
    tokens: (result.tokens || []).map((tk: any) => ({
      text: tk.text || '',
      itext: casefold(tk.text || ''),
      pos: tk.pos || '',
      finePOS: tk.finePos || tk.pos || '',
      lemma: tk.lemma || casefold(tk.text || ''),
      sentenceId: tk.sentenceId ?? 0,
      withinSentenceId: tk.withinSentenceId ?? 0,
      event: !!tk.event,
      isStop: !!tk.isStop,
      likeNum: !!tk.likeNum,
      morphTense: tk.morph?.Tense ?? null,
    })),
    entities: (result.entities || []).map((e: any) => ({
      startToken: e.startToken,
      endToken: e.endToken,
      cat: e.cat || '',
      text: e.text || '',
      coref: e.coref ?? -1,
      prop: e.prop || '',
    })),
    supersense: result.supersense || [],
    sentences: (result.sents || []).map((s: any) => ({ root: { text: '', pos: '', dep: '', children: [] } })),
  };

  return ctx;
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

  const uniqueWords = new Set(words.map((w) => casefold(w)));
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

// Syllable estimation is provided by the imported `estimateSyllables` from `syllables.ts`.

/**
 * Extract stopword ratio.
 */
export function extractStopwords(tokens: TokenData[] | null): number {
  // Rely on `token.isStop` provided by BookNLP; if tokens are not available,
  // return 0 so frontend does not guess stopwords.
  if (!tokens || tokens.length === 0) return 0;
  const stopwordCount = tokens.filter((t) => !!t.isStop).length;
  return stopwordCount / tokens.length;
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
    const lower = casefold(word);
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
export function extractContractions(tokens: TokenData[] | null): number {
  // Match Python behaviour: fraction of words containing an apostrophe
  if (!tokens || tokens.length === 0) return 0;
  const contractionCount = tokens.filter((t) => (t.text || "").includes("'")).length;
  return contractionCount / tokens.length;
}

/**
 * Extract casing features (position × casing).
 */
export function extractCasing(tokens: TokenData[] | null): number[] {
  // Mirror Python: position is relative to sentence (first, inbetween, last)
  const counts = new Array(9).fill(0);
  if (!tokens || tokens.length === 0) return counts;

  const getCasingType = (text: string): number => {
    if (!text) return 0;
    if (text === text.toLowerCase()) return 0; // lower
    if (text === text.toUpperCase()) return 1; // upper
    // Title-case heuristic: first character uppercase
    const firstChar = text[0];
    if (firstChar === firstChar.toUpperCase()) return 2; // title
    return 0;
  };

  for (let i = 0; i < tokens.length; i++) {
    const token = tokens[i];
    const casingType = getCasingType(token.text || "");

    let positionIdx = 1; // inbetween
    if (token.withinSentenceId === 0) {
      positionIdx = 0; // first
    } else if (i === tokens.length - 1 || (tokens[i + 1] && tokens[i + 1].withinSentenceId === 0)) {
      positionIdx = 2; // last
    }

    counts[positionIdx * 3 + casingType]++;
  }

  return counts.map((c) => c / tokens.length);
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
  const counts = new Array(ENG_POS_TAGS.length).fill(0);
  if (!tokens || tokens.length === 0) return counts;

  let total = 0;
  for (const t of tokens) {
    const tag = (t.finePOS || t.pos || '').toString();
    if (tag in ENG_POS_TAG_MAP) {
      counts[ENG_POS_TAG_MAP[tag]]++;
    }
    total++;
  }

  if (total === 0) return counts;
  return counts.map((c) => c / total);
}

export function extractPOSNgrams(tokens: TokenData[] | null): number[] {
  const DIM = 995;
  const counts = new Array(DIM).fill(0);
  if (!tokens || tokens.length === 0) return counts;

  // Use preloaded pos ngram map if available
  const seq = tokens.map((t) => (t.finePOS || t.pos || ''));

  // Try to use map for matching ngrams up to length 3
  const tryMatch = (ng: string) => {
    if (!_posFeaturesMap) return -1;
    if (ng in _posFeaturesMap) return _posFeaturesMap[ng];
    const norm = ng.replace(/\s+/g, '').replace(/\(|\)|\"|\'|\[/g, '').replace(/,/g, '_');
    if (norm in _posFeaturesMap) return _posFeaturesMap[norm];
    return -1;
  };

  const ngramList: number[] = [];
  for (let n = 1; n <= 3; n++) {
    for (let i = 0; i + n <= seq.length; i++) {
      const ng = seq.slice(i, i + n).join(',');
      const idx = tryMatch(ng);
      if (idx >= 0) ngramList.push(idx);
    }
  }

  if (ngramList.length === 0) return counts;
  for (const idx of ngramList) counts[idx]++;
  const total = ngramList.length;
  return counts.map((c) => c / total);
}

export function extractDependencyTreeStructure(
  sentences: SentenceData[] | null
): number[] {
  const DIM = 92;
  const out = new Array(DIM).fill(0);
  if (!sentences || sentences.length === 0) return out;

  // Collect simple stats: avg depth, max depth, avg branching
  let totalDepth = 0;
  let totalNodes = 0;
  let maxDepth = 0;
  let totalBranching = 0;

  function walk(node: any, depth: number) {
    totalDepth += depth;
    totalNodes++;
    maxDepth = Math.max(maxDepth, depth);
    totalBranching += (node.children || []).length;
    for (const c of node.children || []) walk(c, depth + 1);
  }

  for (const s of sentences) {
    if (!s || !s.root) continue;
    walk(s.root, 0);
  }

  if (totalNodes === 0) return out;

  const avgDepth = totalDepth / totalNodes;
  const avgBranch = totalBranching / totalNodes;

  // Fill first slots with these normalized values
  out[0] = avgDepth / (maxDepth || 1);
  out[1] = maxDepth / 30;
  out[2] = avgBranch / 5;

  // Remaining dims: simple histograms via hashing of depth and branching
  let idx = 3;
  for (let d = 0; d < 44 && idx < DIM; d++, idx++) out[idx] = Math.min(1, (d === Math.floor(avgDepth) ? 1 : 0));
  for (let b = 0; b < 45 && idx < DIM; b++, idx++) out[idx] = Math.min(1, (b === Math.floor(avgBranch) ? 1 : 0));

  return out;
}

export function extractDependencyTreeRelations(
  sentences: SentenceData[] | null
): number[] {
  const DIM = 1319;
  const out = new Array(DIM).fill(0);
  if (!sentences || sentences.length === 0) return out;

  // Collect relation strings
  const rels: string[] = [];
  function walkCollect(node: any) {
    for (const c of node.children || []) {
      if (c.dep) rels.push(c.dep);
      walkCollect(c);
    }
  }

  for (const s of sentences) {
    if (!s || !s.root) continue;
    walkCollect(s.root);
  }

  if (_depRelationFeaturesMap) {
    for (const r of rels) {
      const idx = _depRelationFeaturesMap[r];
      if (idx !== undefined) out[idx]++;
    }
    const total = rels.length || 1;
    return out.map((c) => c / total);
  }

  // fallback: hash into buckets
  for (const r of rels) {
    let h = 2166136261 >>> 0;
    for (let i = 0; i < r.length; i++) {
      h ^= r.charCodeAt(i);
      h = Math.imul(h, 16777619) >>> 0;
    }
    out[h % DIM]++;
  }

  const total = rels.length || 1;
  return out.map((c) => c / total);
}

export function extractNounPhraseLengths(words: string[]): number[] {
  // Returns 14-dimensional vector of noun-phrase length distribution
  const DIM = 14;
  const counts = new Array(DIM).fill(0);
  if (words.length === 0) return counts;

  // approximate by grouping adjacent capitalized tokens as NPs
  let i = 0;
  const n = words.length;
  while (i < n) {
    if (words[i][0] === words[i][0]?.toUpperCase()) {
      let j = i + 1;
      while (j < n && words[j][0] === words[j][0]?.toUpperCase()) j++;
      const len = Math.min(j - i, DIM);
      counts[len - 1]++;
      i = j;
    } else {
      i++;
    }
  }

  const total = counts.reduce((a, b) => a + b, 0) || 1;
  return counts.map((c) => c / total);
}

export function extractEntityCategories(
  tokens: TokenData[] | null,
  words: string[]
): number[] {
  const counts = new Array(6).fill(0);
  if (!tokens || tokens.length === 0) return counts;

  // Entities are not directly available here; infer by capitalization sequences
  // Simple heuristic: capitalized sequences => PERSON/ORG/GPE
  let i = 0;
  while (i < tokens.length) {
    const t = tokens[i];
    if (t.text[0] === t.text[0]?.toUpperCase()) {
      let j = i + 1;
      while (j < tokens.length && tokens[j].text[0] === tokens[j].text[0]?.toUpperCase()) j++;
      // heuristic category
      const span = tokens.slice(i, j).map((x) => x.text).join(' ');
      if (span.match(/\b(St|Mr|Mrs|Ms)\b/)) counts[0]++; // PER
      else if (span.match(/\b(Street|St|Ave|Road|Rd|Lane|Blvd)\b/)) counts[1]++; // FAC
      else if (span.match(/\b(City|Town|County|State)\b/)) counts[2]++; // GPE
      else counts[5]++; // ORG/other
      i = j;
    } else {
      i++;
    }
  }

  const total = counts.reduce((a, b) => a + b, 0) || 1;
  return counts.map((c) => c / total);
}

export function extractEvents(tokens: TokenData[] | null): number {
  if (!tokens || tokens.length === 0) return 0;
  const ev = tokens.filter((t) => !!t.event).length;
  return ev / tokens.length;
}

export function extractSupersense(tokens: TokenData[] | null, words: string[]): number[] {
  const counts = new Array(SUPERSENSE_LABELS.length).fill(0);
  // In our BookNLPContext the supersense annotations are provided separately; here
  // we cannot access them if only tokens are passed. Return zeros unless tokens
  // include a special supersense field (not present in TokenData type).
  return counts;
}

export function extractTense(tokens: TokenData[] | null): number[] {
  if (!tokens || tokens.length === 0) return [0, 0, 0, 0];
  let past = 0;
  let present = 0;
  let future = 0;
  let none = 0;
  for (const t of tokens) {
    const tense = casefold(t.morphTense || '');
    if (tense.includes('past')) past++;
    else if (tense.includes('pres')) present++;
    else if (tense.includes('fut')) future++;
    else none++;
  }
  const total = tokens.length || 1;
  return [past / total, present / total, future / total, none / total];
}

export function extractPolysemy(words: string[]): number[] {
  const DIM = 15;
  const counts = new Array(DIM).fill(0);
  if (!words || words.length === 0) return counts;

  try {
    // trigger background download of default WordNet resource
    const defaultWn = WORDNETS && WORDNETS.length > 0 ? WORDNETS[0] : null;
    if (defaultWn) {
      void downloadAndLoadWordNet(defaultWn.id, defaultWn.downloadUrl).catch(() => {});
    }

    for (const w of words) {
      const n = getPolysemyCount(w);
      const idx = Math.min(Math.max(0, Math.floor(n)), DIM - 1);
      counts[idx] += 1;
    }

    return counts.map((c) => c / words.length);
  } catch (err) {
    return new Array(DIM).fill(0);
  }
}

export function extractWordConcreteness(words: string[]): number[] {
  const DIM = 20;
  const out = new Array(DIM + 1).fill(0);
  if (!words || words.length === 0) return out;
  const scores: number[] = [];

  const multi = _concretenessMulti || {};
  const single = _concretenessSingle || {};
  const multiKeys = Object.keys(multi);
  let maxMultiWords = 1;
  for (const k of multiKeys) {
    const len = k.split(/\s+/).length;
    if (len > maxMultiWords) maxMultiWords = len;
  }

  let i = 0;
  while (i < words.length) {
    let matched = false;
    for (let w = maxMultiWords; w >= 1; w--) {
      if (i + w > words.length) continue;
      const phrase = casefold(words.slice(i, i + w).join(' '));
      if (multi[phrase] !== undefined) {
        scores.push(multi[phrase]);
        i += w;
        matched = true;
        break;
      }
    }
    if (!matched) {
      const s = single[casefold(words[i])];
      if (s !== undefined) scores.push(s);
      i++;
    }
  }

  if (scores.length === 0) return out;

  // assume concreteness scores are in range ~1..5
  const min = 1;
  const max = 5;
  const binSize = (max - min) / DIM;
  for (const sc of scores) {
    const idx = Math.min(DIM - 1, Math.max(0, Math.floor((sc - min) / binSize)));
    out[idx]++;
  }

  const total = scores.length;
  for (let j = 0; j < DIM; j++) out[j] = out[j] / total;
  const avg = scores.reduce((a, b) => a + b, 0) / total;
  out[DIM] = avg;
  return out;
}

export function extractPrepositionImageability(words: string[]): number[] {
  const BINS = 10;
  const out = new Array(BINS + 1).fill(0);
  if (!words || words.length === 0) return out;
  if (!_prepositions || _prepositions.length === 0) return out;

  // build map for quick lookup
  const prepMap: Record<string, number> = {};
  let maxMulti = 1;
  for (const p of _prepositions) {
    const key = casefold(p.pattern || '');
    prepMap[key] = p.score;
    const l = key.split(/\s+/).length;
    if (l > maxMulti) maxMulti = l;
  }

  const scores: number[] = [];
  let i = 0;
  while (i < words.length) {
    let matched = false;
    for (let w = maxMulti; w >= 1; w--) {
      if (i + w > words.length) continue;
      const phrase = casefold(words.slice(i, i + w).join(' '));
      if (prepMap[phrase] !== undefined) {
        scores.push(prepMap[phrase]);
        i += w;
        matched = true;
        break;
      }
    }
    if (!matched) i++;
  }

  if (scores.length === 0) return out;

  // assume scores roughly 1..5
  const min = 1;
  const max = 5;
  const binSize = (max - min) / BINS;
  for (const sc of scores) {
    const idx = Math.min(BINS - 1, Math.max(0, Math.floor((sc - min) / binSize)));
    out[idx]++;
  }
  const total = scores.length;
  for (let j = 0; j < BINS; j++) out[j] = out[j] / total;
  out[BINS] = scores.reduce((a, b) => a + b, 0) / total;
  return out;
}

export function extractPlaces(words: string[]): number {
  if (!words || words.length === 0) return 0;

  const set = new Set(_placesList!.map((p) => casefold(p)));
  // determine max phrase length in words
  let maxLen = 1;
  for (const p of _placesList!) {
    const l = p.trim().split(/\s+/).length;
    if (l > maxLen) maxLen = l;
  }

  let i = 0;
  let matches = 0;
  while (i < words.length) {
    let matched = false;
    for (let l = Math.min(maxLen, words.length - i); l >= 1; l--) {
      const phrase = casefold(words.slice(i, i + l).join(" "));
      if (set.has(phrase)) {
        matches++;
        i += l;
        matched = true;
        break;
      }
    }
    if (!matched) i++;
  }
  return matches / words.length;
}

export function extractCharNgrams(text: string): number[] {
  const DIM = 1000;
  const counts = new Array(DIM).fill(0);
  if (!text) return counts;

  if (!_charFeaturesMap) return counts;

  const s = casefold(text);
  // generate char ngrams of length 3..5
  for (let n = 3; n <= 5; n++) {
    for (let i = 0; i + n <= s.length; i++) {
      const ng = s.substring(i, i + n);
      const idx = _charFeaturesMap[ng];
      if (idx !== undefined) counts[idx]++;
    }
  }

  const total = counts.reduce((a, b) => a + b, 0) || 1;
  return counts.map((c) => c / total);
}

/**
 * High-level feature extraction for a single text. Synchronous and uses
 * the individual extractor functions exported in this module. Returns
 * an object with the original text and the assembled feature vector.
 */
export async function extractFeaturesFromText(text: string): Promise<{ text: string; features: number[] }> {
  const preprocessedText = preprocess(text);
  await ensureFeatureResourcesLoaded();
  const ctx = await getBookNLPContext(preprocessedText);
  const words = (ctx.tokens || []).map((t: any) => t.itext || t.text || "").filter((w: any) => w.length > 0);

  const features: number[] = [];

  features.push(...extractQuoteRatioFeature(text));
  features.push(...extractCharNgrams(text));
  features.push(...extractWordLengthByChar(words));
  features.push(...extractNgramWordLengthByChar(words));

  // Sentence length features not available here (needs sentence tokenizer)
  features.push(...new Array(27).fill(0)); // 26 bins + avg

  features.push(extractNumericWordRatio(words));
  features.push(extractTTR(words));
  features.push(extractLexicalDensity(words));
  features.push(...extractSyllableRatios(words));
  features.push(extractStopwords(ctx.tokens));
  features.push(...extractArticles(words));
  features.push(...extractPunctuation(text));
  features.push(extractContractions(ctx.tokens));
  features.push(...extractCasing(ctx.tokens));
  features.push(...extractCasingBigrams(words));

  // BookNLP-dependent features using the extracted context
  features.push(...extractPOSFrequency(ctx.tokens));
  features.push(...extractPOSNgrams(ctx.tokens));
  features.push(...extractDependencyTreeStructure(ctx.sentences));
  features.push(...extractDependencyTreeRelations(ctx.sentences));

  features.push(...extractNounPhraseLengths(words));
  features.push(...extractEntityCategories(ctx.tokens, words));
  features.push(extractEvents(ctx.tokens));
  features.push(...extractSupersense(ctx.tokens, words));
  features.push(...extractTense(ctx.tokens));
  features.push(...extractPolysemy(words));
  features.push(...extractWordConcreteness(words));
  features.push(...extractPrepositionImageability(words));
  features.push(extractPlaces(words));

  return { text, features: features.map((f) => (Number.isNaN(Number(f)) ? 0 : f)) };
}
