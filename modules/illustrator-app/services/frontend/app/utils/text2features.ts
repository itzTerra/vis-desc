import { getPolysemyCount, downloadAndLoadWordNet } from "~/utils/wordnet";
import { estimate as estimateSyllables } from "~/utils/syllables";
import { WORDNETS } from "~/utils/models";
import { MultiWordExpressionTrie } from "~/utils/multiword-trie";
import { BookNLP, type BookNLPResult, type ExecutionProvider, type SpaCyContext } from "booknlp-ts";
import { pipeline, type FeatureExtractionPipeline } from "@huggingface/transformers";
import charNgrams from "~/assets/data/features/char_ngrams_features.json";
import posNgrams from "~/assets/data/features/pos_ngrams_features.json";
import depNodeNgrams from "~/assets/data/features/dep_tree_node_ngrams_features.json";
import depRelationNgrams from "~/assets/data/features/dep_tree_relation_ngrams_features.json";
import depCompleteNgrams from "~/assets/data/features/dep_tree_complete_ngrams_features.json";
import concretenessWords from "~/assets/data/features/concreteness/words.json";
import concretenessMulti from "~/assets/data/features/concreteness/multiword.json";
import concretenessPrepositions from "~/assets/data/features/concreteness/prepositions.json";
import concretenessPlaces from "~/assets/data/features/concreteness/places.json";
import type { ProgressCallback } from "~/types/common";
import type { HFPipelineConfig } from "~/types/cache";

interface TokenData {
  text: string;
  itext: string; // lowercase text
  pos: string; // UD POS tag
  fine_pos: string; // fine-grained POS tag
  lemma: string;
  sentence_id: number;
  within_sentence_id: number;
  event: boolean;
  is_stop: boolean;
  like_num: boolean;
  morph_tense?: string | null;
}

interface EntityData {
  start_token: number;
  end_token: number;
  cat: string;
  text: string;
  prop: string;
  coref: number;
}

interface NounChunkData {
  text: string;
  start: number;
  end: number;
  length: number;
}

class SentenceToken {
  text: string;
  pos: string;
  dep: string;
  children: SentenceToken[];

  constructor(
    text: string,
    pos: string,
    dep: string,
    children: SentenceToken[] = []
  ) {
    this.text = text;
    this.pos = pos;
    this.dep = dep;
    this.children = children;
  }

  // Depth-first tokens list
  getTokens(): SentenceToken[] {
    const out: SentenceToken[] = [];
    const traverse = (t: SentenceToken) => {
      out.push(t);
      for (const c of t.children) traverse(c);
    };
    traverse(this);
    return out;
  }
}

class SentenceData {
  root: SentenceToken;

  constructor(root: SentenceToken) {
    this.root = root;
  }

  get_tokens(): SentenceToken[] {
    return this.root.getTokens();
  }
}

type SupersenseEntry = [number, number, string, string];

class ExtCtx {
  text: string;
  tokens: TokenData[];
  words: TokenData[];
  noun_chunks: NounChunkData[];
  entities: EntityData[];
  supersense: SupersenseEntry[];
  sents: SentenceData[];

  constructor(
    text: string,
    tokens: TokenData[] = [],
    words: TokenData[] = [],
    noun_chunks: NounChunkData[] = [],
    entities: EntityData[] = [],
    supersense: SupersenseEntry[] = [],
    sents: SentenceData[] = []
  ) {
    this.text = text;
    this.tokens = tokens;
    this.words = words;
    this.noun_chunks = noun_chunks;
    this.entities = entities;
    this.supersense = supersense;
    this.sents = sents;
  }

  static fromBookNLPResult(booknlpResult: BookNLPResult, text: string): ExtCtx {
    const tokens: TokenData[] = booknlpResult.tokens.map((t) => ({
      text: t.text,
      itext: t.itext,
      pos: t.pos!,
      fine_pos: t.finePos!,
      lemma: t.lemma!,
      sentence_id: t.sentenceId,
      within_sentence_id: t.withinSentenceId,
      event: t.event,
      is_stop: t.isStop,
      like_num: t.likeNum,
      morph_tense: t.morph.Tense,
    }));

    const words: TokenData[] = tokens.filter((tk) => tk.pos !== "PUNCT" && tk.pos !== "SYM");

    const entities: EntityData[] = booknlpResult.entities.map((e) => ({
      start_token: e.startToken,
      end_token: e.endToken,
      cat: e.cat,
      text: e.text,
      prop: e.prop,
      coref: e.coref,
    }));

    const noun_chunks: NounChunkData[] = booknlpResult.nounChunks.map((n) => ({
      text: n.text,
      start: n.start,
      end: n.end,
      length: n.end - n.start + 1,
    }));

    const supersense: SupersenseEntry[] = booknlpResult.supersense;

    const buildTokenTree = (spacyToken: any): SentenceToken => {
      const pos = spacyToken.pos_;
      const dep = spacyToken.dep_;
      const node = new SentenceToken(spacyToken.text, pos, dep, []);
      const children = spacyToken.children || [];
      for (const c of children) {
        node.children.push(buildTokenTree(c));
      }
      return node;
    };

    const sents: SentenceData[] = booknlpResult.sents.map((s: any) => {
      const root = buildTokenTree(s.root);
      return new SentenceData(root);
    });

    return new ExtCtx(text, tokens, words, noun_chunks, entities, supersense, sents);
  }
}

class WorkerPool {
  workers: Worker[];
  next = 0;

  constructor(n: number) {
    this.workers = Array.from({ length: n }, () => new Worker(new URL("../workers/spacy-worker.ts", import.meta.url), { type: "module" }));
  }

  request(action: "fetchJson"|"fetchProto", url: string, texts: string[]): Promise<any[]> {
    const worker = this.workers[this.next];
    this.next = (this.next + 1) % this.workers.length;
    const id = Math.random().toString(36).slice(2);
    return new Promise((resolve, reject) => {
      const onmsg = (ev: MessageEvent) => {
        const d = ev.data as any;
        if (d.id !== id) return;
        worker.removeEventListener("message", onmsg);
        if (d.ok) resolve(d.contexts);
        else reject(new Error(d.error || "worker error"));
      };
      worker.addEventListener("message", onmsg);
      worker.postMessage({ id, action, url, texts });
    });
  }

  terminate() {
    for (const w of this.workers) w.terminate();
  }
}

class FeatureExtractorPipeline {
  bookNLP: BookNLP;
  spacyCtxUrl: string;
  // Cache maps a batch-hash -> Promise resolving to an array of SpaCyContext
  spacyCtxCache: Record<string, Promise<SpaCyContext[]>> = {};
  spacyWorkerPool: WorkerPool | null = null;
  ENG_POS_TAGS: string[];
  ENG_POS_TAG_MAP: Record<string, number>;
  ENTITY_CATEGORY_MAP: Record<string, number>;
  SUPERSENSE_LABELS: string[];
  SUPERSENSE_LABEL_MAP: Record<string, number>;
  FEATURE_COUNT = 3671;
  EXTRACTOR_FEATURE_COUNTS: Record<string, number>;
  CONTENT_UD_POS_TAGS: Set<string>;
  PUNCTUATION_SYMBOL_MAP: Record<string, number>;
  _depTreeNgramMap: Map<string, number>;
  _nodeFeaturesEnd = 0;
  _relationFeaturesEnd = 0;
  _depTreeFeaturesTotal = 0;
  _pos_features_map: Record<string, number> = {};
  _placesTrie: MultiWordExpressionTrie;
  _multiwordTrie: MultiWordExpressionTrie;
  _prepCompiledRegexPatterns: Array<[RegExp, number]> = [];
  _prepMultiWordPatterns: Record<string, number> = {};
  _prepExactMatchPatterns: Record<string, [number, number, number]> = {};

  _ready: boolean = false;

  constructor(spacyCtxUrl: string) {
    this.bookNLP = new BookNLP();
    this.spacyCtxUrl = spacyCtxUrl;

    this.ENG_POS_TAGS = [
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

    this.ENG_POS_TAG_MAP = this.ENG_POS_TAGS.reduce((acc, t, i) => {
      acc[t] = i;
      return acc;
    }, {} as Record<string, number>);

    this.ENTITY_CATEGORY_MAP = {
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

    this.SUPERSENSE_LABELS = [
      ...SUPERSENSE_LABELS_NOUNS.map((l) => `noun.${l}`),
      ...SUPERSENSE_LABELS_VERBS.map((l) => `verb.${l}`),
    ];

    this.SUPERSENSE_LABEL_MAP = this.SUPERSENSE_LABELS.reduce((acc, s, i) => {
      acc[s] = i;
      return acc;
    }, {} as Record<string, number>);

    this.EXTRACTOR_FEATURE_COUNTS = {
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

    // Runtime sanity checks: ensure FEATURE_COUNT matches extractor counts sum
    const total = Object.values(this.EXTRACTOR_FEATURE_COUNTS).reduce((a, b) => a + b, 0);
    if (total !== this.FEATURE_COUNT) {
      throw new Error(`FEATURE_COUNT mismatch: ${total} != ${this.FEATURE_COUNT}`);
    }

    this.CONTENT_UD_POS_TAGS = new Set(["NOUN", "PROPN", "VERB", "ADJ", "ADV"]);

    this.PUNCTUATION_SYMBOL_MAP = {
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

    // Build combined dependency-tree ngram map (node ngrams, relation ngrams, complete ngrams)
    this._depTreeNgramMap = new Map<string, number>();
    let offset = 0;
    const nodeEntries = Object.entries(depNodeNgrams as Record<string, number>).sort((a, b) => a[1] - b[1]);
    for (const [k] of nodeEntries) {
      this._depTreeNgramMap.set(k, offset);
      offset += 1;
    }
    this._nodeFeaturesEnd = offset;

    const relEntries = Object.entries(depRelationNgrams as Record<string, number>).sort((a, b) => a[1] - b[1]);
    for (const [k] of relEntries) {
      this._depTreeNgramMap.set(k, offset);
      offset += 1;
    }
    this._relationFeaturesEnd = offset;

    const completeEntries = Object.entries(depCompleteNgrams as Record<string, number>).sort((a, b) => a[1] - b[1]);
    for (const [k] of completeEntries) {
      this._depTreeNgramMap.set(k, offset);
      offset += 1;
    }
    this._depTreeFeaturesTotal = offset;

    // Build a normalized lookup map for POS ngram features so we can match
    // different string representations (tuple-like or underscore-separated).
    for (const [k, v] of Object.entries(posNgrams as Record<string, number>)) {
      this._pos_features_map[k] = v;
      const norm = k.replace(/\s+/g, "").replace(/\(|\)|"|'|\[/g, "").replace(/,/g, "_");
      this._pos_features_map[norm] = v;
      const undersc = k.replace(/[^A-Za-z0-9]+/g, "_").replace(/^_+|_+$/g, "");
      this._pos_features_map[undersc] = v;
    }

    // Build multiword trie for concreteness (single + multiword entries)
    this._multiwordTrie = new MultiWordExpressionTrie();
    for (const [w, v] of Object.entries(concretenessWords)) {
      const expr = casefold(w);
      this._multiwordTrie.addExpression(expr, Number(v), 1);
    }
    for (const [exprRaw, v] of Object.entries(concretenessMulti)) {
      const expr = casefold(exprRaw);
      const nWords = expr.trim().split(/\s+/).filter(Boolean).length || 1;
      this._multiwordTrie.addExpression(expr, Number(v), nWords);
    }

    // Build preposition patterns similar to Python pipeline
    this._prepCompiledRegexPatterns = [];
    this._prepMultiWordPatterns = {};
    this._prepExactMatchPatterns = {};
    if (Array.isArray(concretenessPrepositions) && (concretenessPrepositions as any).length > 0) {
      for (const p of concretenessPrepositions as any) {
        const prep = (p.pattern || p.prep || "").toString();
        const imag = Number(p.score ?? p.imag ?? p.value ?? NaN);
        const isRegex = !!p.is_regex || !!p.isRegex;
        const nWords = Number(p.n_words ?? p.nWords ?? (prep.trim().split(/\s+/).filter(Boolean).length || 1));
        const pos_adp = p.pos_adp !== undefined ? Number(p.pos_adp) : NaN;
        const pos_nonadp = p.pos_nonadp !== undefined ? Number(p.pos_nonadp) : NaN;

        if (isRegex) {
          const flags = "i"; // case-insensitive like Python regex.IGNORECASE
          this._prepCompiledRegexPatterns.push([new RegExp(prep, flags), imag]);
        } else if (nWords > 1) {
          const key = casefold(prep);
          this._prepMultiWordPatterns[key] = imag;
        } else {
          const key = casefold(prep);
          this._prepExactMatchPatterns[key] = [imag, pos_adp, pos_nonadp];
        }
      }
    }

    this._placesTrie = new MultiWordExpressionTrie();
    for (const p of concretenessPlaces) {
      const expr = casefold((p || "").toString());
      const nWords = expr.trim().split(/\s+/).filter(Boolean).length || 1;
      this._placesTrie.addExpression(expr, 0, nWords);
    }
  }

  _pythonTupleString(parts: string[]): string {
    if (parts.length === 0) return "()";
    if (parts.length === 1) return `('${parts[0]}',)`;
    return `(${parts.map((p) => `'${p}'`).join(", ")})`;
  }

  async init({progressCallback, provider}: {progressCallback?: ProgressCallback, provider?: ExecutionProvider} = {}): Promise<void> {
    this.spacyWorkerPool = new WorkerPool(4);

    let booknlpProgress = 0;
    let wordnetProgress = 0;
    const report = () => {
      const combined = 0.5 * booknlpProgress + 0.5 * wordnetProgress;
      if (progressCallback) {
        progressCallback(Math.min(Math.max(combined, 0), 1));
      }
    };

    const bookNLPPromise = this.bookNLP.initialize({
      pipeline: ["entity", "supersense", "event"],
      executionProviders: provider ? [provider] : undefined,
      cacheName: CACHE_NAME,
      dtype: "fp32",
    }, (progress) => {
      booknlpProgress = progress;
      report();
    });

    const defaultWn = WORDNETS[0];
    const wordnetPromise = downloadAndLoadWordNet(defaultWn.id, defaultWn.downloadUrl, CACHE_NAME, (progress) => {
      wordnetProgress = progress;
      report();
    });

    await Promise.all([bookNLPPromise, wordnetPromise]);

    this._ready = true;
  }

  /**
   * Count characters between properly closed quotes only.
   * Time complexity: O(n)
   */
  extractQuoteRatio(text: string): number {
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
   * Extract word length distribution.
   * Returns 15-dimensional vector where each dimension represents
   * relative frequency of words with that character length.
   */
  extractWordLengthByChar(words: TokenData[]): number[] {
    const DIM = 15;
    const lengthCounts = new Array(DIM).fill(0);

    if (words.length === 0) {
      return lengthCounts;
    }

    for (const word of words) {
      const len = word.text.length;
      const binIdx = Math.min(len - 1, DIM - 1);
      lengthCounts[binIdx]++;
    }

    return lengthCounts.map((count) => count / words.length);
  }

  /**
   * Extract word length by n-grams (3-grams).
   * Returns 15-dimensional vector of normalized lengths.
   */
  extractNgramWordLengthByChar(words: TokenData[]): number[] {
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
        .reduce((sum, w) => sum + w.text.length, 0);
      const binIdx = Math.min(Math.floor(ngramLength / 3), DIM - 1);
      lengthCounts[binIdx]++;
    }

    return lengthCounts.map((count) => count / totalNgrams);
  }

  extractSentenceLengthByWord(ctx: ExtCtx | null): number[] {
    const BIN_SIZE = 3;
    const UPPER_SENTENCE_LENGTH = 78;
    const DIM = UPPER_SENTENCE_LENGTH / BIN_SIZE; // 26
    const lengthCounts = new Array(DIM).fill(0);

    if (!ctx || !Array.isArray(ctx.words) || ctx.words.length === 0) {
      return [...lengthCounts, 0];
    }

    let cur_sent_idx = 0;
    let cur_sent_length = 0;
    let total_length = 0;

    for (const token of ctx.words) {
      const sent_idx = token.sentence_id;
      if (sent_idx === cur_sent_idx) {
        cur_sent_length += 1;
      } else {
        total_length += cur_sent_length;
        const bin_idx = Math.floor(cur_sent_length / BIN_SIZE);
        if (bin_idx >= DIM) {
          lengthCounts[DIM - 1] += 1;
        } else {
          lengthCounts[bin_idx] += 1;
        }
        cur_sent_idx = sent_idx;
        cur_sent_length = 1;
      }
    }

    // finalize last sentence
    total_length += cur_sent_length;
    const bin_idx = Math.floor(cur_sent_length / BIN_SIZE);
    if (bin_idx >= DIM) {
      lengthCounts[DIM - 1] += 1;
    } else {
      lengthCounts[bin_idx] += 1;
    }

    const num_sents = cur_sent_idx + 1;
    const rel_freq = lengthCounts.map((c) => (num_sents > 0 ? c / num_sents : 0));
    const avg_sent_length = Math.min(total_length / (num_sents || 1) / UPPER_SENTENCE_LENGTH, 1.0);
    return [...rel_freq, avg_sent_length];
  }

  /**
   * Count numeric words in text.
   */
  extractNumericWordRatio(words: TokenData[]): number {
    if (words.length === 0) return 0;

    const numericCount = words.filter((word) =>
      /^\d+(\.\d+)?$/.test(word.text)
    ).length;

    return numericCount / words.length;
  }

  /**
   * Calculate type-token ratio (unique words / total words).
   */
  extractTTR(words: TokenData[]): number {
    if (words.length === 0) return 0;

    const uniqueWords = new Set(words.map((w) => casefold(w.text)));
    return uniqueWords.size / words.length;
  }

  extractLexicalDensity(ctx: ExtCtx | null): number {
    // Required ctx: `words` (non-punctuation tokens)
    if (!ctx || !Array.isArray(ctx.words) || ctx.words.length === 0) return 0;

    // Count tokens whose UD POS is in CONTENT_UD_POS_TAGS
    const contentWordCount = ctx.words.filter((token) => this.CONTENT_UD_POS_TAGS.has(token.pos)).length;
    return contentWordCount / ctx.words.length;
  }

  /**
   * Extract syllable count ratios.
   */
  extractSyllableRatios(words: TokenData[]): number[] {
    if (words.length === 0) {
      return [0, 0];
    }

    const UPPER_SYLLABLE_COUNT = 8;
    let syllableTotal = 0;
    let threeOrMoreCount = 0;

    for (const word of words) {
      const syllables = estimateSyllables(word.text);
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
  extractStopwords(tokens: TokenData[] | null): number {
    // return 0 so frontend does not guess stopwords.
    if (!tokens || tokens.length === 0) return 0;
    const stopwordCount = tokens.filter((t) => !!t.is_stop).length;
    return stopwordCount / tokens.length;
  }

  /**
   * Extract article ratio (definite and indefinite).
   */
  extractArticles(words: TokenData[]): number[] {
    if (words.length === 0) {
      return [0, 0];
    }

    let definiteCount = 0;
    let indefiniteCount = 0;

    for (const word of words) {
      const lower = casefold(word.text);
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
  extractPunctuation(text: string): number[] {
    const counts = new Array(Object.keys(this.PUNCTUATION_SYMBOL_MAP).length).fill(0);

    if (text.length === 0) {
      return counts;
    }

    for (const char of text) {
      if (char in this.PUNCTUATION_SYMBOL_MAP) {
        counts[this.PUNCTUATION_SYMBOL_MAP[char]]++;
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
  extractContractions(tokens: TokenData[] | null): number {
    // Match Python behaviour: fraction of words containing an apostrophe
    if (!tokens || tokens.length === 0) return 0;
    const contractionCount = tokens.filter((t) => (t.text || "").includes("'")).length;
    return contractionCount / tokens.length;
  }

  /**
   * Extract casing features (position × casing).
   */
  extractCasing(tokens: TokenData[] | null): number[] {
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
      if (token.within_sentence_id === 0) {
        positionIdx = 0; // first
      } else if (i === tokens.length - 1 || (tokens[i + 1] && tokens[i + 1].within_sentence_id === 0)) {
        positionIdx = 2; // last
      }

      counts[positionIdx * 3 + casingType]++;
    }

    return counts.map((c) => c / tokens.length);
  }

  /**
   * Extract casing bigrams.
   */
  extractCasingBigrams(words: TokenData[]): number[] {
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
      const casingType1 = getCasingType(words[i].text || "");
      const casingType2 = getCasingType(words[i + 1].text || "");
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

  extractPOSFrequency(
    tokens: TokenData[] | null
  ): number[] {
    const counts = new Array(this.ENG_POS_TAGS.length).fill(0);
    if (!tokens || tokens.length === 0) return counts;

    let total = 0;
    for (const t of tokens) {
      const tag = (t.fine_pos || t.pos || "").toString();
      if (tag in this.ENG_POS_TAG_MAP) {
        counts[this.ENG_POS_TAG_MAP[tag]]++;
      }
      total++;
    }

    if (total === 0) return counts;
    return counts.map((c) => c / total);
  }

  extractPOSNgrams(tokens: TokenData[] | null): number[] {
    const DIM = 995;
    const counts = new Array(DIM).fill(0);
    if (!tokens || tokens.length === 0) return counts;

    // Use preloaded pos ngram map if available
    const seq = tokens.map((t) => (t.fine_pos || t.pos || ""));

    // Try to use map for matching ngrams up to length 3
    const tryMatch = (ng: string) => {
      if (!this._pos_features_map) return -1;
      if (ng in this._pos_features_map) return this._pos_features_map[ng];
      const norm = ng.replace(/\s+/g, "").replace(/\(|\)|"|'|\[/g, "").replace(/,/g, "_");
      if (norm in this._pos_features_map) return this._pos_features_map[norm];
      const undersc = ng.replace(/[^A-Za-z0-9]+/g, "_").replace(/^_+|_+$/g, "");
      if (undersc in this._pos_features_map) return this._pos_features_map[undersc];
      return -1;
    };

    const ngramList: number[] = [];
    for (let n = 1; n <= 3; n++) {
      for (let i = 0; i + n <= seq.length; i++) {
        const ng = seq.slice(i, i + n).join(",");
        const idx = tryMatch(ng);
        if (idx >= 0) ngramList.push(idx);
      }
    }

    if (ngramList.length === 0) return counts;
    for (const idx of ngramList) counts[idx]++;
    const total = ngramList.length;
    return counts.map((c) => c / total);
  }

  extractDependencyTreeStructure(
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

    const walk = (node: any, depth: number) => {
      totalDepth += depth;
      totalNodes++;
      maxDepth = Math.max(maxDepth, depth);
      totalBranching += (node.children || []).length;
      for (const c of node.children || []) walk(c, depth + 1);
    };

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

  extractDependencyTreeRelations(
    sentences: SentenceData[] | null
  ): number[] {
    const DIM = this._depTreeFeaturesTotal || 1319;
    const out = new Array(DIM).fill(0);
    if (!sentences || sentences.length === 0) return out;

    const getAscendingPaths = (
      token: SentenceToken,
      pathNodes: string[],
      pathRels: string[],
      visited: Set<SentenceToken>
    ) => {
      if (visited.has(token)) return;
      visited.add(token);
      pathNodes.push(token.pos || "");

      const pathLen = pathNodes.length;
      // node n-grams (2-4)
      for (let n = 2; n <= Math.min(4, pathLen); n++) {
        const ng = pathNodes.slice(pathLen - n, pathLen);
        const key = this._pythonTupleString(ng);
        const idx = this._depTreeNgramMap.get(key);
        if (idx !== undefined) out[idx]++;
      }

      const relLen = pathRels.length;
      // relation n-grams (1-4)
      for (let n = 1; n <= Math.min(4, relLen); n++) {
        const ng = pathRels.slice(relLen - n, relLen);
        const key = this._pythonTupleString(ng);
        const idx = this._depTreeNgramMap.get(key);
        if (idx !== undefined) out[idx]++;
      }

      // complete n-grams (node-rel-node alternating) (2-4 nodes -> length 2-4)
      for (let n = 2; n <= Math.min(4, pathLen); n++) {
        if (relLen >= n - 1) {
          const completePath: string[] = [];
          for (let i = 0; i < n; i++) {
            completePath.push(pathNodes[pathLen - n + i]);
            if (i < n - 1) {
              completePath.push(pathRels[relLen - n + 1 + i]);
            }
          }
          const key = this._pythonTupleString(completePath);
          const idx = this._depTreeNgramMap.get(key);
          if (idx !== undefined) out[idx]++;
        }
      }

      for (const child of token.children || []) {
        pathRels.push(child.dep || "");
        getAscendingPaths(child, pathNodes, pathRels, visited);
        pathRels.pop();
      }

      pathNodes.pop();
      visited.delete(token);
    };

    for (const s of sentences) {
      if (!s || !s.root) continue;
      getAscendingPaths(s.root, [], [], new Set());
    }

    // normalize per-section as in Python implementation
    const nodeSum = out.slice(0, this._nodeFeaturesEnd).reduce((a, b) => a + b, 0);
    if (nodeSum > 0) {
      for (let i = 0; i < this._nodeFeaturesEnd; i++) out[i] = out[i] / nodeSum;
    }

    const relationSum = out.slice(this._nodeFeaturesEnd, this._relationFeaturesEnd).reduce((a, b) => a + b, 0);
    if (relationSum > 0) {
      for (let i = this._nodeFeaturesEnd; i < this._relationFeaturesEnd; i++) out[i] = out[i] / relationSum;
    }

    const completeSum = out.slice(this._relationFeaturesEnd).reduce((a, b) => a + b, 0);
    if (completeSum > 0) {
      for (let i = this._relationFeaturesEnd; i < out.length; i++) out[i] = out[i] / completeSum;
    }

    return out;
  }

  extractNounPhraseLengths(words: TokenData[]): number[] {
    // Returns 14-dimensional vector of noun-phrase length distribution
    const DIM = 14;
    const counts = new Array(DIM).fill(0);
    if (words.length === 0) return counts;

    // approximate by grouping adjacent capitalized tokens as NPs
    let i = 0;
    const n = words.length;
    while (i < n) {
      if (words[i].text[0] === words[i].text[0]?.toUpperCase()) {
        let j = i + 1;
        while (j < n && words[j].text[0] === words[j].text[0]?.toUpperCase()) j++;
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

  extractEntityCategories(ctx: ExtCtx): number[] {
    const counts = new Array(Object.keys(this.ENTITY_CATEGORY_MAP).length).fill(0);

    if (!ctx || !Array.isArray(ctx.entities) || !Array.isArray(ctx.words) || ctx.words.length === 0) {
      return counts;
    }

    const seenCorefs = new Set<any>();
    const entities = ctx.entities as any[];
    const total = ctx.words.length;

    for (let i = 0; i < entities.length; i++) {
      const ent = entities[i];
      const coref = ent && ent.coref !== undefined ? ent.coref : `__ent_${i}`;
      if (seenCorefs.has(coref)) continue;
      seenCorefs.add(coref);
      const cat = ent && ent.cat;
      if (typeof cat === "string" && cat in this.ENTITY_CATEGORY_MAP) {
        counts[this.ENTITY_CATEGORY_MAP[cat]]++;
      }
    }

    return counts.map((c) => c / (total || 1));
  }

  extractEvents(tokens: TokenData[] | null): number {
    if (!tokens || tokens.length === 0) return 0;
    const ev = tokens.filter((t) => !!t.event).length;
    return ev / tokens.length;
  }

  extractSupersense(ctx: ExtCtx): number[] {
    const counts = new Array(this.SUPERSENSE_LABELS.length).fill(0);

    if (!ctx || !Array.isArray((ctx as any).supersense) || !Array.isArray((ctx as any).words) || (ctx as any).words.length === 0) {
      return counts;
    }

    const total = (ctx as any).words.length;
    for (const ann of (ctx as any).supersense) {
      // annotations expected as tuples like [start, end, label, score]
      const label = Array.isArray(ann) && ann.length >= 3 ? ann[2] : ann;
      if (typeof label === "string" && label in this.SUPERSENSE_LABEL_MAP) {
        counts[this.SUPERSENSE_LABEL_MAP[label]]++;
      }
    }

    return counts.map((c) => c / (total || 1));
  }

  extractTense(tokens: TokenData[] | null): number[] {
    if (!tokens || tokens.length === 0) return [0, 0, 0, 0];
    let past = 0;
    let present = 0;
    let future = 0;
    let none = 0;
    for (const t of tokens) {
      const tense = casefold(t.morph_tense || "");
      if (tense.includes("past")) past++;
      else if (tense.includes("pres")) present++;
      else if (tense.includes("fut")) future++;
      else none++;
    }
    const total = tokens.length || 1;
    return [past / total, present / total, future / total, none / total];
  }

  extractPolysemy(words: TokenData[]): number[] {
    const DIM = 15;
    const counts = new Array(DIM).fill(0);
    if (!words || words.length === 0) return counts;

    for (const w of words) {
      const n = getPolysemyCount(w.text);
      const idx = Math.min(Math.max(0, Math.floor(n)), DIM - 1);
      counts[idx] += 1;
    }

    return counts.map((c) => c / words.length);
  }

  extractWordConcreteness(ctx: ExtCtx): number[] {
    const BINS = 20;
    const out = new Array(BINS + 1).fill(0);
    if (!Array.isArray(ctx.tokens) || !Array.isArray(ctx.words) || ctx.tokens.length === 0) {
      return out;
    }

    // Use the prebuilt multiword trie to find longest matches (start,end inclusive,score)
    const matches = this._multiwordTrie.findLongestMatches(
      ctx.tokens.map((t) => ({ text: t.text, lemma: t.lemma || t.itext || t.text }))
    );

    if (!matches || matches.length === 0) return out;

    const occurences = new Array(BINS).fill(0);
    let totalScore = 0;
    for (const [, , score] of matches) {
      const idx = Math.min(Math.floor(score * BINS), BINS - 1);
      occurences[idx] += 1;
      totalScore += score;
    }

    // multiword_reduction = sum(end - start) (end inclusive)
    let multiwordReduction = 0;
    for (const [start, end] of matches) multiwordReduction += (end - start);

    const effectiveWordCount = Math.max(1, ctx.words.length - multiwordReduction);

    for (let j = 0; j < BINS; j++) out[j] = occurences[j] / effectiveWordCount;
    out[BINS] = totalScore / matches.length;
    return out;
  }

  extractPrepositionImageability(ctx: ExtCtx): number[] {
    const BINS = 10;
    const out = new Array(BINS + 1).fill(0);
    if (!Array.isArray(ctx.tokens) || !Array.isArray(ctx.words) || ctx.tokens.length === 0) return out;

    const tokens = ctx.tokens;
    const imageabilityValues: number[] = [];
    const matchedPositions = new Set<number>();

    // Multiword exact matches (longest-first behavior handled by marking positions)
    for (const [prepPhrase, imagVal] of Object.entries(this._prepMultiWordPatterns)) {
      const phraseWords = (prepPhrase || "").split(/\s+/).filter(Boolean);
      const phraseLen = phraseWords.length;
      for (let i = 0; i <= tokens.length - phraseLen; i++) {
        if (matchedPositions.has(i)) continue;
        let ok = true;
        for (let j = 0; j < phraseLen; j++) {
          const tok = tokens[i + j];
          const tokText = (tok.itext || tok.text || "").toString();
          if (tokText !== phraseWords[j]) {
            ok = false;
            break;
          }
        }
        if (!ok) continue;
        // ensure no overlap
        let overlap = false;
        for (let pos = i; pos < i + phraseLen; pos++) if (matchedPositions.has(pos)) { overlap = true; break; }
        if (overlap) continue;
        imageabilityValues.push(imagVal);
        for (let pos = i; pos < i + phraseLen; pos++) matchedPositions.add(pos);
      }
    }

    // Singleword regex
    for (const [regexObj, imagVal] of this._prepCompiledRegexPatterns) {
      for (let i = 0; i < tokens.length; i++) {
        if (matchedPositions.has(i)) continue;
        const tok = tokens[i];
        const txt = (tok.itext || tok.text || "").toString();
        if (regexObj.test(txt)) {
          imageabilityValues.push(imagVal);
          matchedPositions.add(i);
        }
      }
    }

    // Singleword exact matches (itext or lemma)
    for (let i = 0; i < tokens.length; i++) {
      if (matchedPositions.has(i)) continue;
      const token = tokens[i];
      const itext = (token.itext || token.text || "").toString();
      const lemma = (token.lemma || "").toString();
      const key = itext in this._prepExactMatchPatterns ? itext : lemma in this._prepExactMatchPatterns ? lemma : null;
      if (key === null) continue;
      const [val, pos_adp, pos_nonadp] = this._prepExactMatchPatterns[key];
      let finalVal: number | null = null;
      if (token.pos === "ADP" && !Number.isNaN(pos_adp)) finalVal = pos_adp;
      else if (token.pos !== "ADP" && !Number.isNaN(pos_nonadp)) finalVal = pos_nonadp;
      else if (Number.isNaN(pos_adp) && Number.isNaN(pos_nonadp)) finalVal = val;
      if (finalVal !== null) {
        imageabilityValues.push(finalVal);
        matchedPositions.add(i);
      }
    }

    if (imageabilityValues.length === 0) return out;

    const histCounts = new Array(BINS).fill(0);
    let total = 0;
    for (const imagVal of imageabilityValues) {
      const binIdx = Math.min(Math.floor(imagVal * BINS), BINS - 1);
      histCounts[binIdx]++;
      total += imagVal;
    }

    const wordCount = Math.max(1, ctx.words.length);
    for (let j = 0; j < BINS; j++) out[j] = histCounts[j] / wordCount;
    out[BINS] = total / imageabilityValues.length;
    return out;
  }

  extractPlaces(words: TokenData[]): number {
    if (!words || words.length === 0) return 0;

    const tokens = words.map((w) => ({ text: w.text, lemma: (w.lemma || w.itext || w.text) }));
    const matches = this._placesTrie.findLongestMatches(tokens as any);
    return matches.length / words.length;
  }

  extractCharNgrams(text: string): number[] {
    const DIM = 1000;
    const counts = new Array(DIM).fill(0);
    if (!text) return counts;

    if (!charNgrams) return counts;

    const s = casefold(text);
    // generate char ngrams of length 3..5
    for (let n = 3; n <= 5; n++) {
      for (let i = 0; i + n <= s.length; i++) {
        const ng = s.substring(i, i + n);
        const idx = (charNgrams as any)[ng];
        if (idx !== undefined) counts[idx]++;
      }
    }

    const total = counts.reduce((a, b) => a + b, 0) || 1;
    return counts.map((c) => c / total);
  }

  private _hashBatch = (batch: string[]): string => {
    // Join with a control character to avoid collisions from simple separators
    const s = batch.join("\u0001");
    // djb2-like hashing
    let h = 5381;
    for (let i = 0; i < s.length; i++) {
      h = ((h << 5) + h) ^ s.charCodeAt(i);
    }
    return (h >>> 0).toString(16);
  };

  async fetchSpaCyContexts(texts: string[]): Promise<any[]> {
    if (!this.spacyWorkerPool) {
      throw new Error("SpaCy worker pool not initialized");
    }
    const p = this.spacyCtxUrl.endsWith("/proto")
      ? this.spacyWorkerPool.request("fetchProto", this.spacyCtxUrl, texts)
      : this.spacyWorkerPool.request("fetchJson", this.spacyCtxUrl, texts);
    this.spacyCtxCache[this._hashBatch(texts)] = p;
    return await p;
  }

  /**
   * Prefetch SpaCy contexts into local cache for all batches of texts, one batch at a time
   */
  async prefetchSpaCyContexts(textBatches: string[][]): Promise<void> {
    this.spacyCtxCache = {};
    const concurrency = this.spacyWorkerPool ? Math.max(1, this.spacyWorkerPool.workers.length) : 1;

    // Pause/continue support
    async function waitIfFeatureExtractorPaused() {
      if (!(globalThis as any).__featureExtractorPauseRequested) return;
      if (!(globalThis as any).__featureExtractorPausePromise) {
        (globalThis as any).__featureExtractorPausePromise = new Promise(resolve => {
          (globalThis as any).__featureExtractorPauseResolve = resolve;
        });
      }
      await (globalThis as any).__featureExtractorPausePromise;
    }

    // Maintain up to `concurrency` active requests; start the next as soon as any finishes.
    const active: { id: string; promise: Promise<{ id: string; ok: boolean; err?: any }> }[] = [];

    for (const batch of textBatches) {
      await waitIfFeatureExtractorPaused();
      const key = this._hashBatch(batch);
      // @ts-ignore
      if (this.spacyCtxCache[key]) continue;

      const taskPromise = (async () => {
        const p = this.fetchSpaCyContexts(batch);
        try {
          await p;
          // console.log(`Prefetched SpaCy contexts for batch with hash ${key}`);
          return { id: key, ok: true };
        } catch (err) {
          // remove failed entry so callers can retry
          delete this.spacyCtxCache[key];
          return { id: key, ok: false, err };
        }
      })();

      active.push({ id: key, promise: taskPromise });

      if (active.length >= concurrency) {
        const res = await Promise.race(active.map((a) => a.promise));
        const idx = active.findIndex((a) => a.id === res.id);
        if (idx >= 0) active.splice(idx, 1);
        if (!res.ok) throw res.err;
      }
    }

    // Wait for remaining active tasks, handling each as it finishes.
    while (active.length > 0) {
      await waitIfFeatureExtractorPaused();
      const res = await Promise.race(active.map((a) => a.promise));
      const idx = active.findIndex((a) => a.id === res.id);
      if (idx >= 0) active.splice(idx, 1);
      if (!res.ok) throw res.err;
    }
  }

  /**
   * High-level feature extraction for a single text. Synchronous and uses
   * the individual extractor functions exported in this module. Returns
   * an object with the original text and the assembled feature vector.
   */
  async extract(texts: string[]): Promise<number[][]> {
    if (!this._ready) {
      throw new Error("BookNLP or WordNet not ready");
    }

    const spacyPromise = this.spacyCtxCache[this._hashBatch(texts)];
    const spaCyContexts = spacyPromise ? (await spacyPromise) : await this.fetchSpaCyContexts(texts);

    const results: number[][] = [];
    for (let i = 0; i < texts.length; i++) {
      const text = texts[i];
      const spacyCtx = spaCyContexts[i];
      const ctx = ExtCtx.fromBookNLPResult(await this.bookNLP.process(spacyCtx), text);

      const features: number[] = [];
      features.push(this.extractQuoteRatio(text));
      features.push(...this.extractCharNgrams(text));
      features.push(...this.extractWordLengthByChar(ctx.words));
      features.push(...this.extractNgramWordLengthByChar(ctx.words));
      features.push(...this.extractSentenceLengthByWord(ctx));
      features.push(this.extractNumericWordRatio(ctx.words));
      features.push(this.extractTTR(ctx.words));
      features.push(this.extractLexicalDensity(ctx));
      features.push(...this.extractSyllableRatios(ctx.words));
      features.push(this.extractStopwords(ctx.tokens));
      features.push(...this.extractArticles(ctx.words));
      features.push(...this.extractPunctuation(text));
      features.push(this.extractContractions(ctx.tokens));
      features.push(...this.extractCasing(ctx.tokens));
      features.push(...this.extractCasingBigrams(ctx.words));
      features.push(...this.extractPOSFrequency(ctx.tokens));
      features.push(...this.extractPOSNgrams(ctx.tokens));
      features.push(...this.extractDependencyTreeStructure(ctx.sents));
      features.push(...this.extractDependencyTreeRelations(ctx.sents));
      features.push(...this.extractNounPhraseLengths(ctx.words));
      features.push(...this.extractEntityCategories(ctx));
      features.push(this.extractEvents(ctx.tokens));
      features.push(...this.extractSupersense(ctx));
      features.push(...this.extractTense(ctx.tokens));
      features.push(...this.extractPolysemy(ctx.words));
      features.push(...this.extractWordConcreteness(ctx));
      features.push(...this.extractPrepositionImageability(ctx));
      features.push(this.extractPlaces(ctx.words));
      results.push(features);
    }
    return results;
  }
}

export class FeatureService {
  featureExtractor: FeatureExtractorPipeline;
  embedMiniLM: FeatureExtractionPipeline | null = null;

  constructor(spacyCtxUrl: string) {
    this.featureExtractor = new FeatureExtractorPipeline(spacyCtxUrl);
  }

  async init(embeddingPipelineConfig: HFPipelineConfig, {progressCallback, provider, }: {progressCallback?: ProgressCallback, provider?: ExecutionProvider} = {}): Promise<void> {
    // Track each part's progress and report combined progress (0..1).
    let extractorProgress = 0;
    let pipelineProgress = 0;
    const report = () => {
      const combined = 0.5 * extractorProgress + 0.5 * pipelineProgress;
      if (progressCallback) {
        progressCallback(Math.min(Math.max(combined, 0), 1));
      }
    };

    // Start both initialization promises concurrently and await them together.
    const extractorPromise = this.featureExtractor.init({ progressCallback: (progress) => {
      extractorProgress = Math.min(Math.max(progress ?? 0, 0), 1);
      report();
    }, provider });

    // @ts-ignore
    const pipelinePromise = pipeline(
      embeddingPipelineConfig.type,
      embeddingPipelineConfig.model,
      {
        progress_callback: ((data: any) => {
          if (data.progress !== undefined && data.file?.endsWith(".onnx")) {
            pipelineProgress = Math.min(Math.max(data.progress ?? 0, 0), 1);
            report();
          }
        }) as (data: any) => void,
        dtype: embeddingPipelineConfig.dtype,
        device: embeddingPipelineConfig.device as any,
      }
    ) as Promise<FeatureExtractionPipeline>;

    const [embed] = await Promise.all([pipelinePromise, extractorPromise]);
    this.embedMiniLM = embed as FeatureExtractionPipeline;
  }

  async getFeaturesAllBatchPreload(textBatches: string[][]): Promise<void> {
    await this.featureExtractor.prefetchSpaCyContexts(textBatches);
  }

  async getFeatures(texts: string[]): Promise<number[][]> {
    if (!this.embedMiniLM) {
      throw new Error("Pipeline not initialized");
    }

    const [extractorFeatures, embeddingTensor] = await Promise.all([
      this.featureExtractor.extract(texts),
      this.embedMiniLM(texts, { pooling: "mean", normalize: true }),
    ]);
    const embeddings = embeddingTensor.tolist();
    return extractorFeatures.map((featVec, i) => [...embeddings[i], ...featVec]);
  }
}
