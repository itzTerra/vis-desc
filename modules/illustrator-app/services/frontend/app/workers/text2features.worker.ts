import { pipeline, env, type Pipeline, type FeatureExtractionPipeline } from "@huggingface/transformers";
import type {
  LoadMessage,
  WorkerMessage,
  ExtractMessage,
  FeatureExtractionResult,
  ExtractProgressPayload,
} from "~/types/text2features-worker";
import * as featureExtractors from "~/utils/feature-extractors";
import { MultiWordExpressionTrie } from "~/utils/multiword-trie";

env.allowLocalModels = false;
env.allowRemoteModels = true;
env.useBrowserCache = true;

let embeddingPipeline: FeatureExtractionPipeline | null = null;
const trie = new MultiWordExpressionTrie();

/**
 * Split text into words for feature extraction.
 * Simple whitespace-based tokenization.
 */
function tokenizeSimple(text: string): string[] {
  return text
    .toLowerCase()
    .split(/\s+/)
    .filter((word) => word.length > 0);
}

/**
 * Load the embedding model.
 */
async function loadModel(config: LoadMessage["payload"]): Promise<void> {
  try {
    embeddingPipeline = await pipeline(
      "feature-extraction",
      config.huggingFaceId,
      {
        progress_callback: ((data: any) => {
          if (data.progress !== undefined && data.file?.endsWith(".onnx")) {
            self.postMessage({
              type: "progress",
              payload: {
                progress: data.progress,
              },
            });
          }
        }) as (data: any) => void,
        dtype: "q8",
        device: "wasm",
      }
    ) as FeatureExtractionPipeline;

    self.postMessage({
      type: "ready",
      payload: { success: true },
    });
  } catch (error) {
    self.postMessage({
      type: "error",
      payload: { message: (error as Error).message },
    });
  }
}

/**
 * Extract features from a single text.
 */
async function extractFeaturesFromText(
  text: string
): Promise<FeatureExtractionResult> {
  const preprocessedText = featureExtractors.preprocess(text);
  const words = tokenizeSimple(preprocessedText);

  // Get embeddings
  const embeddings = await getEmbeddings(text);

  // Extract features
  const features: number[] = [];

  features.push(...featureExtractors.extractQuoteRatioFeature(text));
  features.push(...featureExtractors.extractCharNgrams(text));
  features.push(...featureExtractors.extractWordLengthByChar(words));
  features.push(...featureExtractors.extractNgramWordLengthByChar(words));

  // Sentence length features would go here (requires sentence tokenization)
  features.push(...new Array(27).fill(0)); // 26 + 1 for avg

  features.push(featureExtractors.extractNumericWordRatio(words));
  features.push(featureExtractors.extractTTR(words));
  features.push(featureExtractors.extractLexicalDensity(words));
  features.push(...featureExtractors.extractSyllableRatios(words));
  features.push(featureExtractors.extractStopwords(words));
  features.push(...featureExtractors.extractArticles(words));
  features.push(...featureExtractors.extractPunctuation(text));
  features.push(featureExtractors.extractContractions(words));
  features.push(...featureExtractors.extractCasing(words));
  features.push(...featureExtractors.extractCasingBigrams(words));

  // POS and dependency features (requires BookNLP)
  features.push(...featureExtractors.extractPOSFrequency(null));
  features.push(...featureExtractors.extractPOSNgrams(null));
  features.push(...featureExtractors.extractDependencyTreeStructure(null));
  features.push(...featureExtractors.extractDependencyTreeRelations(null));

  // Noun phrase and entity features (requires booknlp)
  features.push(...featureExtractors.extractNounPhraseLengths(words));
  features.push(...featureExtractors.extractEntityCategories(null, words));
  features.push(featureExtractors.extractEvents(null));
  features.push(...featureExtractors.extractSupersense(null, words));
  features.push(...featureExtractors.extractTense(null));
  features.push(...featureExtractors.extractPolysemy(words));
  features.push(...featureExtractors.extractWordConcreteness(words));
  features.push(...featureExtractors.extractPrepositionImageability());
  features.push(featureExtractors.extractPlaces(words));

  return {
    text,
    features: features.map((f) => (isNaN(f) ? 0 : f)),
    embeddings,
  };
}

/**
 * Get embeddings for text using the loaded model.
 */
async function getEmbeddings(text: string): Promise<number[]> {
  if (!embeddingPipeline) {
    throw new Error("Embedding pipeline not loaded");
  }

  const output = (await embeddingPipeline(text, {
    pooling: "mean",
    normalize: true,
  })) as any;

  if (Array.isArray(output)) {
    return Array.from(output);
  }

  if ("data" in output) {
    return Array.from(output.data);
  }

  return [];
}

/**
 * Process batch of texts and extract features.
 */
async function extractSegments(data: ExtractMessage["payload"]): Promise<void> {
  if (!embeddingPipeline) {
    self.postMessage({
      type: "error",
      payload: { message: "Pipeline not loaded" },
    });
    return;
  }

  const { texts, batchSize } = data;
  const totalBatches = Math.ceil(texts.length / batchSize);

  try {
    for (let i = 0; i < texts.length; i += batchSize) {
      const batchTexts = texts.slice(i, i + batchSize);
      const batchResults: FeatureExtractionResult[] = [];

      for (const text of batchTexts) {
        const result = await extractFeaturesFromText(text);
        batchResults.push(result);
      }

      const progressPayload: ExtractProgressPayload = {
        batchIndex: Math.floor(i / batchSize) + 1,
        totalBatches,
        results: batchResults,
      };

      self.postMessage({
        type: "progress",
        payload: progressPayload,
      });
    }

    self.postMessage({
      type: "complete",
      payload: { success: true },
    });
  } catch (error) {
    self.postMessage({
      type: "error",
      payload: { message: (error as Error).message },
    });
  }
}

/**
 * Handle incoming worker messages.
 */
self.onmessage = async (event: MessageEvent<WorkerMessage>) => {
  switch (event.data.type) {
  case "load":
    await loadModel((event.data as LoadMessage).payload);
    break;
  case "extract":
    await extractSegments((event.data as ExtractMessage).payload);
    break;
  }
};
