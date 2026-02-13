import { pipeline, env, type Pipeline, type FeatureExtractionPipeline } from "@huggingface/transformers";
import type {
  LoadMessage,
  WorkerMessage,
  ExtractMessage,
  FeatureExtractionResult,
  ExtractProgressPayload,
} from "~/types/text2features-worker";
import * as featureExtractors from "~/utils/text2features";
import { MultiWordExpressionTrie } from "~/utils/multiword-trie";

env.allowLocalModels = false;
env.allowRemoteModels = true;
env.useBrowserCache = true;

let embeddingPipeline: FeatureExtractionPipeline | null = null;
const trie = new MultiWordExpressionTrie();

// Tokenization and feature assembly live in utils/text2features.ts

/**
 * Load the embedding model.
 */
async function loadModel(config: LoadMessage["payload"]): Promise<void> {
  try {
    // The pipeline helper's return type can produce an extremely complex
    // union that TypeScript sometimes cannot represent. Ignore the check
    // here and assert the expected pipeline shape via a cast below.
    // eslint-disable-next-line @typescript-eslint/ban-ts-comment
    // @ts-ignore
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
  // Let the utility module build the feature vector
  const { features } = await featureExtractors.extractFeaturesFromText(text);
  const embeddings = await getEmbeddings(text);

  return {
    text,
    features,
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
