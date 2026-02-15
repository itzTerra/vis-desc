import { pipeline, type Pipeline } from "@huggingface/transformers";
import type { TModelInfo, TransformersModelConfig, WordNetInfo } from "~/utils/models";
import { WORDNETS } from "~/utils/models";
import { downloadAndLoadWordNet, clearWordNetCache } from "~/utils/wordnet";

interface ModelLoadState {
  isLoading: boolean;
  isCached: boolean;
  isLoadedInMemory: boolean;
  error: string | null;
  progress: number;
}

type ModelCacheStatus = "cached" | "not-cached" | "downloading";

const TRANSFORMERS_CACHE_NAME = "transformers-cache";

const useModelsState = () => useState<Map<string, ModelLoadState>>("models-state", () => new Map());
const pipelinesCache = new Map<string, Pipeline>();
const workersCache = new Map<string, Worker>();

const getModelKey = (huggingFaceId: string): string => {
  return huggingFaceId;
};

const getWorkerUrl = (workerName: string): URL => {
  const basePath = "../workers";
  const fileName = `${workerName}.worker.ts`;
  return new URL(`${basePath}/${fileName}`, import.meta.url);
};

const getModelState = (huggingFaceId: string): ModelLoadState => {
  const modelsState = useModelsState();
  const key = getModelKey(huggingFaceId);

  if (!modelsState.value.has(key)) {
    modelsState.value.set(key, {
      isLoading: false,
      isCached: false,
      isLoadedInMemory: false,
      error: null,
      progress: 0,
    });
  }
  return modelsState.value.get(key)!;
};

const checkModelCached = async (huggingFaceId: string): Promise<boolean> => {
  try {
    const cache = await caches.open(TRANSFORMERS_CACHE_NAME);
    const keys = await cache.keys();
    const hasModel = keys.some(request =>
      request.url.includes(huggingFaceId)
    );
    return hasModel;
  } catch {
    return false;
  }
};

const syncCacheState = async (huggingFaceId: string): Promise<boolean> => {
  const isCached = await checkModelCached(huggingFaceId);
  const state = getModelState(huggingFaceId);
  state.isCached = isCached;
  return isCached;
};

const downloadModel = async (
  config: TransformersModelConfig,
  onProgress?: (progress: number) => void
) => {
  if (!config) {
    throw new Error("No transformers.js configuration found");
  }

  const state = getModelState(config.huggingFaceId);
  const key = getModelKey(config.huggingFaceId);
  state.isLoading = true;
  state.error = null;

  try {
    const model = await pipeline(config.pipeline, config.huggingFaceId, {
      subfolder: config.subfolder || "",
      model_file_name: config.modelFileName,
      progress_callback: (data: any) => {
        if (data.progress !== undefined && data.file.endsWith(".onnx")) {
          state.progress = data.progress;
          onProgress?.(data.progress);
        }
      },
    });

    pipelinesCache.set(key, model as Pipeline);
    state.isLoadedInMemory = true;
    state.isCached = true;
    state.isLoading = false;
    state.progress = 100;

    return model;
  } catch (error) {
    state.error = error instanceof Error ? error.message : "Failed to load model";
    state.isLoading = false;
    throw error;
  }
};

const getOrLoadModel = async (
  config: TransformersModelConfig,
  options?: {
    forceDownload?: boolean;
    onProgress?: (progress: number) => void;
    onCacheCheck?: (isCached: boolean) => void;
  }
) => {
  if (!config) {
    throw new Error("No transformers.js configuration found");
  }

  const key = getModelKey(config.huggingFaceId);
  const state = getModelState(config.huggingFaceId);

  if (pipelinesCache.has(key) && state.isLoadedInMemory) {
    return pipelinesCache.get(key)!;
  }

  const isCached = !options?.forceDownload && await checkModelCached(config.huggingFaceId);
  options?.onCacheCheck?.(isCached);
  state.isCached = isCached;

  return downloadModel(config, options?.onProgress);
};

const getModelCacheStatus = (config: TransformersModelConfig): ModelCacheStatus => {
  const state = getModelState(config.huggingFaceId);

  if (state.isLoading) return "downloading";
  if (state.isCached) return "cached";
  return "not-cached";
};

const clearModelCache = async (config: TransformersModelConfig) => {
  if (!config) {
    throw new Error("No transformers.js configuration found");
  }

  const key = getModelKey(config.huggingFaceId);
  const modelId = config.huggingFaceId;

  try {
    const cache = await caches.open(TRANSFORMERS_CACHE_NAME);
    const keys = await cache.keys();
    const modelKeys = keys.filter(request => request.url.includes(modelId));

    await Promise.all(modelKeys.map(key => cache.delete(key)));

    const state = getModelState(config.huggingFaceId);
    pipelinesCache.delete(key);
    state.isLoadedInMemory = false;
    state.isCached = false;
    state.progress = 0;
  } catch (error) {
    console.error("Failed to clear model cache:", error);
    throw error;
  }
};

const clearAllCache = async () => {
  try {
    await caches.delete(TRANSFORMERS_CACHE_NAME);

    const modelsState = useModelsState();
    modelsState.value.clear();
  } catch (error) {
    console.error("Failed to clear all models cache:", error);
    throw error;
  }
};

const syncAllCacheState = async () => {
  const models = MODELS.filter(m => m.transformersConfig) as TModelInfo[];
  await Promise.all(models.map(m => syncCacheState(m.transformersConfig.huggingFaceId)));
  // also sync wordnet cache state for known wordnets
  await Promise.all(WORDNETS.map(async (w) => {
    const isCached = await checkModelCached(w.downloadUrl);
    const state = getModelState(w.id);
    state.isCached = isCached;
  }));
};

const getWordNetCacheStatus = (wordnet: WordNetInfo): ModelCacheStatus => {
  const state = getModelState(wordnet.id);

  if (state.isLoading) return "downloading";
  if (state.isCached) return "cached";
  return "not-cached";
};

const getOrLoadWordNet = async (
  wordnet: WordNetInfo,
  options?: {
    forceDownload?: boolean;
    onProgress?: (progress: number) => void;
  }
) => {
  const state = getModelState(wordnet.id);
  if (state.isLoadedInMemory) {
    return; // wordnet is in memory (wordnet module holds data)
  }

  state.isLoading = true;
  try {
    await downloadAndLoadWordNet(wordnet.id, wordnet.downloadUrl, undefined, options?.onProgress);
    state.isLoadedInMemory = true;
    state.isCached = true;
    state.isLoading = false;
    state.progress = 100;
  } catch (err) {
    state.error = err instanceof Error ? err.message : String(err);
    state.isLoading = false;
    throw err;
  }
};

const clearWordNetFromCache = async (wordnet: WordNetInfo) => {
  await clearWordNetCache(wordnet.downloadUrl);
  const state = getModelState(wordnet.id);
  state.isCached = false;
  state.isLoadedInMemory = false;
  state.progress = 0;
};

const loadModelInWorker = async (
  config: TransformersModelConfig,
  worker: Worker,
  onProgress?: (progress: number) => void
): Promise<void> => {
  return new Promise((resolve, reject) => {
    const messageHandler = (event: MessageEvent) => {
      const { type, payload } = event.data;

      switch (type) {
      case "ready":
        worker.removeEventListener("message", messageHandler);
        resolve();
        break;
      case "error":
        worker.removeEventListener("message", messageHandler);
        reject(new Error(payload.message));
        break;
      case "progress":
        onProgress?.(payload.progress);
        break;
      }
    };

    worker.addEventListener("message", messageHandler);

    worker.postMessage({
      type: "load",
      payload: {
        ...config,
      }
    });
  });
};

const getOrLoadWorker = async (
  config: TransformersModelConfig,
  options?: {
    onProgress?: (progress: number) => void;
  }
): Promise<Worker> => {
  if (!config) {
    throw new Error("No transformers.js configuration found");
  }

  const key = getModelKey(config.huggingFaceId);
  const state = getModelState(config.huggingFaceId);

  if (workersCache.has(key)) {
    return workersCache.get(key)!;
  }

  state.isLoading = true;
  state.error = null;

  try {
    const workerUrl = getWorkerUrl(config.workerName);
    const worker = new Worker(workerUrl, { type: "module" });

    await loadModelInWorker(config, worker, options?.onProgress);

    workersCache.set(key, worker);
    state.isLoadedInMemory = true;
    state.isLoading = false;

    return worker;
  } catch (error) {
    state.error = error instanceof Error ? error.message : "Failed to load worker";
    state.isLoading = false;
    throw error;
  }
};

const terminateWorker = (huggingFaceId: string) => {
  const key = getModelKey(huggingFaceId);
  const worker = workersCache.get(key);

  if (worker) {
    worker.terminate();
    workersCache.delete(key);

    const state = getModelState(huggingFaceId);
    state.isLoadedInMemory = false;
  }
};

export const useModelLoader = () => {
  return {
    getOrLoadModel,
    getOrLoadWorker,
    terminateWorker,
    getModelCacheStatus,
    clearModelCache,
    clearAllCache,
    syncAllCacheState,
    // WordNet helpers
    getOrLoadWordNet,
    getWordNetCacheStatus,
    clearWordNetFromCache,
  };
};
