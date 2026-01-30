import { pipeline, type Pipeline } from "@huggingface/transformers";
import type { TModelInfo } from "~/utils/models";

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

const getModelKey = (modelInfo: TModelInfo): string => {
  return modelInfo.transformersConfig.huggingFaceId;
};

const getModelState = (modelInfo: TModelInfo): ModelLoadState => {
  const modelsState = useModelsState();
  const key = getModelKey(modelInfo);

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

const checkModelCached = async (modelInfo: TModelInfo): Promise<boolean> => {
  try {
    const cache = await caches.open(TRANSFORMERS_CACHE_NAME);
    const keys = await cache.keys();
    const hasModel = keys.some(request =>
      request.url.includes(modelInfo.transformersConfig.huggingFaceId)
    );
    return hasModel;
  } catch {
    return false;
  }
};

const syncCacheState = async (modelInfo: TModelInfo): Promise<boolean> => {
  const isCached = await checkModelCached(modelInfo);
  const state = getModelState(modelInfo);
  state.isCached = isCached;
  return isCached;
};

const downloadModel = async (
  modelInfo: TModelInfo,
  onProgress?: (progress: number) => void
) => {
  const config = modelInfo.transformersConfig;
  if (!config) {
    throw new Error(`No transformers.js configuration found for model: ${modelInfo.id}`);
  }

  const state = getModelState(modelInfo);
  const key = getModelKey(modelInfo);
  state.isLoading = true;
  state.error = null;

  try {
    const model = await pipeline(config.pipeline, config.huggingFaceId, {
      subfolder: "",
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
  modelInfo: TModelInfo,
  options?: {
    forceDownload?: boolean;
    onProgress?: (progress: number) => void;
    onCacheCheck?: (isCached: boolean) => void;
  }
) => {
  const config = modelInfo.transformersConfig;
  if (!config) {
    throw new Error(`No transformers.js configuration found for model: ${modelInfo.id}`);
  }

  const key = getModelKey(modelInfo);
  const state = getModelState(modelInfo);

  if (pipelinesCache.has(key) && state.isLoadedInMemory) {
    return pipelinesCache.get(key)!;
  }

  const isCached = !options?.forceDownload && await checkModelCached(modelInfo);
  options?.onCacheCheck?.(isCached);
  state.isCached = isCached;

  return downloadModel(modelInfo, options?.onProgress);
};

const getModelCacheStatus = (modelInfo: TModelInfo): ModelCacheStatus => {
  const state = getModelState(modelInfo);

  if (state.isLoading) return "downloading";
  if (state.isCached) return "cached";
  return "not-cached";
};

const clearModelCache = async (modelInfo: TModelInfo) => {
  const config = modelInfo.transformersConfig;
  if (!config) {
    throw new Error(`No transformers.js configuration found for model: ${modelInfo.id}`);
  }

  const key = getModelKey(modelInfo);
  const modelId = config.huggingFaceId;

  try {
    const cache = await caches.open(TRANSFORMERS_CACHE_NAME);
    const keys = await cache.keys();
    const modelKeys = keys.filter(request => request.url.includes(modelId));

    await Promise.all(modelKeys.map(key => cache.delete(key)));

    const state = getModelState(modelInfo);
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
  await Promise.all(models.map(m => syncCacheState(m)));
};

const loadModelInWorker = async (
  modelInfo: TModelInfo,
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

    const config = modelInfo.transformersConfig;
    worker.postMessage({
      type: "load",
      payload: {
        huggingFaceId: config.huggingFaceId,
        modelFileName: config.modelFileName,
        pipeline: config.pipeline,
      }
    });
  });
};

const getOrLoadWorker = async (
  modelInfo: TModelInfo,
  options?: {
    onProgress?: (progress: number) => void;
  }
): Promise<Worker> => {
  const config = modelInfo.transformersConfig;
  if (!config) {
    throw new Error(`No transformers.js configuration found for model: ${modelInfo.id}`);
  }

  const key = getModelKey(modelInfo);
  const state = getModelState(modelInfo);

  if (workersCache.has(key)) {
    return workersCache.get(key)!;
  }

  state.isLoading = true;
  state.error = null;

  try {
    const worker = new Worker(
      new URL("~/utils/nli.worker.ts", import.meta.url),
      { type: "module" }
    );

    await loadModelInWorker(modelInfo, worker, options?.onProgress);

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

const terminateWorker = (modelInfo: TModelInfo) => {
  const key = getModelKey(modelInfo);
  const worker = workersCache.get(key);

  if (worker) {
    worker.terminate();
    workersCache.delete(key);

    const state = getModelState(modelInfo);
    state.isLoadedInMemory = false;
  }
};

export const useModelLoader = () => {
  return {
    getOrLoadModel,
    getOrLoadWorker,
    terminateWorker,
    getModelState,
    getModelCacheStatus,
    clearModelCache,
    clearAllCache,
    syncAllCacheState
  };
};
