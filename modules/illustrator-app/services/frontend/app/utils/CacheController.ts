import type { ModelGroup } from "~/types/cache";
import type { ProgressCallback } from "~/types/common";
import { CACHE_NAME } from "~/utils/utils";

export function getGroupSize(group: ModelGroup): number {
  return group.downloadables.reduce((sum, dl) => sum + dl.sizeMb, 0);
}

export class CacheController {
  private cacheName = CACHE_NAME;
  private cache: Cache | null = null;
  private initPromise: Promise<void> | null = null;
  private keyMap: Map<string, string[]> = new Map([
    [DOWNLOADABLES.featureService.id, [
      "english-wordnet-2025-json.zip",
      "Terraa/entities_google_bert_uncased_L-4_H-256_A-4-v1.0-ONNX/resolve/main/onnx/model.onnx",
      "Terraa/entities_google_bert_uncased_L-4_H-256_A-4-v1.0-ONNX/resolve/main/config.json",
      "Terraa/entities_google_bert_uncased_L-4_H-256_A-4-v1.0-ONNX/resolve/main/vocab.txt",
      "Terraa/entities_google_bert_uncased_L-4_H-256_A-4-v1.0-ONNX/resolve/main/vocab.txt::meta",
      "Xenova/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx",
      "Xenova/all-MiniLM-L6-v2/resolve/main/config.json",
      "Xenova/all-MiniLM-L6-v2/resolve/main/tokenizer.json",
      "Xenova/all-MiniLM-L6-v2/resolve/main/tokenizer_config.json"
    ]],
    [DOWNLOADABLES.catboost.id, []],
    [DOWNLOADABLES.nliRoberta.id, [
      "richardr1126/roberta-base-zeroshot-v2.0-c-ONNX/resolve/main/model_quantized.onnx",
      "richardr1126/roberta-base-zeroshot-v2.0-c-ONNX/resolve/main/config.json",
      "richardr1126/roberta-base-zeroshot-v2.0-c-ONNX/resolve/main/tokenizer.json",
      "richardr1126/roberta-base-zeroshot-v2.0-c-ONNX/resolve/main/tokenizer_config.json"
    ]]
  ]);

  private getKeys(key: string): string[] {
    if (!this.keyMap.has(key)) {
      console.error(`Unknown downloadable ID: ${key}`);
      return ["NON_EXISTENT_KEY_SO_THAT_THIS_DOWNLOADABLE_IS_ALWAYS_MISSING"];
    }
    return this.keyMap.get(key)!;
  }

  async init(): Promise<void> {
    if (this.cache) return;
    if (this.initPromise) return this.initPromise;

    this.initPromise = (async () => {
      this.cache = await caches.open(this.cacheName);
      this.initPromise = null;
    })();

    return this.initPromise;
  }

  async exists(downloadableId: string): Promise<boolean> {
    await this.init();
    if (!this.cache) {
      return false;
    }
    const cacheKeys = await this.cache.keys();
    return this.getKeys(downloadableId).every((key) => cacheKeys.some((request) => request.url.includes(key)));
  }

  async downloadGroup(
    group: ModelGroup,
    onProgress: ProgressCallback
  ): Promise<void> {
    await this.init();

    const downloadables = group.downloadables;
    const totalSize = getGroupSize(group);
    const downloadProgress: Record<string, number> = {};

    for (const downloadable of downloadables) {
      downloadProgress[downloadable.id] = 0;
    }

    try {
      for (const downloadable of downloadables) {
        const alreadyCached = await this.exists(downloadable.id);

        if (!alreadyCached) {
          await downloadable.download((progress: number) => {
            downloadProgress[downloadable.id] = progress;

            let totalDownloaded = 0;
            for (const dl of downloadables) {
              totalDownloaded += dl.sizeMb * ((downloadProgress[dl.id] ?? 0) / 100);
            }

            const overallProgress = totalSize > 0 ? totalDownloaded / totalSize : 0;
            onProgress(Math.min(100, Math.floor(overallProgress * 100)));
          });
        }
        downloadProgress[downloadable.id] = 100;
      }
    } catch (error) {
      throw new Error(`Failed to download ${group.id}: ${error instanceof Error ? error.message : "Unknown error"}`);
    }
  }

  async remove(downloadableId: string): Promise<void> {
    await this.init();
    if (!this.cache) return;
    const cacheKeys = await this.cache.keys();
    for (const key of this.getKeys(downloadableId)) {
      let found = false;
      for (const request of cacheKeys) {
        if (request.url.includes(key)) {
          await this.cache!.delete(request);
          found = true;
        }
      }
      if (!found) {
        console.warn(`No cached entry found for key fragment: ${key}`);
      }
    }
  }

  async clear(): Promise<void> {
    // delete the whole cache for simplicity
    await caches.delete(this.cacheName);
    this.cache = null;
  }
}
