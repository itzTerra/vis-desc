import type { Downloadable, ModelGroup, ScorerProgress } from "~/types/cache";

export class CacheController {
  private namespace = "transformers-cache";
  private cache: IDBDatabase | null = null;
  private initPromise: Promise<void> | null = null;

  async init(): Promise<void> {
    if (this.cache) {
      return;
    }

    if (this.initPromise) {
      return this.initPromise;
    }

    this.initPromise = new Promise((resolve, reject) => {
      const request = indexedDB.open(this.namespace, 1);

      request.onerror = () => {
        this.initPromise = null;
        reject(new Error(`Failed to open IndexedDB: ${request.error?.message || "Unknown error"}`));
      };

      request.onsuccess = () => {
        this.cache = request.result;
        this.initPromise = null;
        resolve();
      };

      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result;
        if (!db.objectStoreNames.contains("downloads")) {
          db.createObjectStore("downloads", { keyPath: "key" });
        }
      };
    });
  }

  async exists(key: string): Promise<boolean> {
    if (!this.cache) {
      await this.init();
    }

    return new Promise((resolve, reject) => {
      const transaction = this.cache!.transaction(["downloads"], "readonly");
      const store = transaction.objectStore("downloads");
      const request = store.get(`${this.namespace}:${key}`);

      request.onerror = () => {
        reject(new Error(`Failed to check cache: ${request.error?.message || "Unknown error"}`));
      };

      request.onsuccess = () => {
        resolve(!!request.result);
      };
    });
  }

  async downloadGroup(
    group: ModelGroup,
    onProgress: (progress: ScorerProgress) => void
  ): Promise<void> {
    if (!this.cache) {
      await this.init();
    }

    const downloadables = group.downloadables;
    const totalSize = downloadables.reduce((sum, d) => sum + d.sizeMb, 0);
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
            onProgress({
              stage: `Downloading ${group.name}...`,
              progress: Math.min(100, Math.floor(overallProgress * 100)),
            });
          });

          await this.cacheDownloadable(downloadable);
          downloadProgress[downloadable.id] = 1;
        } else {
          downloadProgress[downloadable.id] = 1;
        }
      }
    } catch (error) {
      throw new Error(`Failed to download group ${group.id}: ${error instanceof Error ? error.message : "Unknown error"}`);
    }
  }

  private async cacheDownloadable(downloadable: Downloadable): Promise<void> {
    if (!this.cache) {
      await this.init();
    }

    return new Promise((resolve, reject) => {
      const transaction = this.cache!.transaction(["downloads"], "readwrite");
      const store = transaction.objectStore("downloads");
      const request = store.put({
        key: `${this.namespace}:${downloadable.id}`,
        id: downloadable.id,
        label: downloadable.label,
        timestamp: Date.now(),
      });

      request.onerror = () => {
        reject(new Error(`Failed to cache downloadable: ${request.error?.message || "Unknown error"}`));
      };

      request.onsuccess = () => {
        resolve();
      };
    });
  }
}
