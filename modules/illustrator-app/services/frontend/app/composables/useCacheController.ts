import type { ModelGroup, ScorerProgress } from "~/types/cache";
import { CacheController } from "~/utils/CacheController";

let cacheControllerInstance: CacheController | null = null;

function getCacheControllerInstance(): CacheController {
  if (!cacheControllerInstance) {
    cacheControllerInstance = new CacheController();
  }
  return cacheControllerInstance;
}

export function useCacheController() {
  const controller = getCacheControllerInstance();

  const downloadGroup = async (
    group: ModelGroup,
    onProgress: (progress: ScorerProgress) => void
  ): Promise<void> => {
    return controller.downloadGroup(group, onProgress);
  };

  const checkGroupCached = async (group: ModelGroup): Promise<boolean> => {
    const allCached = await Promise.all(
      group.downloadables.map((downloadable) => controller.exists(downloadable.id))
    );
    return allCached.every((cached) => cached);
  };

  return {
    downloadGroup,
    checkGroupCached,
  };
}
