<template>
  <div class="z-100">
    <div
      v-if="!isExpanded"
    >
      <!-- No download in progress -->
      <div v-if="!hasActiveDownload">
        <button class="btn btn-sm btn-accent btn-outline" title="Manage Downloads" @click="isExpanded = true">
          <Icon name="lucide:download" class="w-4 h-4 mr-1" />
          Manage Downloads
        </button>
      </div>
      <!-- Minimized download progress bar (Steam-like) -->
      <div v-else class="bg-base-200 rounded-lg shadow-xl border border-base-content/20 min-w-[400px] max-w-[600px] cursor-pointer" @click="isExpanded = true">
        <div class="flex items-center justify-between px-4 pt-2 pb-1">
          <div class="flex items-center gap-2 flex-1">
            <div class="text-sm font-semibold text-base-content/80">
              <Icon name="lucide:download" class="w-4 h-4 mr-1 mb-[-2px]" />
              Downloading {{ currentDownload?.model.label }} |
            </div>
            <div class="text-xs text-base-content/60">
              {{ currentDownload?.progress }}% • {{ currentDownload?.model.transformersConfig.sizeMb }} MB
              <span v-if="downloadQueue.length > 1" class="ml-2">
                (+{{ downloadQueue.length - 1 }} in queue)
              </span>
            </div>
          </div>
          <button class="btn btn-ghost btn-xs">
            <Icon name="lucide:chevron-up" class="w-5 h-5" />
          </button>
        </div>
        <progress :value="currentDownload?.progress || 0" max="100" class="progress progress-primary w-full rounded-none block h-2" />
      </div>
    </div>

    <!-- Expanded modal -->
    <dialog ref="modelManagerDialog" class="modal modal-bottom">
      <div class="modal-box w-full py-3">
        <!-- Header -->
        <div class="flex items-center justify-between border-b border-base-content/10">
          <h2 class="text-lg font-bold">
            Model Manager
          </h2>
          <form method="dialog">
            <button class="btn btn-ghost btn-circle">
              <Icon name="lucide:x" />
            </button>
          </form>
        </div>

        <!-- Content -->
        <div class="pt-2 space-y-4 max-h-[calc(80vh-200px)] overflow-y-auto">
          <div class="space-y-2">
            <div
              v-for="model in transformersModels"
              :key="model.id"
              class="bg-base-200 rounded-lg p-4 border border-base-content/10"
            >
              <div class="flex items-start justify-between gap-4">
                <div class="flex-1">
                  <div class="flex items-center gap-2">
                    <h4 class="font-semibold">
                      {{ model.label }}
                    </h4>
                    <span
                      v-if="getModelDownloadStatus(model) === 'cached'"
                      class="badge badge-success"
                    >
                      <Icon name="lucide:check" class="w-3 h-3 mr-1" />
                      Cached
                    </span>
                    <span
                      v-else-if="getModelDownloadStatus(model) === 'downloading'"
                      class="badge badge-warning"
                    >
                      <span class="loading loading-spinner loading-xs mr-1" />
                      Downloading {{ getModelProgress(model) }}%
                    </span>
                    <span
                      v-else-if="getModelDownloadStatus(model) === 'queued'"
                      class="badge badge-info"
                    >
                      <Icon name="lucide:clock" class="w-3 h-3 mr-1" />
                      Queued ({{ downloadQueue.findIndex(item => item.model.id === model.id) + 1 }})
                    </span>
                    <span
                      v-else
                      class="badge badge-neutral"
                    >
                      Not Cached
                    </span>
                  </div>
                  <progress
                    v-if="getModelDownloadStatus(model) === 'downloading'"
                    :value="getModelProgress(model) || 0"
                    max="100"
                    class="progress progress-primary w-full block h-1 mt-1"
                  />
                </div>
                <div class="flex gap-2">
                  <div class="flex items-end gap-4 text-sm text-base-content/50">
                    <span>{{ model.transformersConfig?.sizeMb }} MB</span>
                  </div>
                  <button
                    v-if="getModelDownloadStatus(model) === 'not-cached'"
                    class="btn btn-primary btn-sm"
                    @click="queueDownload(model)"
                  >
                    <Icon name="lucide:download" class="w-4 h-4" />
                    Download
                  </button>
                  <button
                    class="btn btn-error btn-sm btn-outline"
                    :disabled="getModelDownloadStatus(model) !== 'cached'"
                    @click="removeModelFromCache(model)"
                  >
                    <Icon name="lucide:trash-2" class="w-4 h-4" />
                    Delete
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Footer -->
        <div class="flex items-center justify-between py-3">
          <small class="flex items-start gap-2">
            <Icon name="lucide:info" class="w-6 h-6 text-info mt-0.5" />
            <p>Models are downloaded once and cached in your browser for offline use. No data is sent to external servers during inference on downloaded models.</p>
          </small>
          <button
            class="btn btn-error btn-sm btn-outline"
            :disabled="hasActiveDownload || !hasCachedModels"
            @click="clearAllCache"
          >
            <Icon name="lucide:trash-2" class="w-4 h-4" />
            Clear Cache
          </button>
        </div>
      </div>
      <form method="dialog" class="modal-backdrop">
        <button>close</button>
      </form>
    </dialog>
  </div>
</template>

<script setup lang="ts">
import type { ModelId, TModelInfo } from "~/utils/models";

interface DownloadQueueItem {
  model: TModelInfo;
  progress: number;
}

const pendingModel = defineModel<TModelInfo | null>("pendingModel", { required: true });
const isExpanded = defineModel<boolean>("isExpanded", { default: false, set: (v) => {
  if (v) {
    modelManagerDialog.value?.showModal();
  } else {
    modelManagerDialog.value?.close();
  }
  return v;
} });

const emit = defineEmits<{
  "update:modelsState": [];
  modelReady: [modelId: ModelId];
}>();

const { getOrLoadModel, getModelCacheStatus, clearModelCache, clearAllCache: clearAllCacheFn, syncAllCacheState } = useModelLoader();

const modelManagerDialog = ref<HTMLDialogElement>();
const downloadQueue = ref<DownloadQueueItem[]>([]);
const isProcessingQueue = ref(false);

const transformersModels = computed<TModelInfo[]>(() =>
  MODELS.filter(m => m.transformersConfig && !m.disabled) as TModelInfo[]
);

const currentDownload = computed(() => downloadQueue.value[0] || null);

const hasActiveDownload = computed(() => downloadQueue.value.length > 0);

const hasCachedModels = computed(() =>
  transformersModels.value.some(m => getModelCacheStatus(m.transformersConfig) === "cached")
);

function getModelDownloadStatus(model: TModelInfo): "cached" | "downloading" | "queued" | "not-cached" {
  const cacheStatus = getModelCacheStatus(model.transformersConfig);
  if (cacheStatus === "cached") return "cached";

  const inQueue = downloadQueue.value.find(item => item.model.id === model.id);
  if (inQueue) {
    return inQueue === currentDownload.value ? "downloading" : "queued";
  }

  return "not-cached";
}

function getModelProgress(model: TModelInfo): number {
  const queueItem = downloadQueue.value.find(item => item.model.id === model.id);
  return queueItem?.progress || 0;
}

watch(pendingModel, async (modelInfo) => {
  if (modelInfo) {
    const status = getModelCacheStatus(modelInfo.transformersConfig);
    if (status === "cached") {
      emit("update:modelsState");
      emit("modelReady", modelInfo.id);
      pendingModel.value = null;
    }
  }
}, { immediate: true });

async function queueDownload(model: TModelInfo) {
  if (downloadQueue.value.some(item => item.model.id === model.id)) {
    return;
  }

  await syncAllCacheState();

  if (getModelCacheStatus(model.transformersConfig) === "cached") {
    useNotifier().info(`${model.label} is already cached`);
    return;
  }

  downloadQueue.value.push({ model, progress: 0 });
  if (!isProcessingQueue.value) {
    processQueue();
  }
}

async function processQueue() {
  if (isProcessingQueue.value || downloadQueue.value.length === 0) {
    return;
  }

  isProcessingQueue.value = true;

  while (downloadQueue.value.length > 0) {
    const current = downloadQueue.value[0];
    const modelInfo = current.model;

    try {
      await getOrLoadModel(modelInfo.transformersConfig, {
        onProgress: (progress: number) => {
          current.progress = Math.round(progress);
        },
      });

      useNotifier().success(`${modelInfo.label} downloaded successfully`);
      emit("update:modelsState");
      emit("modelReady", modelInfo.id);
    } catch (error) {
      console.error("Failed to download model:", error);
      useNotifier().error(`Failed to download ${modelInfo.label}: ${error instanceof Error ? error.message : "Unknown error"}`);
    }

    downloadQueue.value.shift();
  }

  isProcessingQueue.value = false;
}

async function removeModelFromCache(model: TModelInfo) {
  if (!confirm(`Remove ${model.label} from cache?`)) {
    return;
  }

  try {
    await clearModelCache(model.transformersConfig);
    useNotifier().success("Model removed from cache");
    emit("update:modelsState");
  } catch (error) {
    console.error("Failed to remove model from cache:", error);
    useNotifier().error("Failed to remove model from cache");
  }
}

async function clearAllCache() {
  if (!confirm("Clear all cached models? This will free up storage space but you'll need to download models again.")) {
    return;
  }

  try {
    await clearAllCacheFn();
    useNotifier().success("All models removed from cache");
    emit("update:modelsState");
  } catch (error) {
    console.error("Failed to clear cache:", error);
    useNotifier().error("Failed to clear cache");
  }
}

defineExpose({
  queueDownload
});

const dialogHandler = () => {
  isExpanded.value = false;
};

onMounted(async () => {
  await syncAllCacheState();
  if (isExpanded.value) {
    modelManagerDialog.value?.showModal();
  }

  if (modelManagerDialog.value) {
    modelManagerDialog.value.addEventListener("close", dialogHandler);
  }
});

onUnmounted(() => {
  if (modelManagerDialog.value) {
    modelManagerDialog.value.removeEventListener("close", dialogHandler);
  }
});
</script>
