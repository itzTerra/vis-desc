<template>
  <div class="z-100">
    <div>
      <!-- No download in progress -->
      <div v-if="!hasActiveDownload">
        <button class="btn btn-sm btn-accent btn-soft" title="Manage Downloads" @click="isExpanded = true">
          <Icon name="lucide:download" class="w-4 h-4 mr-1" />
          Manage Downloads
        </button>
      </div>
      <!-- Minimized download progress bar (Steam-like) -->
      <div v-else class="bg-base-200 rounded-lg shadow-xl border border-base-content/20 min-w-100 max-w-150 cursor-pointer" @click="isExpanded = true">
        <div class="flex items-center justify-between px-4 pt-2 pb-1">
          <div class="flex items-center gap-2 flex-1">
            <div class="text-sm font-semibold text-base-content/80">
              <Icon name="lucide:download" class="w-4 h-4 mr-1 -mb-0.5" />
              Downloading {{ currentDownload?.group.name }} |
            </div>
            <div class="text-xs text-base-content/60">
              {{ currentDownload?.progress }}% • {{ getGroupSize(currentDownload?.group) }} MB
              <span v-if="downloadQueue.length > 1" class="ml-2">
                (+{{ downloadQueue.length - 1 }} in queue)
              </span>
            </div>
          </div>
          <button class="btn btn-ghost btn-xs">
            <Icon name="lucide:chevron-up" size="8px" />
          </button>
        </div>
        <progress :value="currentDownload?.progress || 0" max="100" class="progress progress-primary w-full rounded-none block h-2" />
      </div>
    </div>

    <!-- Expanded modal -->
    <dialog ref="cacheManagerDialog" class="modal modal-bottom">
      <div class="modal-box w-full py-3">
        <!-- Header -->
        <div class="flex items-center justify-between border-b border-base-content/10">
          <h2 class="text-lg font-bold">
            Cache Manager
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
              v-for="group in MODEL_GROUPS"
              :key="group.id"
              class="bg-base-200 rounded-lg p-4 border border-base-content/10"
            >
              <div class="flex items-start justify-between gap-4">
                <div class="flex-1">
                  <div class="flex items-center gap-2">
                    <h4 class="font-semibold">
                      {{ group.name }}
                    </h4>
                    <span class="text-sm text-base-content/60">
                      {{ getGroupSize(group) }} MB
                    </span>
                    <span
                      v-if="getGroupDownloadStatus(group) === 'cached'"
                      class="badge badge-success"
                    >
                      <Icon name="lucide:check" class="w-3 h-3 mr-1" />
                      Downloaded
                    </span>
                    <span
                      v-else-if="getGroupDownloadStatus(group) === 'downloading'"
                      class="badge badge-warning"
                    >
                      <span class="loading loading-spinner loading-xs mr-1" />
                      Downloading {{ getGroupProgress(group) }}%
                    </span>
                    <span
                      v-else-if="getGroupDownloadStatus(group) === 'queued'"
                      class="badge badge-info"
                    >
                      <Icon name="lucide:clock" class="w-3 h-3 mr-1" />
                      Queued ({{ getGroupQueuePosition(group) }})
                    </span>
                  </div>
                  <progress
                    v-if="getGroupDownloadStatus(group) === 'downloading'"
                    :value="getGroupProgress(group) || 0"
                    max="100"
                    class="progress progress-primary w-full block h-1 mt-1"
                  />
                </div>
                <div class="flex gap-2 items-center">
                  <div v-if="isOnnxGroup(group)" class="flex items-center gap-2">
                    <label class="label cursor-pointer gap-2">
                      <span class="label-text text-xs">GPU</span>
                      <input
                        type="checkbox"
                        class="toggle toggle-sm"
                        :checked="provider[group.id] === 'webgpu'"
                        @change="(e: any) => updateProvider(group.id, e.target.checked)"
                      >
                    </label>
                  </div>
                  <button
                    v-if="getGroupDownloadStatus(group) === 'not-cached'"
                    class="btn btn-primary btn-sm"
                    @click="queueDownload(group)"
                  >
                    <Icon name="lucide:download" class="w-4 h-4" />
                    Download
                  </button>
                  <button
                    v-if="getGroupDownloadStatus(group) === 'cached'"
                    class="btn btn-error btn-sm btn-soft"
                    @click="removeGroupFromCache(group)"
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
            class="btn btn-error btn-sm btn-soft"
            :disabled="hasActiveDownload || !hasCachedGroups"
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
import { MODEL_GROUPS } from "~/utils/models";
import { useCacheController } from "~/composables/useCacheController";
import { CacheController } from "~/utils/CacheController";
import type { ModelGroup } from "~/types/cache";

interface DownloadQueueItem {
  group: ModelGroup;
  progress: number;
}

const isExpanded = defineModel<boolean>("isExpanded", { default: false });

watch(isExpanded, (v) => {
  if (v) {
    cacheManagerDialog.value?.showModal();
  } else {
    cacheManagerDialog.value?.close();
  }
});

const emit = defineEmits<{
  "update:groupsState": [];
}>();

const { downloadGroup, checkGroupCached } = useCacheController();
const cacheController = new CacheController();
const cacheManagerDialog = ref<HTMLDialogElement>();
const downloadQueue = ref<DownloadQueueItem[]>([]);
const isProcessingQueue = ref(false);
const provider = ref<Record<string, string>>({});
const cachedGroups = ref<Set<string>>(new Set());

const currentDownload = computed(() => downloadQueue.value[0] || null);
const hasActiveDownload = computed(() => downloadQueue.value.length > 0);
const hasCachedGroups = computed(() => cachedGroups.value.size > 0);

function isOnnxGroup(group: ModelGroup): boolean {
  return group.downloadables.some((dl) => dl.id.includes("catboost") || dl.id.includes("roberta"));
}

function getGroupDownloadStatus(group: ModelGroup): "cached" | "downloading" | "queued" | "not-cached" {
  if (cachedGroups.value.has(group.id)) return "cached";

  const inQueue = downloadQueue.value.find((item) => item.group.id === group.id);
  if (inQueue) {
    return inQueue === currentDownload.value ? "downloading" : "queued";
  }

  return "not-cached";
}

function getGroupProgress(group: ModelGroup): number {
  const queueItem = downloadQueue.value.find((item) => item.group.id === group.id);
  return queueItem?.progress || 0;
}

function getGroupQueuePosition(group: ModelGroup): number {
  const index = downloadQueue.value.findIndex((item) => item.group.id === group.id);
  return index >= 0 ? index + 1 : -1;
}

function updateProvider(groupId: string, enabled: boolean): void {
  provider.value[groupId] = enabled ? "webgpu" : "wasm";
  localStorage.setItem("onnx_provider", JSON.stringify(provider.value));
}

async function checkCachedGroups(): Promise<void> {
  const cached = new Set<string>();
  for (const group of MODEL_GROUPS) {
    if (await checkGroupCached(group)) {
      cached.add(group.id);
    }
  }
  cachedGroups.value = cached;
}

async function queueDownload(group: ModelGroup): Promise<void> {
  if (downloadQueue.value.some((item) => item.group.id === group.id)) {
    return;
  }

  await checkCachedGroups();

  if (downloadQueue.value.some((item) => item.group.id === group.id)) {
    return;
  }

  if (cachedGroups.value.has(group.id)) {
    useNotifier().info(`${group.name} is already cached`);
    return;
  }

  downloadQueue.value.push({ group, progress: 0 });
  if (!isProcessingQueue.value) {
    processQueue();
  }
}

async function processQueue(): Promise<void> {
  if (isProcessingQueue.value || downloadQueue.value.length === 0) {
    return;
  }

  isProcessingQueue.value = true;

  while (downloadQueue.value.length > 0) {
    const current = downloadQueue.value[0];
    const group = current.group;

    try {
      await downloadGroup(group, (progress) => {
        current.progress = progress;
      });

      useNotifier().success(`${group.name} downloaded successfully`);
      await checkCachedGroups();
      emit("update:groupsState");
    } catch (error) {
      console.error("Failed to download group:", error);
      useNotifier().error(error instanceof Error ? error.message : "Unknown error occured during download");
    }

    downloadQueue.value.shift();
  }

  isProcessingQueue.value = false;
}

async function removeGroupFromCache(group: ModelGroup): Promise<void> {
  if (!confirm(`Remove ${group.name} from cache?`)) {
    return;
  }

  try {
    for (const downloadable of group.downloadables) {
      const existsInOtherGroup = MODEL_GROUPS.some(
        (g) => g.id !== group.id && cachedGroups.value.has(g.id) && g.downloadables.some((d) => d.id === downloadable.id)
      );

      if (!existsInOtherGroup) {
        await cacheController.remove(downloadable.id);
      }
    }
    useNotifier().success("Group removed from cache");
    await checkCachedGroups();
    emit("update:groupsState");
  } catch (error) {
    console.error("Failed to remove group from cache:", error);
    useNotifier().error("Failed to remove group from cache");
  }
}

async function clearAllCache(): Promise<void> {
  if (
    !confirm(
      "Clear all cached models? This will free up storage space but you'll need to download models again."
    )
  ) {
    return;
  }

  try {
    await cacheController.clear();
    cachedGroups.value.clear();
    useNotifier().success("All models removed from cache");
    emit("update:groupsState");
  } catch (error) {
    console.error("Failed to clear cache:", error);
    useNotifier().error("Failed to clear cache");
  }
}

const dialogHandler = () => {
  isExpanded.value = false;
};

onMounted(async () => {
  const stored = localStorage.getItem("onnx_provider");
  if (stored) {
    try {
      provider.value = JSON.parse(stored);
    } catch {
      // Invalid JSON, use defaults
    }
  }

  try {
    await cacheController.init();
    await checkCachedGroups();
  } catch (error) {
    console.error("Failed to initialize cache:", error);
    useNotifier().error("Failed to initialize cache");
  }

  if (isExpanded.value) {
    cacheManagerDialog.value?.showModal();
  }

  if (cacheManagerDialog.value) {
    cacheManagerDialog.value.addEventListener("close", dialogHandler);
  }
});

onUnmounted(() => {
  if (cacheManagerDialog.value) {
    cacheManagerDialog.value.removeEventListener("close", dialogHandler);
  }
});

defineExpose({
  queueDownload,
  getGroupDownloadStatus,
  openNonModal: () => {
    if (cacheManagerDialog.value && !cacheManagerDialog.value.open) {
      cacheManagerDialog.value.setAttribute("open", "");
    }
  },
  closeNonModal: () => {
    if (cacheManagerDialog.value && cacheManagerDialog.value.open) {
      cacheManagerDialog.value.removeAttribute("open");
    }
  },
});
</script>
