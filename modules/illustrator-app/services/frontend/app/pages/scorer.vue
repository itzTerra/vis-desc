<template>
  <div class="min-h-screen bg-base-100 p-4">
    <div class="max-w-4xl mx-auto space-y-4">
      <div class="flex items-center justify-between">
        <h1 class="text-3xl font-bold">
          Text Scorer
        </h1>
        <NuxtLink to="/" class="btn btn-ghost btn-sm">
          <Icon name="lucide:home" />
          Back to Home
        </NuxtLink>
      </div>

      <div class="card bg-base-200 shadow-xl">
        <div class="card-body space-y-4">
          <div class="flex flex-wrap gap-4 items-center">
            <ModelSelect
              v-model="selectedModel"
              @request-model-download="handleModelDownloadRequest"
            />
            <div class="form-control">
              <label class="label cursor-pointer gap-2">
                <span class="label-text font-semibold">Use Splitter</span>
                <input
                  v-model="useSplitter"
                  type="checkbox"
                  class="toggle toggle-primary"
                >
              </label>
            </div>
          </div>
          <div class="form-control">
            <textarea
              v-model="inputText"
              class="textarea textarea-bordered h-48 w-full"
              placeholder="Enter the text you want to score..."
            />
          </div>

          <div class="flex flex-wrap gap-4 items-center">
            <button
              class="btn btn-primary gap-2 ms-auto"
              :disabled="!inputText.trim() || isProcessing"
              @click="handleScore"
            >
              <Icon name="lucide:sparkles" />
              {{ isProcessing ? 'Scoring...' : 'Score' }}
            </button>
          </div>

          <div v-if="isProcessing" class="alert alert-info">
            <Icon name="lucide:loader-2" class="animate-spin" />
            <span>{{ currentStage }}</span>
          </div>
        </div>
      </div>

      <div v-if="results.length > 0" class="card bg-base-200 shadow-xl">
        <div class="card-body">
          <h2 class="card-title">
            Results
          </h2>
          <div class="overflow-x-auto">
            <table class="table table-zebra">
              <thead>
                <tr>
                  <th class="w-16">
                    #
                  </th>
                  <th class="w-24">
                    Score
                  </th>
                  <th>
                    Text
                  </th>
                </tr>
              </thead>
              <tbody>
                <tr
                  v-for="(result, idx) in results"
                  :key="idx"
                  :class="{ 'bg-success/20': result.score && result.score >= 0.6 }"
                >
                  <td>{{ idx + 1 }}</td>
                  <td>
                    <span
                      v-if="result.score !== undefined"
                      class="badge"
                      :class="{
                        'badge-success': result.score >= 0.6,
                        'badge-warning': result.score >= 0.25 && result.score < 0.6,
                        'badge-error': result.score < 0.25
                      }"
                    >
                      {{ result.score.toFixed(3) }}
                    </span>
                    <span v-else class="loading loading-spinner loading-xs" />
                  </td>
                  <td class="max-w-md">
                    <div class="truncate">
                      {{ result.text }}
                    </div>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import type { Segment } from "~/types/common";
import { SCORERS } from "~/utils/models";

const { $api: call } = useNuxtApp();
const runtimeConfig = useRuntimeConfig();
const { selectedModel } = useAppStorage();

const inputText = ref("");
const useSplitter = ref(false);
const isProcessing = ref(false);
const currentStage = ref("Initializing...");
const results = ref<Array<{ text: string; score?: number; id?: number }>>([]);

const selectedScorer = computed(() =>
  SCORERS.find(s => s.id === selectedModel.value)
);

const socket = useWebSocket(`${runtimeConfig.public.wsBaseUrl}/ws/progress/`, {
  immediate: false,
  onConnected: () => {
    console.log("WS: connected");
  },
  onDisconnected: () => {
    console.log("WS: disconnected");
    isProcessing.value = false;
  },
  onMessage: (ws, e) => {
    const data = JSON.parse(e.data) as SocketMessage;
    switch (data.type) {
    case "batch":
    {
      const batch = data.content as Segment[];
      for (const segment of batch) {
        updateSegmentScore(segment);
      }
      break;
    }
    case "segment":
    {
      updateSegmentScore(data.content as Segment);
      break;
    }
    case "info":
      console.log("WS[INFO]:", data.content);
      break;
    case "error":
      console.error("WS[ERROR]:", data.content);
      useNotifier().error("Scoring error: " + String(data.content));
      break;
    case "success":
      console.log("WS[SUCCESS]:", data.content);
      ws.close();
      isProcessing.value = false;
      currentStage.value = "Completed";
      break;
    }
  },
});

function updateSegmentScore(segment: Segment) {
  const idx = results.value.findIndex(
    (r) => ((segment as any).id !== undefined && r.id === (segment as any).id) || r.text === segment.text
  );
  if (idx !== -1) {
    results.value[idx].score = segment.score;
  }
}

async function handleScore() {
  const scorer = selectedScorer.value;
  if (!scorer) {
    useNotifier().error("No scorer selected.");
    return;
  }

  if (!inputText.value.trim()) {
    useNotifier().error("Please enter some text to score.");
    return;
  }

  isProcessing.value = true;
  currentStage.value = useSplitter.value ? "Segmenting text..." : "Preparing...";
  results.value = [];

  try {
    const data = await call(scorer.socketBased ? "/api/scorer/process/text" : "/api/scorer/segment/text", {
      method: "POST",
      body: {
        text: inputText.value,
        model: selectedModel.value as any,
        split: useSplitter.value,
      },
    });

    if (data?.segments && Array.isArray(data.segments)) {
      results.value = data.segments.map((seg: any) => ({
        id: seg.id,
        text: seg.text,
        score: undefined,
      }));
    }

    if (scorer.socketBased) {
      socket.open();

      await new Promise<void>((resolve, reject) => {
        const timeout = setTimeout(() => {
          stopWatcher();
          reject(new Error("Socket connection timeout"));
        }, 10000);

        const stopWatcher = watch(
          () => socket.status.value,
          (status) => {
            if (status === "OPEN") {
              clearTimeout(timeout);
              stopWatcher();
              resolve();
            } else if (status === "CLOSED") {
              clearTimeout(timeout);
              stopWatcher();
              reject(new Error("Socket connection closed"));
            }
          },
          { immediate: true }
        );
      });
      socket.send(JSON.stringify({ ws_key: (data as any).ws_key }));

      currentStage.value = "Scoring...";
    } else {
      await scorer.score(
        data,
        (progress) => {
          currentStage.value = progress.stage || "Scoring...";
          for (const segment of progress.results || []) {
            updateSegmentScore(segment);
          }
        },
        scorer.socketBased ? socket : undefined
      );
      isProcessing.value = false;
      currentStage.value = "Completed";
    }
  } catch (error) {
    console.error("Scoring error:", error);
    useNotifier().error("Failed to score text: " + (error as Error).message);
    isProcessing.value = false;
    currentStage.value = "Error";
  }
}

function handleModelDownloadRequest(scorerId: string) {
  console.log("Model download requested:", scorerId);
}

onBeforeUnmount(() => {
  if (socket.status.value === "OPEN") {
    socket.close();
  }
});
</script>
