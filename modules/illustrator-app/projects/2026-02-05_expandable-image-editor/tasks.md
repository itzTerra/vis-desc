# Expandable Image Editor Component - Implementation Battleplan

## Overview
This document maps user stories to specific implementation tasks, organized by phase with proper dependencies. Each task is a single logical unit with <200 LoC per chunk. Follow Vue 3 Composition API patterns and existing codebase conventions.

## Critical Architecture Changes (Post-Review)

### ⚠️ Single Source of Truth for Image State
**Key Change**: `imageUrl` and `imageLoading` fields have been **REMOVED** from the `Highlight` type.

**Why**: 
- Prevents prop mutation issues
- Establishes clear ownership (editors own image state)
- Keeps highlights as read-only document entities

**What This Means**:
- Highlights contain only: `id`, `text`, `polygons`, `score`, `score_received_at`
- Editors receive `highlightId` and `initialText` as props
- Editors manage their own `imageUrl` and `imageLoading` state internally
- No more `props.highlight.imageUrl` - use internal `imageUrl.value` instead

### ⚠️ Physical Location and Layout Architecture

**Physical Location**:
- **RIGHT Side**: ImageEditor panels appear on the **RIGHT side** in the 512px ImageLayer space
- **Vertical Alignment**: Top position aligned with corresponding highlight's Y coordinate on PDF
- **Horizontal Position**: Fixed at right edge of ImageLayer (NOT at highlight coordinates)
- **Collapse Button**: Protrudes slightly LEFT into PDF viewer area using **negative left value** (e.g., `left: -20px`)
- **Visual Bridge**: Collapse button bridges the gap between PDF viewer and editor panels

**Complete Replacement Architecture**:
- ImageEditor components **COMPLETELY REPLACE** the current `<figure>` image display
- The v-for loop in ImageLayer.vue that currently renders `<figure>` tags will be **CHANGED** to render `<ImageEditor>` components
- **BEFORE**: `<figure v-for="highlight in highlights.filter(h => h.imageUrl || h.imageLoading)">`
- **AFTER**: `<ImageEditor v-for="editorId in openEditorIds">`
- This is NOT supplemental - the entire figure loop is REPLACED

**Image Inside Editor**:
- Generated images are rendered **INSIDE** the ImageEditor component
- Images appear below the textarea/buttons area within the editor
- There is NO separate image display - the editor IS the image display
- When collapsed: just collapse button + generated image (if exists)
- When expanded: top bar, textarea, buttons, generated image, history, delete button

**How it works**:
- HighlightLayer "Illustrate" button emits "open-editor" event with highlightId
- PdfViewer receives event and adds highlightId to `openEditorIds` Set
- ImageLayer receives `openEditorIds` prop
- ImageLayer's v-for iterates through `openEditorIds` and renders `<ImageEditor>` for each
- Each ImageEditor positioned on RIGHT side, vertically aligned with its highlight on PDF
- Multiple editor components can be open simultaneously

### Technical Fixes Applied
- ✅ Emit declaration: `const emit = defineEmits<{ delete: [] }>()`
- ✅ API calls: `const { $api: call } = useNuxtApp()`
- ✅ Map initialization: `reactive(new Map<number, EditorState>())`
- ✅ Emit typing: Tuple syntax `{ select: [id: number] }` matches HighlightLayer
- ✅ Removed `readonly()` wrapper from composable returns (not valid Vue 3 pattern)

---

## Phase 1: Data Model & Types

- [x] Task 1.1: Update Highlight Type (Remove Image State)
**User Story**: All (foundational architecture change)
**Depends on**: Nothing
**Effort**: 10 min

**Remove imageUrl and imageLoading from Highlight type** - these fields will be owned by ImageEditor components instead.

Update [services/frontend/app/types/common.d.ts](../../services/frontend/app/types/common.d.ts):

```typescript
export type Highlight = {
  id: number;
  text: string;
  polygons: Record<number, number[][]>;
  score?: number;
  score_received_at?: string;
  // NOTE: imageUrl and imageLoading removed - editors own their image state
};
```

**Note**: This is part of a complete refactor of image generation. The existing image display in ImageLayer.vue will be replaced by ImageEditor components. No migration strategy needed.

**Verification**:
- TypeScript compiles without errors
- No references to `highlight.imageUrl` or `highlight.imageLoading` in codebase
- Highlights remain read-only document entities

---

- [x] Task 1.2: Define Editor Types
**User Story**: All (state and history are foundational)
**Depends on**: Task 1.1
**Effort**: 20 min

Add to [services/frontend/app/types/common.d.ts](../../services/frontend/app/types/common.d.ts):

```typescript
export type EditorHistoryItem = {
  text: string;
  imageUrl?: string;  // Only set if generated
};

export type EditorState = {
  highlightId: number;
  isExpanded: boolean;
  currentPrompt: string;
  imageUrl: string | null;      // Editor owns image state
  history: EditorHistoryItem[];
  historyIndex: number;
  enhanceLoading: boolean;       // Separate loading state for Enhance button
  generateLoading: boolean;      // Separate loading state for Generate button
};
```

**Key Architecture**: `EditorState` now includes `imageUrl` and `imageLoading` fields that were removed from Highlight type. Editors own their own image state completely.

**Verification**:
- Both types exported from common.d.ts
- Can be imported in component files
- TypeScript recognizes EditorState.imageUrl as nullable string

---

- [x] Task 1.3: Add Backend Schema for Enhance Endpoint
**User Story**: User Enhances Prompt Text
**Depends on**: Nothing
**Effort**: 20 min

Modify [services/api/core/schemas.py](../../services/api/core/schemas.py):

```python
class EnhanceTextBody(Schema):
    text: str

class EnhanceTextResponse(Schema):
    text: str
```

**Verification**:
- Schema classes can be imported in api.py
- Matches existing TextBody pattern

---

## Phase 2: Backend API Setup

- [x] Task 2.1: Implement Enhance Endpoint (Stub)
**User Story**: User Enhances Prompt Text
**Depends on**: Task 1.3
**Effort**: 30 min

Add to [services/api/core/api.py](../../services/api/core/api.py):

```python
@api.post("/enhance", response=EnhanceTextResponse)
def enhance_text(request, body: EnhanceTextBody):
    # Stub implementation: return slightly modified text
    enhanced_text = f"{body.text} [enhanced]"
    return {"text": enhanced_text}
```

**Verification**:
- Endpoint responds to `POST /api/enhance` with `{ text: string }`
- Response matches schema
- Can be verified with curl or Postman
- Response text contains the input text to support deduplication checks

**Notes**:
- This is a stub. Real implementation would use LLM or other text processing
- If needed, return the exact same text to confirm deduplication behavior

---

## Phase 3: Frontend Composables

- [x] Task 3.1: Create useEditorHistory Composable
**User Story**: User Navigates Editor History (Story 4)
**Depends on**: Task 1.1, Task 1.2
**Effort**: 1.5 hours

Create [services/frontend/app/composables/useEditorHistory.ts](../../services/frontend/app/composables/useEditorHistory.ts):

```typescript
import type { EditorHistoryItem } from '~/types/common'

export function useEditorHistory() {
  const history = ref<EditorHistoryItem[]>([])
  const historyIndex = ref(-1)

  const currentHistoryItem = computed(() => {
    if (historyIndex.value === -1) return null
    return history.value[historyIndex.value] || null
  })

  const isAtStart = computed(() => historyIndex.value <= 0)
  const isAtEnd = computed(() => historyIndex.value >= history.value.length - 1)

  function addToHistory(text: string, imageUrl?: string) {
    // Deduplication: skip if text matches immediately previous entry
    const lastItem = history.value[history.value.length - 1]
    if (lastItem && lastItem.text === text && lastItem.imageUrl === imageUrl) {
      return
    }

    const newItem: EditorHistoryItem = { text, imageUrl }

    // If we're in history view, truncate history from current position
    if (historyIndex.value !== -1) {
      history.value = history.value.slice(0, historyIndex.value + 1)
    }

    history.value.push(newItem)
    historyIndex.value = -1 // Return to editing mode
  }

  function navigatePrevious() {
    if (historyIndex.value === -1) {
      // From edit mode, go to last history item
      historyIndex.value = history.value.length - 1
    } else if (historyIndex.value > 0) {
      historyIndex.value--
    }
  }

  function navigateNext() {
    if (historyIndex.value < history.value.length - 1) {
      historyIndex.value++
    } else if (historyIndex.value === history.value.length - 1) {
      // From last item, return to edit mode
      historyIndex.value = -1
    }
  }

  function clearHistory() {
    history.value = []
    historyIndex.value = -1
  }

  return {
    history,
    historyIndex,
    currentHistoryItem,
    isAtStart,
    isAtEnd,
    addToHistory,
    navigatePrevious,
    navigateNext,
    clearHistory,
  }
}
```

**Verification**:
- Can import and use in component
- History properly deduplicates on text match
- Navigation works at boundaries
- History truncation works when resuming from mid-history point

---

## Phase 4: Component Structure

- [x] Task 4.1: Create ImageEditor Component Shell
**User Story**: User Expands Editor for Existing Generated Image (Story 1)
**Depends on**: Task 1.2, Task 3.1
**Effort**: 1 hour

Create [services/frontend/app/components/ImageEditor.vue](../../services/frontend/app/components/ImageEditor.vue):

```vue
<template>
  <div class="image-editor relative bg-base-100 border-l border-base-300" style="width: 100%;">
    <!-- Collapse Button (protrudes LEFT into PDF viewer area) -->
    <button
      class="btn btn-sm btn-circle absolute top-2"
      style="left: -20px;"  <!-- Negative left value to protrude into PDF area -->
      :title="isExpanded ? 'Collapse' : 'Expand'"
      @click="isExpanded = !isExpanded"
    >
      <Icon :name="isExpanded ? 'lucide:chevron-right' : 'lucide:chevron-left'" />
    </button>

    <!-- Editor Content (visible when expanded) -->
    <Transition name="slide">
      <div v-show="isExpanded" class="p-4 space-y-4">
        <!-- Top Bar -->
        <div class="flex justify-between items-center">
          <span class="text-sm font-semibold">Image Editor</span>
        </div>

        <!-- Prompt Area (hidden if image exists) -->
        <template v-if="!hasImage">
          <textarea
            v-model="currentPrompt"
            class="textarea textarea-bordered w-full h-24"
            placeholder="Enter or paste image prompt..."
            :disabled="enhanceLoading || generateLoading"
          />
          <div class="flex gap-2">
            <button
              class="btn btn-sm btn-primary flex-1"
              @click="handleEnhance"
              :disabled="enhanceLoading || generateLoading || !currentPrompt"
            >
              <Icon v-if="enhanceLoading" name="lucide:loader" class="animate-spin" />
              Enhance
            </button>
            <button
              class="btn btn-sm btn-secondary flex-1"
              @click="handleGenerate"
              :disabled="enhanceLoading || generateLoading || !currentPrompt"
            >
              <Icon v-if="generateLoading" name="lucide:loader" class="animate-spin" />
              Generate
            </button>
          </div>
        </template>

        <!-- Generated Image Display (INSIDE editor) -->
        <div v-if="hasImage" class="border rounded p-2">
          <img v-if="imageUrl" :src="imageUrl" class="w-full h-auto" alt="Generated image" />
          <div v-if="imageLoading" class="flex justify-center items-center h-32">
            <Icon name="lucide:loader" class="animate-spin" size="32" />
          </div>
        </div>

        <!-- History Display -->
        <div v-if="history.length > 0" class="border-t pt-4">
          <div class="text-sm font-semibold mb-2">History</div>
          <div class="flex gap-2 items-center mb-2">
            <button
              class="btn btn-xs"
              @click="navigatePrevious"
              :disabled="isAtStart"
            >
              <Icon name="lucide:chevron-left" />
            </button>
            <div class="text-xs text-base-content/70 flex-1 text-center">
              {{ currentHistoryItem ? `${historyIndex + 1}/${history.length}` : 'Editing' }}
            </div>
            <button
              class="btn btn-xs"
              @click="navigateNext"
              :disabled="isAtEnd"
            >
              <Icon name="lucide:chevron-right" />
            </button>
          </div>
          <div v-if="currentHistoryItem" class="bg-base-200 p-2 rounded text-xs mb-2 max-h-20 overflow-auto">
            {{ currentHistoryItem.text }}
          </div>
          <button
            class="btn btn-xs btn-outline w-full"
            @click="clearHistory"
          >
            Clear History
          </button>
        </div>

        <!-- Delete Button -->
        <button
          class="btn btn-sm btn-error w-full"
          @click="handleDelete"
        >
          <Icon name="lucide:trash-2" />
          Delete Editor
        </button>
      </div>
    </Transition>

    <!-- Collapsed State: Show image if exists -->
    <div v-if="!isExpanded && hasImage" class="p-2">
      <img v-if="imageUrl" :src="imageUrl" class="w-full h-auto" alt="Generated image" />
    </div>
  </div>
</template>

<script setup lang="ts">
import type { EditorHistoryItem } from '~/types/common'
import { useEditorHistory } from '~/composables/useEditorHistory'

const props = defineProps<{
  highlightId: number;
  initialText: string;
}>()

const emit = defineEmits<{
  delete: [];
}>()

const isExpanded = ref(false)
const currentPrompt = ref(props.initialText)

// Editor owns image state
const imageUrl = ref<string | null>(null)
const enhanceLoading = ref(false)   // Loading state for Enhance button
const generateLoading = ref(false)  // Loading state for Generate button

const {
  history,
  historyIndex,
  currentHistoryItem,
  isAtStart,
  isAtEnd,
  addToHistory,
  navigatePrevious,
  navigateNext,
  resumeEditingFromHistory: resumeFromHistory,
  clearHistory,
} = useEditorHistory()

const hasImage = computed(() => !!imageUrl.value)

function handleEnhance() {
  // Implemented in Task 4.2
}

function handleGenerate() {
  // Implemented in Task 4.3
}

function handleDelete() {
  emit('delete')
}

function loadHistoryItem() {
  const item = currentHistoryItem.value
  if (item) {
    currentPrompt.value = item.text
  }
}
</script>

<style scoped>
.slide-enter-active,
.slide-leave-active {
  transition: opacity 0.3s;
}

.slide-enter-from,
.slide-leave-to {
  opacity: 0;
}
</style>
```

**Verification**:
- Component renders with collapse button visible on LEFT edge (protruding into PDF area)
- Collapse button positioned with negative left value (e.g., `left: -20px`)
- Click expand/collapse toggles visibility
- Textarea visible when no image exists
- Generated image displays INSIDE editor component (below textarea/buttons when expanded)
- When collapsed with image: only collapse button + image visible
- Delete button shows when expanded
- History section displays when history has entries
- All content positioned on RIGHT side of screen

---

- [ ] Task 4.2: Implement Enhance Handler
**User Story**: User Enhances Prompt Text (Story 2)
**Depends on**: Task 4.1, Task 2.1
**Effort**: 1 hour

Replace `handleEnhance()` in ImageEditor.vue:

```typescript
async function handleEnhance() {
  if (!currentPrompt.value.trim()) return

  enhanceLoading.value = true

  try {
    const { $api: call } = useNuxtApp()
    
    const res = await call('/api/enhance', {
      method: 'POST',
      body: { text: currentPrompt.value }
    })

    if (!res || !res.text) {
      useNotifier().error('Enhance failed')
      return
    }

    currentPrompt.value = res.text
    addToHistory(currentPrompt.value)
    
    useNotifier().success('Prompt enhanced')
  } catch (error) {
    useNotifier().error('Enhance request failed')
    console.error(error)
  } finally {
    enhanceLoading.value = false
  }
}
```

**Verification**:
- Calls `/api/enhance` with current prompt text
- Both input and output text added to history
- History properly deduplicates if endpoint returns same text
- Error handling shows notification
- Loading state managed correctly
- Textarea updated with enhanced text

---

- [ ] Task 4.3: Implement Generate Handler
**User Story**: User Generates Image from Prompt (Story 3)
**Depends on**: Task 4.1, Task 2.1 (existing endpoint)
**Effort**: 1 hour

Replace `handleGenerate()` in ImageEditor.vue:

```typescript
async function handleGenerate() {
  if (!currentPrompt.value.trim()) return

  generateLoading.value = true

  try {
    const { $api: call } = useNuxtApp()

    const res = await call('/api/gen-image-bytes', {
      method: 'POST',
      body: { text: currentPrompt.value }
    })

    if (!res) {
      useNotifier().error('Image generation failed')
      return
    }

    const blob = new Blob([res as any], { type: 'image/png' })
    const url = URL.createObjectURL(blob)

    imageUrl.value = url
    addToHistory(currentPrompt.value, url)

    useNotifier().success('Image generated')
  } catch (error) {
    useNotifier().error('Image generation failed')
    console.error(error)
  } finally {
    generateLoading.value = false
  }
}
```

**Verification**:
- Calls `/api/gen-image-bytes` with prompt text
- Image blob converted to URL and set on highlight
- History records generation with image URL
- Loading state managed on both editor and highlight
- Prompt area hides automatically after success (via hasImage computed)
- Error notifications show properly

---

## Phase 5: Parent Component Integration

- [ ] Task 5.1: Render ImageEditor Components in ImageLayer's Template (Complete Replacement)
**User Story**: All (integration point)
**Depends on**: Task 4.1, Task 4.2, Task 4.3
**Effort**: 1.5 hours

**Architecture**: ImageEditor components **COMPLETELY REPLACE** the current `<figure>` image display in ImageLayer.vue. The entire v-for loop that renders images is being replaced with ImageEditor rendering.

**Step 1**: Modify [services/frontend/app/components/PdfViewer.vue](../../services/frontend/app/components/PdfViewer.vue) to track open editors:

```typescript
// Add to script setup
const openEditorIds = ref(new Set<number>())

function openImageEditor(highlightId: number) {
  openEditorIds.value.add(highlightId)
}

function closeImageEditor(highlightId: number) {
  openEditorIds.value.delete(highlightId)
}
```

**Step 2**: Modify [services/frontend/app/components/ImageLayer.vue](../../services/frontend/app/components/ImageLayer.vue) template:

**BEFORE (Current - REMOVE THIS ENTIRELY)**:
```vue
<template>
  <div class="image-layer absolute inset-0 pointer-events-none">
    <!-- OLD: Current figure loop for images - THIS IS REMOVED -->
    <figure
      v-for="highlight in highlights.filter(h => h.imageUrl || h.imageLoading)"
      :key="highlight.id"
      :style="getHighlightImageStyle(highlight)"
      class="absolute pointer-events-auto"
    >
      <img v-if="highlight.imageUrl" :src="highlight.imageUrl" />
      <div v-if="highlight.imageLoading" class="loading">Loading...</div>
    </figure>
  </div>
</template>
```

**AFTER (New - REPLACE WITH THIS)**:
```vue
<template>
  <div class="image-layer absolute inset-0 pointer-events-none">
    <!-- NEW: ImageEditor components REPLACE figure tags entirely -->
    <ImageEditor
      v-for="editorId in openEditorIds"
      :key="editorId"
      :highlight-id="editorId"
      :initial-text="getHighlightText(editorId)"
      :style="getEditorPositionStyle(editorId)"
      class="absolute pointer-events-auto"
      @delete="closeImageEditor(editorId)"
    />
  </div>
</template>
```

Add new props and positioning logic to ImageLayer.vue:

```typescript
// Add to props
const props = defineProps<{
  highlights: Highlight[];
  selectedHighlights: Set<number>;
  openEditorIds: Set<number>;  // NEW: track which editors are open
}>()

// Helper to get highlight text by ID
function getHighlightText(highlightId: number): string {
  const highlight = props.highlights.find(h => h.id === highlightId)
  return highlight?.text || ''
}

// NEW: Position editors on RIGHT side, vertically aligned with highlights
function getEditorPositionStyle(highlightId: number) {
  const highlight = props.highlights.find(h => h.id === highlightId)
  if (!highlight) return {}
  
  const polygons = highlight.polygons[currentPage.value]
  if (!polygons || !polygons[0]) return {}
  
  // Vertical position: align with highlight's Y coordinate on PDF
  const top = Math.min(...polygons.map(p => p[1]))
  
  // Horizontal position: fixed at RIGHT edge of ImageLayer (512px from left)
  const left = 512
  
  return {
    top: `${top}px`,
    left: `${left}px`,
    width: '512px',  // Editor panel width
    zIndex: imageZIndices[highlightId] || 1
  }
}
```

**Pass openEditorIds to ImageLayer** from PdfViewer:

```vue
<ImageLayer
  ref="imageLayer"
  :highlights="highlights"
  :selected-highlights="selectedHighlights"
  :open-editor-ids="openEditorIds"
  @close-editor="closeImageEditor"
/>
```

**Key Architecture Notes**:
- **Complete Replacement**: The entire `<figure>` v-for loop is REMOVED, not modified
- **Physical Location**: Editors positioned on RIGHT side at `left: 512px` (right edge of ImageLayer)
- **Vertical Alignment**: Top position matches highlight's Y coordinate on PDF
- **Collapse Button**: Will protrude LEFT with `left: -20px` from editor's left edge
- **Image Inside**: Generated images rendered INSIDE ImageEditor component
- **Data Flow**: HighlightLayer emits "open-editor" → PdfViewer tracks IDs → ImageLayer renders editors

**Verification**:
- ImageEditor components render in ImageLayer's template (figure loop completely replaced)
- Each editor is positioned on RIGHT side of screen at `left: 512px`
- Editor top position vertically aligned with corresponding highlight on PDF
- Collapse button protrudes LEFT into PDF viewer area
- Multiple editors can be open simultaneously
- Editors receive highlightId and initialText as props
- Editors manage their own image state internally (no imageUrl/imageLoading on highlights)
- Generated images display INSIDE editor components
- Deleting an editor removes it from ImageLayer rendering

**Architecture Note**: This completely replaces the existing image generation workflow. ImageEditor components ARE the new way images are displayed - there is no separate image display.

---

- [ ] Task 5.2: Connect HighlightLayer to ImageEditor Opener
**User Story**: User Expands Editor for Existing Generated Image (Story 1)
**Depends on**: Task 5.1
**Effort**: 1 hour

**Data Flow**: HighlightLayer emits "open-editor" → PdfViewer tracks open editor IDs → ImageLayer renders ImageEditor for each open ID

Modify [services/frontend/app/components/HighlightLayer.vue](../../services/frontend/app/components/HighlightLayer.vue):

Replace the "gen-image" event with "open-editor":

```typescript
const emit = defineEmits<{
  select: [id: number];
  "open-editor": [id: number];  // Replaces "gen-image" event
}>();
```

In highlight dropdown menu, replace "Illustrate" button with "Enhance with AI":

```vue
<!-- In dropdown-content of HighlightLayer -->
<button class="btn btn-xs btn-secondary w-full"
  @click.stop="$emit('open-editor', highlight.id)"
>
  <Icon name="lucide:pencil" />
  Enhance with AI
</button>
```

Update PdfViewer to listen:

```vue
<HighlightLayer
  v-if="highlights.length"
  ...
  @open-editor="openImageEditor"
/>
```

**Verification**:
- "Enhance with AI" button visible in highlight menu (replaces old "Illustrate" button)
- Clicking button emits "open-editor" event to PdfViewer
- PdfViewer adds highlightId to openEditorIds set
- ImageLayer renders ImageEditor component in its v-for loop
- Editor displays at same vertical position as highlight

**Note**: This replaces the existing "Illustrate" button functionality entirely.

---

## Phase 6: UI Polish & Accessibility

- [ ] Task 6.1: Add Responsive Layout & Styling
**User Story**: All (UX polish)
**Depends on**: Task 4.1
**Effort**: 45 min

Update ImageEditor.vue styles and layout:

```vue
<template>
  <!-- Adjust to be a proper sidebar panel -->
  <div class="flex h-full bg-base-100">
    <!-- Minimize/Expand bar -->
    <div class="w-1 bg-primary/20 hover:bg-primary/50 transition cursor-col-resize flex items-start">
      <button
        class="btn btn-xs btn-ghost p-1"
        @click="isExpanded = !isExpanded"
        :title="isExpanded ? 'Collapse' : 'Expand'"
      >
        <Icon :name="isExpanded ? 'lucide:chevron-right' : 'lucide:chevron-left'" size="16" />
      </button>
    </div>

    <!-- Main content area -->
    <div v-show="isExpanded" class="flex-1 flex flex-col overflow-hidden">
      <div class="flex-1 overflow-y-auto p-4 space-y-4">
        <!-- Content here -->
      </div>
    </div>
  </div>
</template>

<style scoped>
:deep(.editor-content) {
  height: 100%;
  display: flex;
  flex-direction: column;
}
</style>
```

**Verification**:
- Editor panel is responsive on different screen sizes
- Textarea autofocuses when opening
- Scrolling works for long histories
- Colors and spacing match DaisyUI theme

---

- [ ] Task 6.2: Accessibility Audit
**User Story**: All (a11y compliance)
**Depends on**: Task 4.1
**Effort**: 30 min

Verify and update ImageEditor.vue:

- [ ] All buttons have aria-label or title
- [ ] Textarea has associated label
- [ ] Focus management works with Tab key
- [ ] Color contrast meets WCAG AA
- [ ] Loading states announced to screen readers
- [ ] Form fields properly labeled
- [ ] Keyboard navigation working

Update component:

```vue
<!-- Example improvements -->
<textarea
  v-model="currentPrompt"
  class="textarea textarea-bordered w-full h-24"
  placeholder="Enter or paste image prompt..."
  aria-label="Image prompt"
  :disabled="isLoading || isHistoryMode"
/>
```

**Verification**:
- Lighthouse accessibility score ≥ 90
- Keyboard navigation fully functional
- Screen reader announces all controls

---

## Summary of Dependencies

```
Task 1.1 (Types)
  ↓
Task 1.2 (State Type) ──→ Task 3.1 (useEditorHistory)
  ↓
Task 1.3 (Schema) ──→ Task 2.1 (Enhance Endpoint)
                          ↓
                       Task 4.2 (Enhance Handler)
                          ↓
                       Task 4.1 (Component Shell)
                          ↓
                       Task 4.3 (Generate Handler)
                          ↓
                       Task 5.1 (PdfViewer Integration)
                          ↓
                       Task 5.2 (HighlightLayer Connection)
                          ↓
                        Task 6.1 (Styling)
                        Task 6.2 (Accessibility)
```

---

## Rollout Strategy

1. **Phase 1-2**: Types & Backend (no UI changes) - safe to merge early
2. **Phase 3**: Composables (isolated, self-contained) - can be verified independently
3. **Phase 4**: Component (with feature flag initially?)
4. **Phase 5**: Integration (connects components together)
5. **Phase 6**: Polish & Accessibility (before production release)

For early validation, expose `openImageEditor` via a manual button or console API:

```typescript
// In PdfViewer or app-level composable
useNuxtApp().$editorDebug = { openImageEditor }
```

Then developers can call `window.__nuxt__.$editorDebug.openImageEditor(highlightId)` in console.

---

## Review

### Completed Tasks

#### Task 4.1: Create ImageEditor Component Shell ✅

**What was implemented**:
- Created `services/frontend/app/components/ImageEditor.vue` with full Vue 3 Composition API implementation
- Component receives `highlightId` and `initialText` as props
- Emits `delete` event when delete button is clicked
- Uses `useEditorHistory` composable for history management
- Collapse button positioned with `left: -20px` to protrude into PDF viewer area
- Component state includes:
  - `isExpanded`: toggle for expand/collapse
  - `currentPrompt`: current textarea value
  - `imageUrl`: ref to generated image URL
  - `enhanceLoading`: Enhance button loading state
  - `generateLoading`: Generate button loading state

**Critical Fixes Applied** (Post-Implementation):
- ✅ Removed explicit Vue imports (`import { computed, ref } from "vue"`) - Nuxt auto-imports these
- ✅ Added `addToHistory` to `useEditorHistory()` destructuring (prefixed with `_` for future use in Tasks 4.2-4.3)
- ✅ Removed unnecessary `imageLoading` ref - replaced with `generateLoading` state
- ✅ Updated image display to show during loading: `v-if="hasImage || generateLoading"`
- ✅ Added history text synchronization: `watch(currentHistoryItem)` updates textarea when navigating history
- ✅ Fixed memory leak: Added watchers to revoke blob URLs on change and component unmount
- ✅ TypeScript compiles without errors, ESLint passes all checks

**Component Features**:
- **Expanded view**: Shows textarea, Enhance/Generate buttons, image display, history navigation, delete button
- **Collapsed view**: Shows collapse button + generated image (if exists)
- **Image display**: Images render INSIDE the editor component below buttons
- **History section**: Displays when `history.length > 0` with previous/next navigation
- **Deduplication**: Managed by `useEditorHistory` composable
- **Placeholder methods**: `handleEnhance()` and `handleGenerate()` ready for Tasks 4.2-4.3

**Verification Checklist**:
- ✅ TypeScript compilation: No errors
- ✅ ESLint validation: Passes all checks
- ✅ Component renders with collapse button visible on LEFT edge
- ✅ Collapse button positioned with negative left value (left: -20px)
- ✅ Click expand/collapse toggles `isExpanded` visibility
- ✅ Textarea visible when no image exists (`!hasImage`)
- ✅ Generated image displays INSIDE editor (below textarea/buttons)
- ✅ When collapsed with image: only collapse button + image visible
- ✅ Delete button shows when expanded
- ✅ History section displays when history has entries
- ✅ All content positioned on RIGHT side of screen (component width: 100%)
- ✅ Styles apply correctly with DaisyUI classes
- ✅ Transition effects for slide animation
- ✅ No explicit Vue imports (uses Nuxt auto-imports)
- ✅ Memory leak prevention: blob URLs revoked properly
- ✅ History navigation updates textarea content

**Architecture Decisions**:
- Image state (`imageUrl`) owned by ImageEditor component, not highlight
- History state managed independently by each editor via composable
- No parent state management required for individual editors
- Multiple editors can coexist without state conflicts
- Ready for integration with PdfViewer (Task 5.1) and HighlightLayer (Task 5.2)

**Next Tasks**:
- Task 4.2: Implement Enhance handler (`handleEnhance()`)
- Task 4.3: Implement Generate handler (`handleGenerate()`)
- Task 5.1: Integrate with PdfViewer to manage `openEditorIds`
- Task 5.2: Update HighlightLayer to render ImageEditor components

