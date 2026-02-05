# Expandable Image Editor Component - Planning

## Project Classification
- **Type**: UI Feature
- **Size**: Medium
- **Effort Estimate**: 5-7 days
- **Date**: February 5, 2026

---

## Goals

Enable flexible image prompt editing with a chainable enhancement pipeline and comprehensive history tracking. Users should be able to:
1. Create independent, expandable editors for each image generation workflow
2. Refine image prompts iteratively through an "Enhance" mechanism
3. Track all text edits and generated images in a per-editor history
4. Quickly toggle editor visibility without losing state

---

## User Stories

### Story 1: User Expands Editor for Existing Generated Image
**As a** user with a generated image shown in the sidebar,
**I want to** expand an editor panel to refine the image further,
**So that** I can iteratively improve the generated result.

**Acceptance Criteria**:
- Editor expands on right side of screen when expand button is clicked
- If image already generated, prompt/button area is hidden by default (collapsed state)
- Collapsing hides the editor without losing state
- Expand button remains visible at right edge at all times

### Story 2: User Enhances Prompt Text
**As a** user with an open editor,
**I want to** type changes to a prompt text and send them to an Enhance endpoint,
**So that** I can refine my image description iteratively.

**Acceptance Criteria**:
- Textarea accepts user input without character limits
- "Enhance" button calls `/api/enhance` with current text
- Response text becomes the new prompt in the editor
- Each enhance step is tracked in history with both input and output text
- User can chain multiple enhance calls sequentially

### Story 3: User Generates Image from Prompt
**As a** user with an edited/enhanced prompt,
**I want to** click "Generate" to create an image from the refined text,
**So that** I can see the visual result of my edits.

**Acceptance Criteria**:
- "Generate" button calls existing `/api/gen-image-bytes` endpoint with textarea text
- Generated image is added to the sidebar image layer (existing behavior)
- Generate action is tracked in history
- Textarea area collapses automatically after generation (prompt area hidden)
- Editor remains open and available for further refinement

### Story 4: User Navigates Editor History
**As a** user with multiple text edits in the editor,
**I want to** see previous states of my prompt using arrow buttons,
**So that** I can review or revert to earlier versions.

**Acceptance Criteria**:
- History shows all text input steps (to Enhance) and output steps (from Enhance)
- History also includes Generate trigger points
- Deduplication: skip adding to history if text exactly matches the previous item
- Image URLs in history deduplicated (same image object reused if URL matches)
- Prev/Next arrows disabled at boundaries
- Navigation updates textarea with historical text (read-only state)
- Clicking "Edit" from historical state resumes editing from that point

### Story 5: User Deletes Editor
**As a** user with an open image editor,
**I want to** delete the editor to clean up the workspace,
**So that** I can focus on other images.

**Acceptance Criteria**:
- Delete button removes the entire ImageEditor component
- Associated highlight object remains untouched (read-only from editor perspective)
- No state is persisted to storage (ephemeral editor state)

---

## Functional Requirements

### Component Architecture
1. **ImageEditor.vue** - Reusable component instantiated once per image/editor workflow
   - Props: `highlightId` (number), `initialText` (string from highlight.text)
   - Each component manages its own editor state independently
   - **Editors own all image state**: `imageUrl`, `imageLoading` stored internally
   - No parent component state management required (except ordering)
   - Multiple editors can be open simultaneously
   - **Physical Location**: ImageEditor panels appear on the **RIGHT side** in the 512px ImageLayer space
   - **Vertical Alignment**: Positioned to vertically align with their corresponding highlights on the PDF
   - **Complete Replacement**: ImageEditor components **REPLACE** the current `<figure>` image display entirely
   - **Image Inside Editor**: Generated images are rendered **INSIDE** the ImageEditor component (below textarea/buttons)
   - **No Separate Image Display**: The editor IS the image display - there is no separate image rendering
   
2. **Editor UI Layout**
   ```
   PDF Viewer (512px)  │  ImageLayer (RIGHT side, 512px)
   ────────────────────┼──────────────────────────────────
                       │
                    [◀]│  ← Collapse button (protrudes LEFT with negative left value)
                       │
                       │  [WHEN EXPANDED - RIGHT SIDE]
                       │  ┌──────────────────────────┐
                       │  │ Top Bar (controls)       │
                       │  ├──────────────────────────┤
                       │  │ Prompt Textarea          │
                       │  │ Enhance | Generate       │
                       │  ├──────────────────────────┤
                       │  │ Generated Image          │  ← Rendered INSIDE editor
                       │  ├──────────────────────────┤
                       │  │ History Display          │
                       │  │ (Prev | Item | Next)     │
                       │  │ [Edit] / [Clear]         │
                       │  ├──────────────────────────┤
                       │  │ Delete Button            │
                       │  └──────────────────────────┘
                       │
                    [◀]│  [WHEN COLLAPSED]
                       │  [Generated Image if exists]
   ```
   
   **Key Layout Points**:
   - Collapse button uses negative `left` value (e.g., `left: -20px`) to protrude into PDF viewer area
   - This bridges the gap between PDF and editor panels visually
   - Entire editor panel is positioned on the RIGHT side at right edge of ImageLayer
   - Vertical position (top) aligns with highlight's Y coordinate on PDF
   - Horizontal position is fixed at right edge
   - Generated image rendered below textarea/buttons when expanded, or below collapse button when collapsed

3. **State Management (per editor)**
   ```typescript
   type EditorState = {
     highlightId: number;       // reference to read-only highlight
     isExpanded: boolean;       // visibility toggle
     currentPrompt: string;     // textarea content
     imageUrl: string | null;   // generated image (owned by editor)
     imageLoading: boolean;     // loading state (owned by editor)
     history: EditorHistoryItem[]; // array of text/image snapshots
     historyIndex: number;      // current position (-1 = editing mode)
     isLoading: boolean;        // for enhance/generate operations
   }
   ```

4. **History Entry Structure**
   ```typescript
   type EditorHistoryItem = {
     id: string;                 // unique identifier
     type: 'text_input' | 'text_output' | 'generate';
     text: string;               // the prompt text at this step
     imageUrl?: string;          // if type === 'generate', the resulting image
     timestamp: number;
   }
   ```

### State Management Rules
- **History triggers**: Add entry only on:
  1. User submits text to Enhance endpoint (both input and output)
  2. User clicks Generate button (records the generation action)
- **Collapse behavior**: Toggling visibility does NOT trigger history (ephemeral UI state)
- **Delete behavior**: Removes component entirely; highlight remains untouched
- **No persistence**: Editor state not saved to localStorage or backend (session-only)

### History Deduplication
1. **Text deduplication**: If new entry text matches immediately previous entry text, skip adding
2. **Image deduplication**: Reuse same image object if URL matches previous generation
3. **No history limit**: Keep all entries (no truncation)

### API Integration
- **POST `/api/enhance`**
  - Request: `{ text: string }`
  - Response: `{ text: string }`
  - Use case: Refine/improve user-provided prompt text
  - Chainable: Output can be used as input for next Enhance call
  - Stub implementation acceptable for MVP

- **POST `/api/gen-image-bytes`** (existing)
  - Already implemented, used for image generation
  - Called with `{ text: string }` from editor textarea

---

## Non-Functional Requirements

### Performance
- Editors should lazy-load only when expanded
- No excessive re-renders when navigating history (use computed properties)
- History array should not impact UI performance even with 50+ entries

### Accessibility
- Expand/collapse button has clear title/aria-label
- Textarea has proper labels and focus management
- Keyboard navigation: Tab through Enhance/Generate buttons, Prev/Next arrows
- History navigation accessible via keyboard (arrow keys, Enter)

### Maintainability
- Follow existing Vue 3 Composition API patterns
- Use DaisyUI components to match visual style
- Component should be self-contained (no external state deps)
- Clear TypeScript types for all internal state
- Reusable composable for history logic (`useEditorHistory`)

---

## Current State Analysis

### Existing ImageLayer Pattern
The [ImageLayer.vue](../../services/frontend/app/components/ImageLayer.vue) component provides reference patterns:
- **Reactive maps** for position/z-index: `imagePositions`, `imageZIndices`
- **Dragging logic**: Pointer event handling with drag state management
- **Style binding**: Dynamic positioning using `StyleValue` from Vue
- **Deletion**: Button with hover controls, click handler removes from parent array
- **Multiple instances**: Handles multiple images simultaneously with independent state

**Key Code Pattern**:
```vue
<script setup lang="ts">
const imagePositions = reactive<Record<number, ImagePosition>>({});
const imageZIndices = reactive<Record<number, number>>({});

function deleteImage(highlight: Highlight) {
  // Component removes self or parent removes component
}
</script>
```

### Component Hierarchy (Updated Architecture)

**Data Flow**:
```
HighlightLayer "Illustrate" button
  ↓ emits "open-editor" event with highlightId
PdfViewer
  ↓ tracks openEditorIds Set<number>
  ↓ passes openEditorIds to ImageLayer
ImageLayer.vue
  ↓ v-for loop: for each highlightId in openEditorIds
  ↓ renders <ImageEditor> component
  ↓ positioned on RIGHT side, vertical alignment with PDF highlight
ImageEditor
  ↓ manages imageUrl, imageLoading state internally
  ↓ renders generated image INSIDE component
  ↓ handles all image generation, display, and history
```

**Key Architecture Points**:
1. **PdfViewer** maintains `openEditorIds` Set to track which editors are open
2. **ImageLayer.vue** receives `openEditorIds` prop and renders `<ImageEditor>` components in its template v-for loop
3. **Complete Replacement**: ImageEditor components **REPLACE** the current `<figure>` tag loop entirely - not supplemental
4. **Physical Location**: Editors positioned on **RIGHT side** of screen, in the 512px ImageLayer space
5. **Vertical Positioning**: Top position aligned with highlight's Y coordinate on the PDF
6. **Horizontal Positioning**: Fixed at right edge of ImageLayer (not "at highlight coordinates")
7. **Collapse Button**: Protrudes slightly LEFT into PDF viewer area using negative left value
8. **Image Display**: Generated images rendered **INSIDE** the ImageEditor component - no separate image display
9. ImageEditor owns all image state (`imageUrl`, `imageLoading`) - no prop mutation

**OLD Pattern (Completely Replaced)**:
```vue
<!-- ImageLayer.vue BEFORE: This entire pattern is REMOVED -->
<template>
  <div v-for="highlight in highlights.filter(h => h.imageUrl || h.imageLoading)" :key="highlight.id">
    <figure :style="getHighlightImageStyle(highlight)">
      <img v-if="highlight.imageUrl" :src="highlight.imageUrl" />
      <div v-if="highlight.imageLoading">Loading...</div>
    </figure>
  </div>
</template>
```

```typescript
// This pattern is REMOVED - no longer using "Illustrate" button
const highlights = defineModel<Highlight[]>("highlights", { required: true });

async function genImage(highlightId: number) {
  const realHighlight = highlights.value.find(h => h.id === highlightId);
  realHighlight.imageLoading = true;  // ❌ REMOVED - violated single source of truth
  const res = await call("/api/gen-image-bytes", {
    method: "POST",
    body: { text: realHighlight.text }
  });
  realHighlight.imageUrl = url;  // ❌ REMOVED - caused prop mutation
}
```

**NEW Pattern (Complete Replacement)**:
```vue
<!-- ImageLayer.vue AFTER: ImageEditor components REPLACE figure tags -->
<template>
  <ImageEditor
    v-for="editorId in openEditorIds"
    :key="editorId"
    :highlight-id="editorId"
    :initial-text="getHighlightText(editorId)"
    :style="getEditorPositionStyle(editorId)"
  />
</template>
```

**Key Changes**:
- The entire `<figure>` v-for loop is **REPLACED**, not supplemented
- ImageEditor components ARE the new image display system
- Editors receive `highlightId` and `initialText` as props
- Editors manage their own image state internally (`imageUrl`, `imageLoading`)
- Editors render generated images INSIDE themselves (below textarea/buttons)
- Editors positioned on RIGHT side, vertically aligned with PDF highlights

### Highlight Type Definition
From [types/common.d.ts](../../services/frontend/app/types/common.d.ts):
```typescript
export type Highlight = {
  id: number;
  text: string;
  polygons: Record<number, number[][]>;
  score?: number;
  score_received_at?: string;
};
```

**Key Architecture Change**: `imageUrl` and `imageLoading` have been **removed** from the Highlight type. These fields caused prop mutation issues and violated single source of truth principles.

**New Pattern**: Editors own their own image state completely. Image generation results are stored in editor-specific state, not on the highlight object. Highlights remain truly read-only and contain only document-related data.

---

## Key Files Involved

| File | Role | Changes |
|------|------|---------|
| [services/frontend/app/components/ImageEditor.vue](../../services/frontend/app/components/ImageEditor.vue) | **New** | Main editor component with expand/collapse, textarea, history |
| [services/frontend/app/composables/useEditorHistory.ts](../../services/frontend/app/composables/useEditorHistory.ts) | **New** | Composable for history management and deduplication |
| [services/frontend/app/components/PdfViewer.vue](../../services/frontend/app/components/PdfViewer.vue) | Integrate | Add ImageEditor instances alongside ImageLayer; manage editor array |
| [services/frontend/app/types/common.d.ts](../../services/frontend/app/types/common.d.ts) | Extend | Add `EditorHistoryItem` and `EditorState` types; remove `imageUrl`/`imageLoading` from Highlight |
| [services/api/core/api.py](../../services/api/core/api.py) | **New endpoint** | Add `/api/enhance` stub endpoint |
| [services/api/core/schemas.py](../../services/api/core/schemas.py) | Extend | Add request/response schema for Enhance endpoint |

---

## Existing Patterns to Follow

### 1. Composition API with Reactive State
**Pattern used in**: ImageLayer.vue, PdfViewer.vue
```typescript
const imagePositions = reactive<Record<number, ImagePosition>>({});
const draggingId = ref<number | null>(null);

// For Maps:
const imageEditors = reactive(new Map<number, EditorState>());
```

**Apply to ImageEditor**: Use `reactive` for editor state object, `ref` for primitives. For Map/Set collections, wrap in `reactive()` directly without type parameter.

### 2. Type-safe Event Handling
**Pattern used in**: HighlightLayer.vue
```typescript
const props = defineProps<{
  highlights: Highlight[];
  selectedHighlights: Set<number>;
}>();

const emit = defineEmits<{
  select: [id: number];
  "gen-image": [id: number];
}>();
```

**Apply to ImageEditor**: Emit `delete` event when user clicks delete; parent removes component. Use tuple syntax for emit typing:
```typescript
const emit = defineEmits<{
  delete: [];
}>();
```

### 3. Computed Styles with Dynamic Values
**Pattern used in**: ImageLayer.vue
```typescript
function getHighlightImageStyle(highlight: Highlight) {
  return {
    top: `${top}px`,
    left: `${left}px`,
    zIndex: imageZIndices[highlight.id] || 1
  };
}
```

**Apply to ImageEditor**: Compute visibility classes based on `isExpanded`, `hasImage`, `isHistoryMode`.

### 4. API Integration Pattern
**Pattern used in**: PdfViewer.vue
```typescript
const { $api: call } = useNuxtApp();

const res = await call("/api/gen-image-bytes", {
  method: "POST",
  body: { text: someText }
});
```

**Apply to ImageEditor**: Use same `$api` pattern for `/api/enhance` and `/api/gen-image-bytes` calls.

### 5. Loading State Management
**OLD Pattern (No Longer Used)**:
```typescript
// This pattern is DEPRECATED - don't mutate highlight props
realHighlight.imageLoading = true;  // ❌ DON'T DO THIS
const res = await call(/* ... */);
realHighlight.imageLoading = false;  // ❌ DON'T DO THIS
```

**NEW Pattern for ImageEditor**: 
```typescript
// Editors manage their own loading states
const imageLoading = ref(false);
const isLoading = ref(false);

async function handleGenerate() {
  isLoading.value = true;
  imageLoading.value = true;
  const res = await call(/* ... */);
  imageUrl.value = url;
  imageLoading.value = false;
  isLoading.value = false;
}
```

**Apply to ImageEditor**: Manage local `isLoading` and `imageLoading` state internally; disable buttons during async operations.

### 6. DaisyUI Component Usage
**Pattern used throughout**: 
- Buttons: `btn btn-sm btn-primary`, `btn-xs`, `btn-circle`, `btn-error`
- Forms: `textarea`, `input`, `label`
- Icons: `Icon name="lucide:x"` from `@nuxt/icon`
- Layout: Flexbox with `flex`, `gap-2`, `justify-between`

**Apply to ImageEditor**: Use consistent DaisyUI button styles and Lucide icons.

---

## Decisions Made During Clarification

1. **Per-image editors**: Each image gets its own `ImageEditor.vue` component with independent state (not a global modal)
2. **Always-visible expand button**: Right edge placement ensures accessibility even when collapsed
3. **Prompt area hidden on generation**: UX improvement to reduce clutter after image is created
4. **History deduplication**: Prevents redundant entries when user cancels edits or enhance returns same text
5. **No state persistence**: Session-only (ephemeral) for simplicity; can add localStorage later
6. **Chainable enhance**: Multiple enhance calls allowed; each step tracked in history
7. **Independent z-ordering**: Editors don't interact with image z-index management (handled by ImageLayer)
8. **Keep all complexity**: Enhance API feature, history deduplication, keyboard navigation, and separate ImageEditor component all remain (no MVP reduction)

## Critical Architecture Decisions (Post-Review)

### Decision 1: Remove imageUrl/imageLoading from Highlight Type (Complete Refactor)
**Problem**: Original design had `imageUrl` and `imageLoading` fields on the Highlight type, which created prop mutation issues and violated single source of truth principles.

**Solution**: 
- **Removed** `imageUrl` and `imageLoading` from Highlight type entirely
- Editors now own their own image state completely
- Highlights remain truly read-only with only: `id`, `text`, `polygons`, `score`, `score_received_at`
- No prop mutation issues since these fields don't exist on highlights anymore

**Impact**:
- This is a **complete refactor**, not a migration
- The existing "Illustrate" button workflow is being replaced entirely
- No imageUrl/imageLoading fields exist on highlights at all
- ImageEditor components manage all image state internally

**Benefits**:
- Clean separation of concerns
- No prop mutation warnings
- Editors are fully independent and self-contained
- Highlights remain document-focused (not UI state)

### Decision 2: ImageEditor Components - Complete Replacement Architecture
**Layout Architecture**: ImageEditor components **COMPLETELY REPLACE** the current `<figure>` image display in ImageLayer.vue.

**Physical Location**:
- **RIGHT Side Positioning**: ImageEditor panels appear on the **RIGHT side** in the 512px ImageLayer space
- **Vertical Alignment**: Top position aligned with corresponding highlight's Y coordinate on the PDF
- **Horizontal Alignment**: Fixed at right edge of ImageLayer (not at highlight's X coordinate)
- **Collapse Button**: Protrudes slightly LEFT into PDF viewer area using negative left value (e.g., `left: -20px`)
- **Visual Bridge**: Collapse button bridges the gap between PDF viewer and editor panels

**Complete Replacement Details**:
- The entire `<figure v-for="highlight in highlights.filter(h => h.imageUrl || h.imageLoading)">` loop is **REMOVED**
- Replaced with: `<ImageEditor v-for="editorId in openEditorIds">` loop
- ImageEditor components ARE the new way images are displayed
- No separate image display exists - editors render images INSIDE themselves
- Generated images appear below textarea/buttons within the editor component

**Implementation Details**:
- PdfViewer tracks which editors are open via `openEditorIds` Set
- ImageLayer receives `openEditorIds` prop
- ImageLayer's v-for loop iterates through `openEditorIds` (not highlights)
- Each ImageEditor positioned absolutely on RIGHT side, vertically aligned with its highlight
- Multiple editor components can be open simultaneously
- HighlightLayer's "gen-image" event becomes "open-editor" event

**Template Structure**:
```vue
<!-- BEFORE (REMOVED): -->
<figure v-for="highlight in highlights.filter(h => h.imageUrl || h.imageLoading)" ...>

<!-- AFTER (NEW): -->
<ImageEditor v-for="editorId in openEditorIds" ... />
```

**Benefits**:
- Crystal clear that ImageEditor IS the image display (not supplemental)
- Clean separation: PDF on left, editors on right
- Collapse button visually connects the two areas
- Vertical alignment maintains spatial relationship with highlights
- No prop mutation issues
- Clean component ownership of all image state

---

## Success Criteria

- [x] Component can be instantiated for multiple images simultaneously
- [x] Expand/collapse works without losing editor state
- [x] Text input and Enhance endpoint integration functional
- [x] Generate button creates images using existing endpoint
- [x] History tracks text edits and generations accurately
- [x] Deduplication prevents redundant history entries
- [x] Navigation through history works without data loss
- [x] Delete removes only the editor, not the highlight
- [x] UI matches existing design system (DaisyUI, Tailwind)
- [x] TypeScript strict mode compliance
- [x] No external state management dependencies
- [x] Editors own their own image state (no prop mutation)
- [x] Highlights remain read-only document entities

---

## What's NOT Changing

### Features to Keep (No MVP Reduction)
- ✅ **Enhance API feature** - Full implementation with chainable enhance calls
- ✅ **History deduplication logic** - As specified, with text and image deduplication
- ✅ **Keyboard navigation** - Ctrl+Arrow keys for history, Ctrl+Enter to generate
- ✅ **Separate ImageEditor component** - Not inline in ImageLayer, proper component separation
- ✅ **Multiple simultaneous editors** - Full support for multiple open editors
- ✅ **History display with prev/next** - Complete history UI with navigation
- ✅ **Edit from history** - Ability to resume editing from any history point

