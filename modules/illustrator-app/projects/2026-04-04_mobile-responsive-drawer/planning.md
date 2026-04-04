---
type: improvement
size: M
---

# Mobile Responsive Drawer Layout

## Overview

The current top bar in the index page stacks all controls vertically on mobile (below the `lg` breakpoint), consuming significant screen real estate and creating a poor experience on small screens. This project replaces the mobile layout with a collapsed top bar showing only the burger menu icon and file input, while all other controls (ModelSelect, EvalProgress, AutoIllustration, Export, HighlightNav, ThemeToggle) move into a custom slide-in drawer panel that appears from the left side.

## Goals

- Reclaim vertical screen space on mobile by collapsing toolbar controls into a drawer
- Keep the primary user entry point (file input) always visible on mobile
- Maintain the current desktop layout unchanged (at and above `lg` breakpoint)
- Ensure all controls remain accessible on mobile via the drawer
- Avoid duplicate DOM elements to preserve HelpOverlay compatibility
- Disable HelpOverlay on mobile to avoid spotlight issues with drawer-resident controls

## User Stories

### As a mobile user, I want a compact top bar so that I can see more document content

**Acceptance criteria:**

- [ ] Below `lg` breakpoint, the top bar shows only a burger menu icon and the file input
- [ ] The top bar height on mobile is a single row (`h-14.5`, 58px — matching desktop)
- [ ] At `lg` and above, the layout is identical to the current three-section row

### As a mobile user, I want to open a side drawer to access all controls

**Acceptance criteria:**

- [ ] Tapping the burger icon opens a custom slide-in panel from the left side
- [ ] The drawer contains: ModelSelect, EvalProgress, AutoIllustration, Export button, HighlightNav, ThemeToggle
- [ ] The drawer has a semi-transparent backdrop (`bg-black/50`) below `top-14.5` that dismisses it on tap
- [ ] A close button inside the panel can also dismiss it
- [ ] Controls inside the drawer are fully functional (not just visual)

### As a mobile user, I want the drawer to close when I interact with the document

**Acceptance criteria:**

- [ ] Selecting a file closes the drawer (if open)
- [ ] Tapping the backdrop closes the drawer
- [ ] A close button inside the drawer closes it

### As a desktop user, I want no changes to my experience

**Acceptance criteria:**

- [ ] At `lg` and above, no burger icon is shown
- [ ] At `lg` and above, no drawer exists; all controls remain inline in the top bar
- [ ] No regressions in existing desktop control behavior

## Requirements

### Functional

- Use `v-if` with a screen-size reactive ref so only ONE copy of each control exists in the DOM at a time
- On mobile (`< lg`): render burger + file input in top bar, controls in drawer panel
- On desktop (`>= lg`): render the normal inline top bar with all controls
- Drawer is a custom Tailwind slide-in panel (not DaisyUI drawer component)
- File input remains in the top bar on mobile
- All `data-help-target` attributes preserved on controls (HelpOverlay compatibility)
- EvalProgress remains visible and functional inside the drawer (progress updates in real time)

### Non-functional

- No new dependencies; use VueUse `useMediaQuery` (already available via `@vueuse/nuxt`)
- Slide transition should feel smooth (CSS `transform transition-transform`)
- Controls in the drawer should be laid out vertically with reasonable spacing
- Drawer z-index: backdrop at `z-[200]`, panel at `z-[210]` -- above PdfViewer toolbar (`z-120`) and bottom bar (`z-130`)

## Scope

### Included

- Refactoring the top bar section of `pages/index.vue` to conditionally render controls via `v-if`
- Adding a custom slide-in drawer panel for mobile controls
- Adding a burger menu button with responsive visibility
- Creating or using a composable for mobile detection (`useMediaQuery`)
- Vertical stacking of controls inside the drawer
- Updating hero min-height in `main.css`
- HighlightNav overflow: only addressed if it is still a problem within the new drawer layout

### Excluded

- Changes to the bottom bar layout
- Changes to the Hero component internals
- Changes to any component internals (ModelSelect, EvalProgress, etc.)
- Desktop layout modifications
- New components (this is purely a layout restructuring in `index.vue`)

---

## Current State

The index page top bar uses a `flex flex-col lg:flex-row` layout. On mobile (below `lg`/1024px), all three control groups stack vertically:

1. **Row 1**: File input + ModelSelect
2. **Row 2**: EvalProgress + AutoIllustration + Export button
3. **Row 3**: HighlightNav + ThemeToggle

This creates a tall top bar that pushes document content far down the viewport. The top bar class is:
```
top-bar w-full px-3 py-2 bg-base-200 flex flex-col lg:flex-row gap-3 justify-between lg:items-center sticky top-0 z-50 lg:h-14.5
```

The hero component also has a breakpoint-dependent `min-height` in `main.css` that accounts for the current top bar height (114px on mobile vs 58px on desktop, with a media query at `40rem`). Since both bars will be 58px after this project, the media query will be removed entirely in favour of a single rule.

## Key Files

- `services/frontend/app/pages/index.vue` - Main page with top bar layout; primary file to modify
- `services/frontend/app/assets/css/main.css` - Hero min-height calculation that depends on top bar height; needs breakpoint update
- `services/frontend/app/components/HighlightNav.vue` - Navigation dropdown; will live inside drawer on mobile
- `services/frontend/app/components/ModelSelect.vue` - Model selector dropdown; will live inside drawer on mobile
- `services/frontend/app/components/EvalProgress.vue` - Progress indicator; will live inside drawer on mobile
- `services/frontend/app/components/AutoIllustration.vue` - Auto-illustration toggle and settings; will live inside drawer on mobile
- `services/frontend/app/components/ThemeToggle.vue` - Theme switch; will live inside drawer on mobile
- `services/frontend/app/composables/useHelpSteps.ts` - Help overlay that targets `data-help-target` attributes; no changes needed (v-if ensures single DOM copy)
- `services/frontend/app/components/PdfViewer.vue` - Contains page toolbar at `z-120`; drawer must layer above it

## Existing Patterns

### Responsive Visibility in Tailwind
The codebase already uses `lg:` prefix for breakpoint behavior. The pattern for mobile-only / desktop-only:
- Mobile only: `lg:hidden`
- Desktop only: `hidden lg:flex` (or `hidden lg:block`)

### VueUse Media Query
The project already includes `@vueuse/nuxt` (v13.9.0). Use `useMediaQuery` for reactive screen-size detection:
```ts
const isMobile = useMediaQuery('(max-width: 1023px)')
```
This returns a reactive `Ref<boolean>` that updates on window resize. Use with `v-if` in templates.

### Component Arrangement in Top Bar
Controls are grouped into three `div` sections with `flex items-center gap-2`. This grouping should be preserved inside the drawer for visual consistency.

## Decisions

### Custom Tailwind Drawer (No DaisyUI Drawer)

**Decision**: Build a custom slide-in panel using plain Tailwind utilities instead of wrapping the page in `<div class="drawer">`.

**Rationale**: The DaisyUI drawer component requires wrapping the entire page in a `<div class="drawer">` container, which risks layout side effects (e.g., `min-h-screen` conflicts, extra nesting). A custom panel is simpler, avoids page-level restructuring, and gives full control over z-index and animation.

**Implementation**:
- A fixed overlay backdrop: `fixed top-14.5 left-0 right-0 bottom-0 bg-black/50 z-[200] transition-opacity duration-300`. Always in DOM when `isMobile` is true; visibility controlled via conditional classes (`:class="drawerOpen ? 'opacity-100' : 'opacity-0 pointer-events-none'"`).
- A fixed side panel: `fixed top-14.5 left-0 bottom-0 w-72 bg-base-200 z-[210] transform transition-transform duration-300`. Always in DOM when `isMobile` is true; visibility controlled via CSS translate.
- Panel slides in with `translate-x-0` (open) and `-translate-x-full` (closed)
- A close button inside the panel
- Controlled by a plain `drawerOpen` ref (`ref<boolean>(false)`), not a checkbox
- Close on backdrop click and on file upload
- Both panel and backdrop start at `top-14.5` so they do not cover the mobile top bar (which is `sticky top-0 z-50 h-14.5`)

### v-if Mobile Detection (No Duplicate DOM)

**Decision**: Use `v-if` with VueUse `useMediaQuery` so only one copy of each control exists in the DOM at a time.

**Rationale**: Using `hidden`/`lg:flex` CSS classes would duplicate every control in the DOM (one in the top bar, one in the drawer). This breaks HelpOverlay, which uses `querySelector` to find `data-help-target` elements -- duplicate targets cause it to find the wrong (hidden) element. With `v-if`, only the mobile OR desktop version is rendered, so `querySelector` always finds the correct visible element. No changes to `useHelpSteps` are needed.

**Implementation**:
```ts
const isMobile = useMediaQuery('(max-width: 1023px)')
```
- `v-if="!isMobile"` renders the inline desktop top bar (existing structure)
- `v-if="isMobile"` renders the mobile top bar (burger + file input) and the custom drawer panel

### Mobile Top Bar Contents

**Decision**: Show only burger icon and file input on mobile.

**Rationale**: The file input is the primary entry point for the app. Keeping it always visible ensures the core workflow is one tap away. All other controls are secondary and can live behind a menu.

### Drawer Z-Index Layering

**Decision**: Backdrop at `z-[200]`, panel at `z-[210]`.

**Rationale**: The PdfViewer toolbar is at `z-120` and the bottom bar is at `z-130`. The drawer must appear above both. Using `z-[200]` and `z-[210]` provides clear separation with room for future layers. Covering the bottom bar with the backdrop when the drawer is open is intentional; users interact with the drawer or tap the backdrop to close it.

### Hero Height — Single Rule

**Decision**: Remove the `@media` block in `main.css` entirely. Set a single rule: `.hero { min-height: calc(100vh - 58px); }` with no breakpoint. The mobile top bar now has explicit `h-14.5` (58px), matching the desktop bar, so one rule covers both.

**Rationale**: The current `40rem` breakpoint does not match `lg`, and with both bars at the same height, the media query is unnecessary. A single rule is simpler and correct.

### HighlightNav Overflow

**Decision**: Do not fix HighlightNav overflow independently. Only address if it remains a problem within the drawer layout.

**Rationale**: The drawer provides a wider, vertically-oriented container that may resolve overflow issues naturally. Fixing it separately risks unnecessary work.

### SSR — Non-Issue

**Decision**: `ssr: false` is confirmed in `nuxt.config.ts`. The `useMediaQuery` composable will not cause hydration mismatches because there is no server-side rendering. Do not add `<ClientOnly>` wrappers around mobile/desktop branches — they are unnecessary and add complexity.

### highlightNav Ref in Both Branches

**Decision**: The `ref="highlightNav"` attribute must be present on the `HighlightNav` component instance in both the mobile drawer branch and the desktop top bar branch. Since `v-if` guarantees only one branch is mounted at a time, the same Vue template ref name works correctly — it always points to the single mounted instance.

### HelpOverlay on Mobile — Hidden

**Decision**: HelpOverlay is desktop-only. On mobile (`isMobile` is true), the help button is hidden entirely. The help button in `index.vue` has two render branches (a tooltip-animated `v-if` branch and a `v-else` branch), and they cannot be cleanly disabled with a single condition. Instead, wrap the entire bottom-bar help section (both branches, lines 73-97) in a `v-if="!isMobile"` condition so the button is not rendered at all on mobile. This avoids spotlight positioning issues with drawer-resident controls.

### DownloadConfirmDialog Z-Index — Close Drawer Before checkScorer

**Decision**: `DownloadConfirmDialog` renders at `z-50`. Close the drawer in both code paths that call `checkScorer`: the `handleFileUpload` function and the `@request-model-download` handler (emitted by `ModelSelect` inside the drawer). In the `@request-model-download` handler in `index.vue`, set `drawerOpen.value = false` before calling `checkScorer`. This ensures the drawer and dialog are never visible simultaneously and no z-index adjustment is needed.

### Backdrop and Panel Rendering Strategy

**Decision**: When `isMobile` is true, both the drawer panel and the backdrop are always present in the DOM (not toggled with `v-if` or `v-show`). The panel uses CSS `translate-x` transform for slide animation. The backdrop uses conditional classes for opacity: `:class="drawerOpen ? 'opacity-100' : 'opacity-0 pointer-events-none'"` with static `transition-opacity duration-300` classes. Using `v-show` would set `display: none`, which bypasses CSS opacity transitions. Neither element is conditionally created/destroyed during open/close — only when `isMobile` changes.

### Mobile Top Bar Height

**Decision**: The mobile top bar div uses `h-14.5` (58px) with `items-center` for vertical centering. Do not include `py-2` from the original top bar class list — with an explicit height, vertical padding would squeeze the content inside the 58px box. This matches the desktop behavior of `lg:h-14.5`. The drawer panel and backdrop starting at `top-14.5` align perfectly with the bottom edge of the mobile top bar.
