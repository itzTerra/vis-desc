# Tasks

## Components

- [x] Add `isMobile` reactive ref using VueUse `useMediaQuery` in `index.vue`
  - `const isMobile = useMediaQuery('(max-width: 1023px)')`
  - This drives all `v-if` conditional rendering below

- [x] Create desktop top bar (renders when `v-if="!isMobile"`)
  - Wrap the existing three-section inline top bar in a `v-if="!isMobile"` block
  - No changes to desktop layout structure, classes, or behavior
  - All controls render inline as they do today
  - Ensure `ref="highlightNav"` is on the `HighlightNav` component instance

- [x] Create mobile top bar (renders when `v-if="isMobile"`)
  - Single-row bar with burger menu button and file input only
  - Remove `py-2` from the mobile top bar div and use `h-14.5` with `items-center` for vertical centering, matching the desktop behavior of `lg:h-14.5`
  - Burger button toggles `drawerOpen` ref
  - Same sticky positioning (`sticky top-0 z-50`) and `bg-base-200` styling as desktop bar

- [x] Build custom slide-in drawer panel (renders when `v-if="isMobile"`)
  - Backdrop: `fixed top-14.5 left-0 right-0 bottom-0 bg-black/50 z-[200] transition-opacity duration-300`
    - Always in DOM when `isMobile` is true (no `v-if` or `v-show` — `v-show` sets `display: none` which breaks CSS opacity transitions)
    - Toggle visibility via conditional class: `:class="drawerOpen ? 'opacity-100' : 'opacity-0 pointer-events-none'"`
  - Panel: `fixed top-14.5 left-0 bottom-0 w-72 bg-base-200 z-[210] transform transition-transform duration-300`
    - Always in DOM when `isMobile` is true (use CSS translate to show/hide)
    - Translate classes: `translate-x-0` when open, `-translate-x-full` when closed
  - Close button inside the panel header
  - Close on backdrop click

- [x] Move secondary controls into drawer panel
  - ModelSelect
  - EvalProgress
  - AutoIllustration
  - Export button
  - HighlightNav (with `ref="highlightNav"` on the component instance)
  - ThemeToggle
  - Stack controls vertically with appropriate spacing
  - Preserve all `data-help-target` attributes
  - Preserve all `v-model` bindings and event handlers

## State Management

- [x] Add `drawerOpen` ref to control drawer visibility
  - `const drawerOpen = ref(false)`
  - Toggled by burger button
  - Set to `false` in `handleFileUpload` after the `fullReset()` call and before the `checkScorer` call
  - Set to `false` in the `@request-model-download` handler in `index.vue` before calling `checkScorer` (ModelSelect inside the drawer also triggers this path)
  - Set to `false` on backdrop click

- [x] Hide HelpOverlay on mobile
  - Wrap the entire bottom-bar help section (both the tooltip `div` at lines 73-84 and the `v-else` span at lines 86-97) in a `v-if="!isMobile"` condition
  - The help button has two render branches that cannot be cleanly disabled with a single condition; hiding the entire section is the correct approach
  - HelpOverlay is desktop-only; on mobile the help button is not rendered to avoid spotlight positioning issues with drawer-resident controls

## Cleanup

- [x] Update hero min-height in `main.css`
  - Remove the `@media` block entirely
  - Set a single rule: `.hero { min-height: calc(100vh - 58px); }` (no breakpoint needed)
  - Both mobile and desktop top bars are now 58px (`h-14.5`)
  - Verify hero fills viewport correctly on mobile and desktop

- [ ] Verify HelpOverlay compatibility
  - Confirm `data-help-target` elements exist exactly once in the DOM at any screen size
  - On desktop: help overlay works as before, targets found inline in the top bar
  - On mobile: help button is disabled, no overlay launched
  - No changes to `useHelpSteps` expected

- [ ] Verify `ref="highlightNav"` works in both branches
  - Confirm the ref is present on `HighlightNav` in both the desktop top bar and the mobile drawer
  - Since `v-if` ensures only one branch is mounted, the same Vue template ref name resolves correctly

- [x] Make BackToTop button smaller on mobile (`BackToTop.vue`)
  - On mobile: add `btn-sm btn-circle` classes; on desktop keep current unsized default
  - Use responsive Tailwind: replace `btn btn-primary` with `btn btn-primary btn-sm btn-circle lg:btn-md lg:btn-square`
  - Or simpler: use `max-lg:btn-sm max-lg:btn-circle` alongside existing classes

- [x] Make PdfViewer toolbar more compact on mobile (`PdfViewer.vue`)
  - All changes scoped to mobile only (below `lg`)
  - Buttons: `btn-sm` → `max-lg:btn-xs`
  - Inputs: `input-sm text-base w-12` → `max-lg:input-xs max-lg:text-xs max-lg:w-8`; scale input `w-16` → `max-lg:w-10`
  - Page count text and separators: add `max-lg:text-xs`
  - "Render text (slow)" label: hide text on mobile, keep checkbox — `<span class="max-lg:hidden">Render text (slow)</span>`
  - Checkbox: `checkbox-sm` → `max-lg:checkbox-xs`
  - Overall bar: reduce padding — `space-x-2` → `max-lg:space-x-1`, `pe-2` → `max-lg:pe-1`
  - Icon sizes: `size="16"` → keep or reduce to `size="12"` on mobile if icons feel large

- [ ] Test HighlightNav inside drawer
  - Verify dropdown opens correctly within the drawer panel
  - Check that the segment table scrolls properly in the drawer container
  - If overflow issues remain, constrain width or adjust dropdown positioning

- [ ] Run lint and type checks
  - `pnpm lint`
  - Verify no TypeScript errors
