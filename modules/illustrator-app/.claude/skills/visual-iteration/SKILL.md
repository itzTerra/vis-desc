---
name: visual-iteration
description: Iterates on visual design and styling using Playwright MCP. Use when fixing visual issues, improving UI appearance, or making design adjustments based on visual inspection.
---

# Visual Iteration with Playwright MCP

Use the Playwright MCP server to inspect and iterate on visual designs in the browser. This workflow is ideal for fixing contrast issues, adjusting layouts, improving spacing, and making design changes based on actual rendered output.

## Workflow

### 1. Navigate and Inspect

Start by navigating to the page and taking screenshots:

```typescript
// Navigate to the page
await mcp__playwright__browser_navigate({ url: 'http://localhost:3000/page' });

// Take full page screenshot to see overall layout
await mcp__playwright__browser_take_screenshot({
  fullPage: true,
  filename: 'initial-state.png'
});

// Or take viewport screenshot for specific section
await mcp__playwright__browser_take_screenshot({
  filename: 'section-detail.png'
});
```

### 2. Scroll to Specific Sections

Use evaluate to scroll to specific parts of the page:

```typescript
await mcp__playwright__browser_evaluate({
  function: `() => {
    const section = Array.from(document.querySelectorAll('h2'))
      .find(h => h.textContent?.includes('Section Name'));
    section?.scrollIntoView({ behavior: 'instant', block: 'center' });
  }`
});
```

### 3. Inspect DOM and Styles

Check actual rendered styles:

```typescript
await mcp__playwright__browser_evaluate({
  function: `() => {
    const element = document.querySelector('.some-class');
    return {
      computedBg: window.getComputedStyle(element).backgroundColor,
      computedColor: window.getComputedStyle(element).color,
      html: element.outerHTML
    };
  }`
});
```

### 4. Make Code Changes

Edit component files based on visual inspection:

- Fix contrast issues (e.g., `text-neutral-600` → `text-neutral-400`)
- Adjust spacing (e.g., `py-16` → `py-20`, `space-y-8` → `space-y-12`)
- Update colors for better visibility
- Fix alignment and centering
- Adjust sizing and proportions

### 5. Verify Changes

Navigate back to the page and take new screenshots:

```typescript
// Refresh to see changes
await mcp__playwright__browser_navigate({ url: 'http://localhost:3000/page' });

// Take screenshot to compare
await mcp__playwright__browser_take_screenshot({
  fullPage: true,
  filename: 'after-changes.png'
});
```

### 6. Test Interactions

Test interactive elements:

```typescript
// Test button clicks (may need force or evaluate)
await mcp__playwright__browser_evaluate({
  function: `() => {
    const scrollY = window.scrollY;
    document.querySelector('[data-testid="button"]')?.click();
    return { scrollBefore: scrollY, scrollAfter: window.scrollY };
  }`
});
```

## Common Visual Issues and Fixes

### Misaligned Elements

**Issue:** CTAs or content not centered

**Fix:**
```tsx
// Wrap in flex container
<div className="flex justify-center">{cta}</div>
```

### Insufficient Spacing

**Issue:** Elements feel cramped

**Fix:**
- Increase section padding: `py-16` → `py-20`
- Increase internal spacing: `space-y-8` → `space-y-12`
- Increase gaps in grids: `gap-8` → `gap-12`

## Iterative Process

1. **Take initial screenshots** - Document current state
2. **Identify issues** - Note contrast, spacing, alignment problems
3. **Make targeted changes** - Fix one issue at a time
4. **Verify in browser** - Navigate and screenshot to confirm
5. **Test interactions** - Ensure clickability, no scroll jumps
6. **Take final screenshots** - Document improved state

## Tips

- **Always navigate fresh** after code changes to see updates
- **Use fullPage: true** for overall layout, viewport for details
- **Check scroll position** before/after interactions to catch unwanted jumps
- **Test both light and dark sections** for consistent contrast
- **Verify responsive behavior** by resizing browser if needed
- **Save screenshots with descriptive names** (e.g., `pricing-before.png`, `pricing-after.png`)

## Example Session

```typescript
// 1. Initial inspection
await mcp__playwright__browser_navigate({ url: 'http://localhost:3000/en' });
await mcp__playwright__browser_take_screenshot({
  fullPage: true,
  filename: 'landing-initial.png'
});

// 2. Scroll to problematic section
await mcp__playwright__browser_evaluate({
  function: `() => {
    const pricing = document.querySelector('h2:has-text("pricing")');
    pricing?.scrollIntoView({ block: 'center' });
  }`
});
await mcp__playwright__browser_take_screenshot({ filename: 'pricing-section.png' });

// 3. Make code changes (Edit files)

// 4. Verify changes
await mcp__playwright__browser_navigate({ url: 'http://localhost:3000/en' });
await mcp__playwright__browser_take_screenshot({
  fullPage: true,
  filename: 'landing-fixed.png'
});

// 5. Test interaction
await mcp__playwright__browser_evaluate({
  function: `() => {
    const button = document.querySelector('[role="radio"][value="monthly"]');
    button?.parentElement?.click();
    return window.location.href;
  }`
});
```

## Common Playwright MCP Patterns

### Navigate to page
```typescript
await mcp__playwright__browser_navigate({ url: 'http://localhost:3000/path' });
```

### Take screenshot
```typescript
await mcp__playwright__browser_take_screenshot({
  fullPage: true,  // or false for viewport only
  filename: 'descriptive-name.png'
});
```

### Execute JavaScript
```typescript
await mcp__playwright__browser_evaluate({
  function: `() => {
    // Your JS code here
    return { result: 'data' };
  }`
});
```

### Scroll to element
```typescript
await mcp__playwright__browser_evaluate({
  function: `() => {
    document.querySelector('.target')?.scrollIntoView({
      behavior: 'instant',
      block: 'center'
    });
  }`
});
```

### Check computed styles
```typescript
await mcp__playwright__browser_evaluate({
  function: `() => {
    const el = document.querySelector('.element');
    return {
      color: window.getComputedStyle(el).color,
      background: window.getComputedStyle(el).backgroundColor
    };
  }`
});
```

Remember: Visual iteration is about rapid feedback. Navigate, screenshot, edit, repeat until the design looks right.
