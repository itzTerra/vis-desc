---
applyTo: packages/e2e/**/*.ts
---

# Playwright E2E Tests

## IMPORTANT: Test Scope

E2E tests verify **user stories only**. Tests should verify **user-facing behavior**, not implementation details. Each test represents a complete user flow (e.g., "completes session", "skips session", "finishes week").

**What to test:**
- User flows end-to-end (click button → see result)
- High-level assertions: current URL, visibility of container elements (sheets, dialogs, sections)
- Minimal UI changes as response to action (e.g., enter set button shows "180kg × 3" after user enters set)

**What NOT to test:**
- Business logic cases → tested via domain unit tests
- Component display/rendering → verified manually via Storybook stories
- Component permutations/variants → verified manually via Storybook stories
- Computed values or data transformations → tested via query/mutation tests
- Type existence (TypeScript handles this)
- Library internals (trust dependencies)
- Duplicate scenarios (consolidate into one test)
- Implementation details (test behavior, not structure)

## test.each

For testing multiple scenarios with same logic:

```ts
test.each([
  { name: 'empty form', form: {}, error: 'required' },
  { name: 'name too long', form: { name: 'a'.repeat(256) }, error: 'too long' },
])('rejects $name', async ({ form, error }) => {
  expect.assertions(1);
  await expect(validate(form)).rejects.toThrowError(error);
});
```

**Helper factories** for concise test data:

```ts
const entry = (reps: number, weight: number) => ({ reps, weight });

test.each([
  { input: entry(10, 80), want: 106.67 },
  { input: entry(5, 100), want: 112.5 },
])('calculates e1rm', ({ input, want }) => {
  expect(calculateE1rm(input)).toBeCloseTo(want, 2);
});
```

## Edge Cases

Group edge cases in dedicated describe block:

```ts
describe('edge cases', () => {
  test('empty input returns null', () => {
    expect(process([])).toBeNull();
  });

  test('handles zero values', () => {
    expect(calculate(0, 0)).toBe(0);
  });
});
```


## Running Tests

```bash
pnpm verify -- <test files>  # Runs against localhost:3000
pnpm dev                     # Start dev server if not started
```

## Test Structure

**Simple flow**: setup → action → assertion. Nothing more.

```ts
test('activates plan', async () => {
  expect.assertions(1);

  await activatePlan(slug);

  expect(await getPlanState(slug)).toBe('active');
});
```

**Terse test names** - describe what happens, not implementation:

```ts
// Good
'activates plan'
'shows dialog'
'returns empty when no entries'

// Bad
'should call the activatePlan function and update the database'
'test that dialog component renders when button is clicked'
```

**Always use `expect.assertions()`** to catch async issues:

```ts
test('throws on invalid input', async () => {
  expect.assertions(1);
  await expect(mutation()).rejects.toThrowError('invalid');
});
```

**Consolidate redundant tests** - same setup = same test with multiple assertions:

```ts
// Bad - two tests with identical setup
test('returns summary', async () => {
  await seedEntry();
  const res = await getSummary();
  expect(res).toHaveLength(1);
});

test('returns exercise name', async () => {
  await seedEntry();
  const res = await getSummary();
  expect(res[0].name).toBe('Squat');
});

// Good - one test, multiple assertions
test('returns summary with exercise details', async () => {
  expect.assertions(2);
  await seedEntry();
  const res = await getSummary();

  expect(res).toHaveLength(1);
  expect(res[0].name).toBe('Squat');
});
```

**Don't test type system guarantees** - if TypeScript enforces it, don't test it:

```ts
// Bad - TypeScript already enforces these fields exist
test('returns exercise details', async () => {
  expect(res.exercise.name).toBeDefined();
  expect(res.exercise.slug).toBeDefined();
});

// Good - test actual behavior, embed type checks in user-story tests
test('returns summary for completed exercise', async () => {
  expect(res.exercise.name).toBe('High-bar squat');
  expect(res.percentChange).toBe(5);
});
```

## Assertions
Keep assertions minimal. If a user story completes successfully (correct URL, expected container visible), the test passes. Don't assert on specific text content, computed values, or internal component states.

**No redundant pre-checks** - interactions fail naturally if preconditions aren't met:

```ts
// Bad
await expect(button).toBeVisible();
await expect(button).toBeEnabled();
await button.click();

// Good - click fails if not visible/enabled
await button.click();
await expect(result).toBeVisible();
```

**Floating point** - use `toBeCloseTo` for decimal comparisons:

```ts
expect(score).toBeCloseTo(106.67, 2);
```

**Inline snapshots** for short, stable outputs:

```ts
expect(res.map((p) => p.name)).toMatchInlineSnapshot(`
  [
    "Plan A",
    "Plan B",
  ]
`);
```

## Fixtures

Import as `base` and extend. Use single fixture per file.

```ts
import { test as base, expect } from '@/fixtures/subscriber';
import { PlanPage } from './plan-page';

const test = base.extend<{ planPage: PlanPage }>({
  planPage: async ({ page, plan }, use) => {
    const planPage = new PlanPage(page);
    await planPage.goto(plan);
    await use(planPage);
  },
});
```

## Page Objects

Locators as properties, not getter methods. Class name matches route.

```ts
import type { Locator, Page } from '@playwright/test';

export class PlanPage {
  page: Page;
  startPlanButton: Locator;
  nextWeeksDialog: Locator;

  constructor(page: Page) {
    this.page = page;
    this.startPlanButton = page.getByRole('button', { name: 'Start plan' });
    this.nextWeeksDialog = page.getByRole('dialog', { name: 'Next weeks' });
  }

  async goto(slug: string): Promise<void> {
    await this.page.goto(`/en/plans/${slug}`, { waitUntil: 'domcontentloaded' });
  }
}
```

**Methods only for:**
- Navigation (`goto`)
- Complex repeated operations (`enterReps`, `selectOption`)
- Dynamic locators (`getWeekButton(n)`)

## Locators

**Always semantic, never test IDs.** Always include `{ name }` when element has text.

```ts
// ✅ Correct
page.getByRole('button', { name: 'Save' });
page.getByRole('dialog', { name: 'Next weeks' });
page.getByLabel('Email');
page.getByText('Content');

// ❌ Wrong
page.getByRole('button'); // Too generic
page.getByTestId('save-btn'); // Never use test IDs
```

**Priority:** `getByRole` → `getByLabel` → `getByText` → `getByPlaceholder`

## File Organization

[[TODO]]

## Checklist

- ✓ Simple: setup → action → assertion
- ✓ No redundant visibility/enabled checks before interaction
- ✓ `getByRole` has `{ name }` when element has accessible text
- ✓ Semantic locators only (no test IDs)
- ✓ Locators as properties in constructor
- ✓ No top-level `test.describe()` blocks
- ✓ Single fixture per file
- ✓ Terse test names ("shows dialog", "enters set")
