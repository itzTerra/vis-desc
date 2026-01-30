---
applyTo: projects/**/*.md
---

# Project Specifications

Projects live in `projects/<date>_<project-name>/` with two files:
- `planning.md` - Requirements, context, key files, patterns, and decisions
- `tasks.md` - Implementation checklist (you tick these off)

## Project Types

| Type            | Description                                  | Examples                                            |
|-----------------|----------------------------------------------|-----------------------------------------------------|
| **feature**     | New user-facing functionality                | Export workout data, social sharing, workout timer  |
| **refactor**    | Restructuring code without changing behavior | Extract domain module, migrate to new API pattern   |
| **improvement** | Enhancing existing functionality             | Faster search, better error messages, accessibility |

## Implementation Order

### Features & Improvements

1. **Domain**: Migrate database, build queries, algorithms, and serializers, write domain tests
2. **Components**: Build necessary UI components and their Storybook stories
3. **Routes**: Build new app routes and adjust existing routes
4. **E2E Tests**: Implement Playwright tests for new/edited app routes

**Migrations**: Batch all schema changes within a project into a single migration file when possible.

### Refactorings

1. **Strategy**: Implement incremental refactoring strategy
2. **Execution**: Incrementally implement the refactoring in atomic pieces

## Project Sizes

| Size   | Files | Time      | Examples                               |
|--------|-------|-----------|----------------------------------------|
| **S**  | 1-2   | <1 hour   | Add field to form, new validation rule |
| **M**  | 3-10  | 1-4 hours | New modal flow, API endpoint with UI   |
| **L**  | 10-30 | 1-2 days  | New domain with queries/mutations/UI   |
| **XL** | 30+   | Multi-day | Major feature, cross-cutting refactor  |

## Folder Structure

```
projects/
  2025-01-15_export-workout-data/
    planning.md    # Requirements, context, and technical details
    tasks.md       # Implementation checklist
```

## planning.md

Combined specification and technical context. Contains everything needed to understand and implement the project.

```md
---
type: feature | refactor | improvement
size: S | M | L | XL
---

# Project Name

## Overview

Brief description of the project and its value.

## Goals

- Primary goal
- Secondary goals

## User Stories

User stories drive E2E test coverage. Every user story should map to one or more Playwright tests.

### As a [role], I want [action] so that [benefit]

**Acceptance criteria:**

- [ ] Criterion 1
- [ ] Criterion 2

## Requirements

### Functional

- Requirement 1
- Requirement 2

### Non-functional

- Performance, security, accessibility considerations

## Scope

### Included

- What's in scope

### Excluded

- What's explicitly out of scope

---

## Current State

Description of how things work today. What exists, what needs to change.

## Key Files

- `path/to/file.ts` - Description of relevance
- `path/to/another.ts` - Description

## Existing Patterns

Relevant code patterns to follow. Include brief examples when helpful.

## Decisions

Document significant implementation decisions made during planning or implementation.

### Decision Title

**Decision**: What was decided.

**Rationale**: Why this approach was chosen over alternatives.

**Implementation**: How it affects the implementation (optional).
```

## tasks.md

Implementation checklist. The implementor ticks these off as they work.

```md
# Tasks

## Domain

- [ ] Database migration
- [ ] Queries with tests
- [ ] Mutations with tests

## Components

- [ ] Component with stories

## Routes

- [ ] Route implementation

## E2E Tests

- [ ] User story 1: test scenarios
- [ ] User story 2: test scenarios

## Cleanup

- [ ] Run types and fmt
- [ ] Verify Storybook
```

Tasks should be:

- **Actionable**: Clear what "done" means
- **Ordered**: Dependencies respected
- **Sized**: Each task ~15-60 min of work
- **Testable**: Can verify completion

**IMPORTANT**: Code and its related tests (query tests, mutation tests, route tests, etc.) must always be in the same commit. Functionality and its test are one atomic unit - never commit code without its test or vice versa.
