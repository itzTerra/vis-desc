---
allowed-tools: Task, Bash
description: Review staged changes with multiple focused reviewers
---

## Your task

Review the current staged changes (`git diff --staged`) by launching 4 specialized reviewer agents (in parallel if parallel subagents are enabled).

### Phase 1: Check for Changes

1. Run `git diff --staged --stat` to see what files are staged
2. If no staged changes, inform the user and stop

### Phase 2: Parallel Review

Launch 4 `code-reviewer` subagents (in parallel if possible), each with a specific focus:
- **Simplicity Reviewer**: "Review the staged changes (`git diff --staged`). Focus exclusively on simplicity, unnecessary abstractions, and code elegance. Question every line - can it be removed or simplified? Be ruthlessly critical."
- **Correctness Reviewer**: "Review the staged changes (`git diff --staged`). Focus exclusively on bugs, logic errors, edge cases, null/undefined handling, and functional correctness. Will this code work correctly in all scenarios? Be ruthlessly critical."
- **Conventions Reviewer**: "Review the staged changes (`git diff --staged`). Focus exclusively on project conventions (check CLAUDE.md), established patterns, naming, file organization, and proper use of existing abstractions. Does this follow codebase standards? Be ruthlessly critical."
- **Test Reviewer**: "Review the staged changes (`git diff --staged`). Focus exclusively on test coverage. Domain work (queries, mutations, serializers) must have unit tests. Route changes should have Playwright E2E tests. Are tests present, correct, and covering edge cases? Be ruthlessly critical."

### Phase 3: Consolidate Feedback

1. Wait for all reviewers to complete
2. Consolidate findings into a single report:
   - **BLOCKING**: Issues with confidence 90+ that must be fixed
   - **NEEDS CHANGES**: Issues with confidence 80+ that should be fixed
   - **PASS**: If no high-confidence issues found
3. If any issues found, provide a prioritized list of fixes
