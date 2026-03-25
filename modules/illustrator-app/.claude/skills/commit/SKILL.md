---
name: commit
description: Creates commit(s) based on code changes.
---

# Commit

## Context

- Current branch: !`git branch --show-current`
- Recent commits: !`git log --oneline -10`
- Current git status: !`git status`
- Current git diff (staged and unstaged changes): !`git diff HEAD`

## Commit Message Guidelines

- Use **conventional commit** message format: `<type>(<scope>): <summary>`
- Use types: feat, fix, docs, style, refactor, perf, test, chore, ci.
- Include scope when changing specific module (e.g. api, frontend). Leave the scope out if: (1) the type is `ci`; (2) if the change is global
- Use imperative mood: 'Add feature' not 'Added feature'. Explain what and why, not how.
- Keep subject line under 50 characters.
- If the summary is not enough to encompass the commit scope, add a well-structured body section with bullet points (*) for clarity.

## Instructions

1. Fetch latest: `git fetch`
2. Run `git diff HEAD` to analyze changes — look at each modified file and determine logical groupings
3. **Ask if ambiguous** — If it's not clear how to split changes, ask the user
4. Verify with `git log --oneline origin..HEAD -10`

