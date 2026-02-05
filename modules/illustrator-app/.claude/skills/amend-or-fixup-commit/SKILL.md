---
name: amend-or-fixup-commit
description: Adds to or makes changes to existing commit. Use when you need to (1) Amend — update the most recent commit (changing the commit message and/or adding staged changes) or (2) Fixup — update an earlier commit than the last one and clean history later using `git rebase -i --autosquash`.
---

# Amend or fixup commit

## Context

- Current branch: !`git branch --show-current`
- Recent commits: !`git log --oneline -10`
- Current git status: !`git status`
- Current git diff (staged and unstaged changes): !`git diff HEAD`

## Instructions

1. Fetch latest: `git fetch`
2. Review changes and history:
	- `git status`
	- `git log --oneline -10`
	- `git diff HEAD`
3. Decide amend vs fixup:
	- If the change belongs to the most recent commit, amend.
	- If the change belongs to an older commit, fixup.
4. Stage relevant files.
5. Amend path:
	- `git commit --amend`
	- If only adding files with no message change: `git commit --amend --no-edit`
6. Fixup path:
	- Identify target commit SHA from the log.
	- `git commit --fixup <commit>`
	- Warn the user clearly that he is supposed to `git rebase -i --autosquash` later.
7. Verify:
	- `git log --oneline -10`
	- `git status`


