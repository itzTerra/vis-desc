---
name: project-review
argument-hint: project-folder
description: Review the last commit against a project's requirements
tools:
  - 'read/readFile'
  - 'edit/editFiles'
  - 'edit/editNotebook'
  - 'edit/createFile'
  - 'edit/createDirectory'
  - 'execute/runInTerminal'
  - 'search/textSearch'
  - 'search/fileSearch'
  - 'agent/runSubagent'
---

## Your task

Review the last atomic commit for the project at `./projects/$ARGUMENTS/`.

1. Read the project files:
    - `./projects/$ARGUMENTS/planning.md` - Requirements, context, and technical details
    - `./projects/$ARGUMENTS/tasks.md` - Implementation checklist

2. Get commit context:
    - Run `git show HEAD --stat` to see what files changed
    - Run `git log --oneline -1` to see the commit message

3. Launch 3 `code-reviewer` subagents (in parallel if possible). Pass the project folder path `./projects/$ARGUMENTS/` so reviewers check against project specs:

    **IMPORTANT**: Reviewers must ONLY review the scope of THIS atomic commit. Do NOT flag issues in other parts of the codebase that will be addressed in later tasks. Check tasks.md to understand what's in scope for this commit vs what's planned for future commits.

    - **Simplicity Reviewer**: Focus exclusively on simplicity, unnecessary abstractions, and code elegance. Question every line - can it be removed or simplified?"
    - **Correctness Reviewer**: Focus exclusively on bugs, logic errors, edge cases, null/undefined handling, and functional correctness. Will this code work correctly in all scenarios?"
    - **Conventions Reviewer**: Focus exclusively on project conventions (check CLAUDE.md), established patterns, naming, file organization, and proper use of existing abstractions. Does this follow codebase standards?"

4. Analyze reviewer feedback:
    - If all reviewers PASS: Report success and stop
    - If any reviewer flags issues with confidence >= 80:
      - Determine if the issue is a **code problem** or a **spec mismatch**
      - Code problem: The implementation is wrong and needs fixing
      - Spec mismatch: The implementation is intentionally different from spec
        (e.g., user requested a change after spec was written)

5. For spec mismatches:
    - Update `tasks.md` and/or `planning.md` to reflect the actual implementation
    - Amend the commit to include the spec updates
    - Report what was updated and why

6. For code problems:
    - Fix the issue directly (small fixes) or run `project-implementor` subagent
    - Run lint and type checks using `format-lint` skill
    - Amend the commit with fixes
    - Re-run the specific reviewer subagent that requested changes until approved

7. Report the final verdict:
    - What was reviewed
    - Issues found and how they were resolved
    - Final status (PASS or what still needs attention)
