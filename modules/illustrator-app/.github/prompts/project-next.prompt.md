---
name: project-next
description: Implement the next atomic task from a project
argument-hint: project-folder
tools:
  - 'read/readFile'
  - 'edit/editFiles'
  - 'edit/editNotebook'
  - 'edit/createFile'
  - 'edit/createDirectory'
  - 'execute/runInTerminal'
  - 'search/textSearch'
  - 'search/fileSearch'
  - 'todo'
  - 'agent/runSubagent'
---

## Your task

1. Read the project files:
   - `./projects/$PROJECT_FOLDER/planning.md` - Requirements, context, and technical details
   - `./projects/$PROJECT_FOLDER/tasks.md` - Implementation checklist

2. Find the next uncompleted tasks (unmarked `- [ ]`) in tasks.md

3. If all tasks are completed:
   - Output a summary of the project implementation
   - Include key changes made and any notes for the user
   - Stop here

4. If there are uncompleted tasks, Use the `project-implementor` subagent to implement the next atomic chunk of tasks from a project:
   - Identify the next **atomic chunk** of related tasks that should be implemented together (e.g., adding multiple columns to a table, creating a DTO with its fixtures, adding related functions). An atomic chunk is a set of tasks that form a single logical unit of work and should be committed together
   - Implement the chunk following codebase patterns
   - Mark all completed tasks as done (`- [x]`) in tasks.md
   - Run related static checks using `format-lint` skill
   - Use the `commit` skill to create a commit for the entire atomic chunk

5. Launch 3 `code-reviewer` subagents (in parallel if possible), each with a specific focus. Pass the project folder path so reviewers check against project specs:

    - **Simplicity Reviewer**: Focus exclusively on simplicity, unnecessary abstractions, and code elegance. Question every line - can it be removed or simplified?"
    - **Correctness Reviewer**: Focus exclusively on bugs, logic errors, edge cases, null/undefined handling, and functional correctness. Will this code work correctly in all scenarios?"
    - **Conventions Reviewer**: Focus exclusively on project conventions (check CLAUDE.md), established patterns, naming, file organization, and proper use of existing abstractions. Does this follow codebase standards?"

    **IMPORTANT**: Reviewers must ONLY review the scope of THIS atomic commit. Do NOT flag issues in other parts of the codebase that will be addressed in later tasks. Check tasks.md to understand what's in scope for this commit vs what's planned for future commits.

6. If any reviewer requests changes:
    - Use the `project-implementor` subagent to address the feedback by making the necessary fixes
    - Run related static checks using `format-lint` skill
    - Amend the commit with fixes
    - Re-run the specific reviewer subagent that requested changes until approved

7. Report what was implemented, the review verdict, and what's next
