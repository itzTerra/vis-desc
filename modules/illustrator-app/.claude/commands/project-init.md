---
allowed-tools: Task, Read, Edit, Glob, Grep, AskUserQuestion
argument-hint: [feature description]
description: Create a project from a text feature description
---

## Your task

Create a comprehensive project specification from a feature description by exploring the codebase and gathering context before clarifying requirements.

### Phase 1: Parse the Feature Description

1. Parse the feature description provided in $ARGUMENTS
2. Extract key information: feature name, goals, key requirements, context

### Phase 2: Codebase Exploration

Launch 3 `code-explorer` subagents (in parallel if possible) to gather context:
- **Similar Features Explorer**: "Find features similar to [feature] and trace their implementation. Focus on: how they're structured, what patterns they use, key abstractions. Return 5-10 key files."
- **Domain Explorer**: "Explore the database schema and domain layer relevant to [feature]. Focus on: table structures, existing queries/mutations, DTOs, algorithms. Return 5-10 key files."
- **UI/Routes Explorer**: "Explore UI components and routes relevant to [feature]. Focus on: component patterns, route structure, client/server boundaries, existing similar UX flows. Return 5-10 key files."

Tailor each prompt to the specific feature content.

### Phase 3: Build Deep Understanding

1. Once agents return, read all key files they identified (deduplicate if needed)
2. Synthesize findings into a coherent understanding of:
   - Existing patterns to follow
   - Database models involved
   - Component/route conventions
   - Similar implementations to reference

### Phase 4: Specification Creation

Launch `project-specifier` subagent with:
- The full ticket details
- Synthesized context from exploration phase
- List of key files discovered

The subagent will:
1. Ask clarifying questions to fill gaps
2. Iterate with the user until requirements are clear
3. Create the project folder at `./projects/YYYY-MM-DD_<project-name>/`

### Output Requirements

**planning.md** (comprehensive yet brief, <500 lines):
- Project classification (type, size)
- Goals and user stories with acceptance criteria
- Functional and non-functional requirements
- Current state analysis
- Key files with descriptions
- Existing patterns to follow (with brief code examples)
- Decisions made during clarification

**tasks.md** (actionable battleplan):
- Ordered implementation checklist
- Grouped by phase: Database → Domain → Components → Routes → E2E Tests → Cleanup
- Each task is atomic (single logical unit)
- Each chunk of atomic tasks should be <200 LoC for easy review
- Tests co-located with functionality they test

### Phase 5: Specification Review

Launch 3 `spec-reviewer` subagents (in parallel if possible) to review the specs:
- **Correctness Reviewer**: "Review the spec at `./projects/[project-name]/` with focus area: CORRECTNESS. Check for contradictions, incorrect assumptions about existing code, missing edge cases, ambiguous requirements, and technical inaccuracies."
- **Architecture Reviewer**: "Review the spec at `./projects/[project-name]/` with focus area: ARCHITECTURE. Check for over-engineering, violations of existing patterns, missing non-functional requirements, and poor separation of concerns."
- **Simplicity Reviewer**: "Review the spec at `./projects/[project-name]/` with focus area: SIMPLICITY. Check for unnecessary features, scope creep, tasks that could be eliminated, and complex solutions when simpler alternatives exist."

If any reviewer returns NEEDS CHANGES or BLOCKING:
1. Summarize the issues found to the user
2. **IMPORTANT**: For non-obvious issues or design decisions that need user input, ask the user using the `AskUserQuestion` tool BEFORE proceeding
3. Once user has clarified their preferences, use the `project-specifier` agent to address the feedback and update the spec files
4. Re-run the failing reviewers until all return APPROVED
5. Present final summary to the user

### Phase 6: User Approval

Ask the user to confirm the specification is ready for implementation using the `AskUserQuestion` tool.
Only proceed to mark the project as ready once the user approves.
