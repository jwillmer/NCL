---
name: code-health
description: Codebase health audit for the MTSS/NCL ingest + RAG pipeline. Evaluates maintainability, KISS, separation of concerns, security, best practices, library usage, and ingest-pipeline integrity. Discovery only — produces a prioritized report, does not modify code.
disable-model-invocation: true
allowed-tools: Read Grep Glob Bash(git *) Bash(python *) Bash(uv *) Bash(npm *) Bash(npx *) Bash(mtss *)
effort: max
---

# Code Health Audit (MTSS)

Comprehensive codebase health review. Surface areas of concern so they can be prioritized and investigated further. **This is discovery, not fixing** — produce the report, let the user decide what to act on.

## Scope

- Backend: `src/mtss/` (Python ingest + RAG pipeline, CLI, API)
- Frontend: `web/src/` (React/Vite/TS chat UI)
- Tests: `tests/` (pytest)
- Scripts: `scripts/` (one-off utilities)
- Docs: `docs/`

Skip: `.venv/`, `.pytest_cache/`, `__pycache__/`, `node_modules/`, `dist/`, `out/`, `data/`, `.claude/`, `migrations/` (generated SQL), `.mypy_cache/`.

If `$ARGUMENTS` is provided, narrow to that area (e.g. `backend`, `frontend`, `ingest`, `security`, a specific subdirectory).

## Evaluation Criteria

Search systematically. Do not rely on memory. Prefer `Grep` + `Glob` over shell `find`/`grep`.

### 1. Ingest Pipeline Integrity (project-critical)

**This pipeline processes production data that costs time and money to regenerate. Weight these findings heavily.**

- Sanitization consistency: filenames, paths, archive keys must use the same sanitizer across all code paths (historical bugs: brackets, parens, tildes, markdown injection)
- Validation coverage: is every data invariant checked by `mtss validate ingest` (22 checks today) or a regression test? Flag invariants that can only be caught post-incident.
- Round-trip integrity: does what's in `*.jsonl` match what's written to Supabase? Check `insert_*` methods include all model fields.
- Crash safety: writes that need `fsync`, atomic persistence, cleanup-before-upload patterns
- Link/reference preservation: content transforms that strip information (URLs, image refs, citations) when it could be preserved as markdown
- TOCTOU patterns in archive generation and file checks

### 2. Maintainability

- Large files doing too much (>500 lines as a rough threshold; investigate, don't auto-flag)
- Long functions that are hard to follow (>80 lines)
- Deeply nested code (3+ levels)
- Duplicated logic across files (3+ occurrences)
- Dead code: unused imports, unreachable branches, commented-out blocks
- Inconsistent patterns: same thing done differently in different places

### 3. KISS

- Over-engineered abstractions: base classes with one subclass, factories for one type
- Unnecessary indirection: wrappers that just forward, layers that add no logic
- Premature generalization: parameterized code only called one way
- Overly complex expressions that obscure intent

### 4. Separation of Concerns

**Pragmatic rule: don't flag splitting for the sake of splitting. Only flag when mixing concerns makes code harder to test, change, or understand.**

- Business logic in CLI command handlers (`cli/*_cmd.py`) instead of ingest/ or processing/ modules
- Supabase queries scattered outside `storage/repositories/`
- Frontend components mixing data fetching, state, and rendering into monoliths
- Configuration scattered instead of centralized in `config.py`

### 5. Security

- SQL injection risks (string formatting in asyncpg queries instead of `$1`-style params)
- Command injection (unsanitized input in subprocess/shell calls — check `scripts/`)
- Path traversal (user-controlled input in file paths, esp. ZIP extraction and archive lookups)
- Hardcoded secrets (Supabase keys, OpenAI/OpenRouter keys, LlamaParse keys)
- Missing input validation at API boundaries (`api/` endpoints)
- Unsafe deserialization (`pickle`, `eval`, `exec` on external input)
- Prompt injection: LLM inputs should flow through `sanitize_input` in `processing/topics.py`
- Dependency vulnerabilities: `cd web && npm audit --omit=dev 2>&1 | tail -30`. For Python, check `uv.lock` against known CVEs if `pip-audit` is available.

### 6. Best Practices

- Error handling: bare `except:`, swallowed errors, overly broad catches
- Resource leaks: missing context managers for files, DB connections, HTTP clients
- Async issues: blocking calls (`time.sleep`, `requests`, file I/O without `aiofiles`) inside async functions; missing `await`; `asyncio.gather` without `return_exceptions` where appropriate
- Type safety: excessive `Any`, missing return types on public APIs
- Test gaps: critical ingest/validation paths without regression coverage (cross-reference `tests/`)
- Python: mutable default args, global state mutation, circular imports, inline `import` statements inside hot-path functions
- TypeScript: `any` casts, non-null `!` assertions on uncertain values, missing error boundaries

### 7. Library Usage

- Custom implementations of things available in stdlib or existing dependencies (`pyproject.toml`, `web/package.json`)
- Multiple libraries doing the same job (check for overlap)
- Outdated dependencies with significant improvements available
- Vendored code that could be a maintained package

## Tooling Commands

Run these to feed the metrics snapshot. Best-effort — skip silently if a tool is unavailable.

```bash
# Repo size
find src/mtss -name "*.py" | wc -l
find web/src -name "*.ts" -o -name "*.tsx" | wc -l
wc -l $(find src/mtss -name "*.py") 2>/dev/null | tail -1
wc -l $(find web/src \( -name "*.ts" -o -name "*.tsx" \)) 2>/dev/null | tail -1

# Largest files (top 5)
wc -l $(find src/mtss -name "*.py") 2>/dev/null | sort -rn | head -6

# Tests
python -m pytest tests/ --collect-only -q 2>&1 | tail -5

# Frontend build sanity
cd web && npm run build 2>&1 | tail -10

# Dependency audit
cd web && npm audit --omit=dev 2>&1 | tail -20
```

Do NOT run the full ingest pipeline (`mtss ingest`) — it costs money and processes real data. Reading `mtss validate ingest` output from recent runs is fine.

## Output Format

Write the report to `code-health-report.md` at the repo root (overwrite if exists). Also print a summary to the console.

```markdown
# Code Health Report — {YYYY-MM-DD}

## Metrics Snapshot
- Backend: {X} Python files, {Y} total lines
- Frontend: {Z} TS/TSX files, {W} total lines
- Largest files: {top 5 by line count}
- Tests: {pytest count}; Frontend build: {pass/fail}
- Dependency audit: {summary}

## Critical (fix now)
{Security issues, data loss risks, correctness bugs, ingest integrity}

### {Finding title}
**Overview:** {1-2 sentences describing the concern}
**Affected files:** {paths, grouped}
**Impact:** {what breaks if ignored}

## High Priority (fix this iteration)
{Maintainability blockers, significant code smells, risky patterns, validation gaps}

## Medium Priority (plan for next cycle)
{Refactoring opportunities, library replacements, test gaps}

## Low Priority (nice to have)
{Style improvements, minor simplifications, doc updates}

## Positive Observations
{Things done well — patterns to keep, good abstractions, well-tested areas}

## Delta vs Previous Report
{If code-health-report.md existed before this run: what's new / fixed / regressed}
```

## Guidelines

- **Overview, not line-by-line fixes**: describe the concern, list affected files. Exact fixes are a follow-up investigation.
- **Group related findings**: if multiple files share the same issue, one finding with a file list.
- **Weigh severity honestly**: not everything is critical. The tiers must mean something.
- **Compare to prior report**: if `code-health-report.md` exists, note new / fixed / regressed items in the Delta section.
- **Cite evidence**: every Critical/High finding needs at least one file path and a line reference or grep hit.
- **Discovery only**: do NOT modify code, do NOT run ingest, do NOT delete files. Produce the report. The user decides next steps.
