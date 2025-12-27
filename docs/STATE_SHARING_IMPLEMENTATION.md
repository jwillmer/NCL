# State Sharing Implementation Plan

## Overview

This document describes how to implement bidirectional state synchronization between the React frontend and Python Pydantic AI agent using the AG-UI protocol.

## Architecture

```
React Frontend (useCoAgent) <--> Node.js Runtime <--> Python Agent (StateDeps)
         |                            |                       |
    state object              forwards AG-UI           state mutations
    auto-updates              STATE_SNAPSHOT/          via ctx.deps.state
                              STATE_DELTA events
```

### How State Sync Works

1. Frontend initializes with `useCoAgent({ initialState })`
2. Agent tool mutates `ctx.deps.state` during execution
3. Tool returns `StateSnapshotEvent` to trigger sync
4. AG-UI protocol sends STATE_SNAPSHOT to frontend via runtime
5. `useCoAgent` hook receives update, React state updates automatically
6. Components re-render with new state

---

## Requirements

### Python Dependencies

Already installed via `pyproject.toml`:
- `pydantic-ai-slim[openai,ag-ui]>=0.2.0`

Additional imports needed:
- `from pydantic_ai.ui import StateDeps`
- `from ag_ui.core import EventType, StateSnapshotEvent`

### Frontend Dependencies

Already installed via `package.json`:
- `@copilotkit/react-core: ^1.50.1`
- `@copilotkit/react-ui: ^1.50.1`

Additional imports needed:
- `import { useCoAgent } from "@copilotkit/react-core"`

---

## Files to Create/Modify

### 1. Backend: `src/ncl/api/state.py` (CREATE)

Define shared state models using Pydantic:

```python
"""Shared state models for frontend/backend synchronization."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class SourceReference(BaseModel):
    """A source document chunk returned from RAG search."""

    file_path: str = Field(description="Path to the source file")
    document_type: str = Field(description="Type: email, pdf, image, etc.")
    email_subject: Optional[str] = Field(default=None, description="Email subject if applicable")
    email_initiator: Optional[str] = Field(default=None, description="Original email sender")
    email_participants: Optional[List[str]] = Field(default=None, description="All email participants")
    email_date: Optional[str] = Field(default=None, description="Email date as ISO string")
    chunk_content: str = Field(description="The text content of this chunk")
    similarity_score: float = Field(description="Vector similarity score (0-1)")
    rerank_score: Optional[float] = Field(default=None, description="Reranker score if applied")
    heading_path: Optional[str] = Field(default=None, description="Section path within document")
    root_file_path: Optional[str] = Field(default=None, description="Parent email path for attachments")


class RAGState(BaseModel):
    """Shared state between frontend and backend for RAG operations.

    This state enables the frontend to:
    - Display retrieved sources with metadata
    - Show loading/error states
    - React to search progress in real-time
    """

    sources: List[SourceReference] = Field(
        default_factory=list,
        description="Sources retrieved from the last search"
    )
    current_query: Optional[str] = Field(
        default=None,
        description="The current/last search query"
    )
    is_searching: bool = Field(
        default=False,
        description="Whether a search is in progress"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if search failed"
    )
```

### 2. Backend: `src/ncl/api/agent.py` (MODIFY)

Update the agent to use StateDeps:

```python
"""Pydantic AI Agent for NCL Email RAG with shared state."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from pydantic_ai import Agent, RunContext
from pydantic_ai.ui import StateDeps
from ag_ui.core import EventType, StateSnapshotEvent

from ..config import get_settings
from ..rag.query_engine import RAGQueryEngine
from .state import RAGState, SourceReference

logger = logging.getLogger(__name__)


def _load_system_prompt() -> str:
    """Load system prompt from file."""
    prompt_path = Path(__file__).parent / "prompts" / "system.md"
    return prompt_path.read_text(encoding="utf-8")


def _create_state_snapshot(ctx: RunContext[StateDeps[RAGState]]) -> StateSnapshotEvent:
    """Helper to create state snapshot event for AG-UI sync."""
    return StateSnapshotEvent(
        type=EventType.STATE_SNAPSHOT,
        snapshot=ctx.deps.state.model_dump()
    )


# Create the Pydantic AI agent with StateDeps for state management
agent = Agent(
    "openai:gpt-4o",
    deps_type=StateDeps[RAGState],
    instructions=_load_system_prompt(),
)


@agent.tool_plain
async def query_email_documents(
    ctx: RunContext[StateDeps[RAGState]],
    question: str
) -> StateSnapshotEvent:
    """Search and answer questions about email documents and attachments.

    Use this tool to find information from emails, PDFs, images, and other
    attachments in the NCL archive. Always use this tool when the user asks
    about their emails or documents.

    Args:
        ctx: Run context with shared state
        question: The question to ask about the email documents

    Returns:
        StateSnapshotEvent with updated state for frontend sync
    """
    # Sanitize input
    question = question.strip()[:2000] if question else ""
    if not question:
        ctx.deps.state.error_message = "Please provide a valid question."
        return _create_state_snapshot(ctx)

    # Update state - mark as searching
    ctx.deps.state.current_query = question
    ctx.deps.state.is_searching = True
    ctx.deps.state.error_message = None
    ctx.deps.state.sources = []  # Clear previous sources

    engine: Optional[RAGQueryEngine] = None
    try:
        engine = RAGQueryEngine()
        settings = get_settings()

        response = await engine.query(
            question=question,
            top_k=settings.rerank_top_n if settings.rerank_enabled else 10,
            use_rerank=settings.rerank_enabled,
        )

        # Update state with results
        ctx.deps.state.sources = [
            SourceReference(
                file_path=s.file_path,
                document_type=s.document_type,
                email_subject=s.email_subject,
                email_initiator=s.email_initiator,
                email_participants=s.email_participants,
                email_date=s.email_date,
                chunk_content=s.chunk_content,
                similarity_score=s.similarity_score,
                rerank_score=s.rerank_score,
                heading_path=" > ".join(s.heading_path) if s.heading_path else None,
                root_file_path=s.root_file_path,
            )
            for s in response.sources
        ]
        ctx.deps.state.is_searching = False

    except Exception as e:
        logger.error("RAG query failed: %s", str(e), exc_info=True)
        ctx.deps.state.error_message = "I encountered an error while searching. Please try again."
        ctx.deps.state.is_searching = False
    finally:
        if engine:
            await engine.close()

    # Return state snapshot to sync with frontend
    return _create_state_snapshot(ctx)


# Expose as AG-UI ASGI app with initial state
agent_app = agent.to_ag_ui(deps=StateDeps(RAGState()))
```

### 3. Frontend: `frontend/src/lib/types.ts` (CREATE)

TypeScript types that mirror the Python Pydantic models:

```typescript
/**
 * Shared state types for frontend/backend synchronization.
 * These types MUST match the Pydantic models in src/ncl/api/state.py
 */

export type SourceReference = {
  file_path: string;
  document_type: string;
  email_subject?: string;
  email_initiator?: string;
  email_participants?: string[];
  email_date?: string;
  chunk_content: string;
  similarity_score: number;
  rerank_score?: number;
  heading_path?: string;
  root_file_path?: string;
};

export type RAGState = {
  sources: SourceReference[];
  current_query?: string;
  is_searching: boolean;
  error_message?: string;
};
```

### 4. Frontend: `frontend/src/components/ChatContainer.tsx` (MODIFY)

Add useCoAgent hook for state synchronization:

```typescript
/**
 * Chat container component with CopilotKit integration and shared state.
 */

import { useCoAgent } from "@copilotkit/react-core";
import { CopilotChat } from "@copilotkit/react-ui";
import "@copilotkit/react-ui/styles.css";
import { RAGState } from "@/lib/types";
import { SourceList } from "./SourceList";

export function ChatContainer() {
  // Shared state with Python agent - bidirectional sync
  const { state } = useCoAgent<RAGState>({
    name: "default",  // Must match agent name in runtime
    initialState: {
      sources: [],
      current_query: undefined,
      is_searching: false,
      error_message: undefined,
    },
  });

  return (
    <div className="flex-1 flex flex-col h-[calc(100vh-8rem)]">
      {/* Search progress indicator */}
      {state.is_searching && (
        <div className="p-3 bg-blue-50 border-b border-blue-200 text-blue-700 text-sm">
          <span className="animate-pulse">Searching:</span> {state.current_query}...
        </div>
      )}

      {/* Error display */}
      {state.error_message && (
        <div className="p-3 bg-red-50 border-b border-red-200 text-red-700 text-sm">
          {state.error_message}
        </div>
      )}

      <CopilotChat
        labels={{
          title: "NCL Email Assistant",
          initial: "Hello! I can help you search through your email archive. Ask me anything about your emails and attachments.",
          placeholder: "Ask about your emails...",
        }}
        className="flex-1 [&_.copilotKitChat]:h-full [&_.copilotKitMessages]:max-h-[calc(100vh-16rem)]"
      />

      {/* Sources panel - shows when sources are found */}
      {state.sources.length > 0 && (
        <SourceList sources={state.sources} />
      )}
    </div>
  );
}
```

### 5. Frontend: `frontend/src/components/SourceList.tsx` (CREATE)

Component to display sources from shared state:

```typescript
/**
 * SourceList component - displays RAG sources from shared state.
 */

import { SourceReference } from "@/lib/types";

interface SourceListProps {
  sources: SourceReference[];
}

export function SourceList({ sources }: SourceListProps) {
  if (sources.length === 0) return null;

  return (
    <div className="border-t border-gray-200 bg-white p-4 max-h-48 overflow-y-auto">
      <h3 className="text-sm font-semibold text-gray-700 mb-2">
        Sources ({sources.length})
      </h3>
      <div className="space-y-2">
        {sources.map((source, idx) => (
          <div
            key={`${source.file_path}-${idx}`}
            className="text-xs p-2 bg-gray-50 rounded border border-gray-100"
          >
            <div className="font-medium text-gray-800">
              {source.email_subject || source.file_path}
            </div>
            {source.email_initiator && (
              <div className="text-gray-500">
                From: {source.email_initiator}
                {source.email_date && ` - ${source.email_date}`}
              </div>
            )}
            <div className="text-gray-500 truncate mt-1">
              {source.chunk_content.slice(0, 150)}...
            </div>
            <div className="text-gray-400 mt-1">
              Score: {source.similarity_score.toFixed(3)}
              {source.rerank_score && ` (rerank: ${source.rerank_score.toFixed(3)})`}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
```

---

## Context7 Resources

When implementing, use Context7 to fetch the latest documentation:

### For Pydantic AI StateDeps

```
Library ID: /websites/ai_pydantic_dev
Topics to search:
- "StateDeps state management"
- "AG-UI StateSnapshotEvent"
- "RunContext deps"
- "agent.to_ag_ui"
```

Key patterns from docs:
```python
from pydantic_ai.ui import StateDeps
from ag_ui.core import EventType, StateSnapshotEvent

agent = Agent("openai:gpt-4o", deps_type=StateDeps[YourState])

@agent.tool_plain
async def my_tool(ctx: RunContext[StateDeps[YourState]], ...) -> StateSnapshotEvent:
    ctx.deps.state.field = value  # Mutate state
    return StateSnapshotEvent(
        type=EventType.STATE_SNAPSHOT,
        snapshot=ctx.deps.state.model_dump()
    )

app = agent.to_ag_ui(deps=StateDeps(YourState()))
```

### For CopilotKit useCoAgent

```
Library ID: /copilotkit/copilotkit
Topics to search:
- "useCoAgent shared state"
- "useCoAgentStateRender"
- "bidirectional state sync"
```

Key patterns from docs:
```typescript
import { useCoAgent } from "@copilotkit/react-core";

const { state, setState, run } = useCoAgent<YourState>({
  name: "agent_name",  // Must match agent registered in runtime
  initialState: { ... }
});

// state updates automatically when agent sends STATE_SNAPSHOT
// setState can be used to update state from frontend
```

### For Full Examples

```
Library ID: /copilotkit/pydantic-ai-todos
Topics to search:
- "agent tools"
- "state snapshot"
- "TypeScript types"
```

This is a complete working example of Pydantic AI + CopilotKit with shared state.

---

## Implementation Checklist

### Backend (Python)

- [ ] Create `src/ncl/api/state.py` with `RAGState` and `SourceReference` models
- [ ] Update `src/ncl/api/agent.py`:
  - [ ] Import `StateDeps` from `pydantic_ai.ui`
  - [ ] Import `EventType, StateSnapshotEvent` from `ag_ui.core`
  - [ ] Import state models from `.state`
  - [ ] Add `deps_type=StateDeps[RAGState]` to Agent constructor
  - [ ] Add `ctx: RunContext[StateDeps[RAGState]]` parameter to tool
  - [ ] Update state during tool execution
  - [ ] Return `StateSnapshotEvent` from tool
  - [ ] Update `agent.to_ag_ui(deps=StateDeps(RAGState()))`

### Frontend (TypeScript/React)

- [ ] Create `frontend/src/lib/types.ts` with TypeScript types
- [ ] Update `frontend/src/components/ChatContainer.tsx`:
  - [ ] Import `useCoAgent` from `@copilotkit/react-core`
  - [ ] Import `RAGState` from `@/lib/types`
  - [ ] Add `useCoAgent<RAGState>` hook with initial state
  - [ ] Add search progress indicator UI
  - [ ] Add error message display
  - [ ] Add SourceList component
- [ ] Create `frontend/src/components/SourceList.tsx`

### Testing

- [ ] Start Python agent: `uv run uvicorn ncl.api.main:app --port 8000`
- [ ] Start Node.js runtime: `cd runtime && npm run dev`
- [ ] Start frontend: `cd frontend && npm run dev`
- [ ] Send a query in the chat
- [ ] Verify "Searching..." indicator appears
- [ ] Verify sources panel populates after response
- [ ] Verify error message displays on failure

---

## Troubleshooting

### State not updating in frontend

1. Check that agent name matches: `useCoAgent({ name: "default" })` must match the agent registered in runtime
2. Verify tool returns `StateSnapshotEvent`, not a dict
3. Check browser console for AG-UI events

### "Agent 'default' not found" error

1. Ensure Node.js runtime is running on port 3001
2. Verify runtime registers agent: `agents: { default: new HttpAgent({ url: AGENT_URL }) }`
3. Check AGENT_URL points to Python agent: `http://localhost:8000/copilotkit`

### State is stale or doesn't sync

1. Check that `agent.to_ag_ui(deps=StateDeps(RAGState()))` is called with initial state
2. Verify STATE_SNAPSHOT events are being sent (check network tab)
3. Ensure frontend `initialState` matches Python model structure

---

## Notes

- **Type Safety**: Keep Python Pydantic models and TypeScript types in sync manually. Consider using a code generator in the future.
- **State Immutability**: Pydantic models are mutable by default. The state is mutated in-place during tool execution.
- **Performance**: STATE_SNAPSHOT sends the entire state. For large states, consider using STATE_DELTA (JSON Patch) for incremental updates.
- **Auth Disabled**: Auth is currently disabled for testing. Re-enable before production.
