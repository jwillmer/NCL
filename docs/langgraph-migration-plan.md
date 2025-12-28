# Migration Plan: PydanticAI → LangGraph

## Overview

Migrate the NCL agent from PydanticAI (AG-UI) to LangGraph to enable **granular UI updates** during tool execution. Currently, state updates only emit on tool completion. LangGraph allows `copilotkit_emit_state()` calls at any point during execution.

## Why LangGraph?

- **Simplest migration** for single-agent, single-tool architecture
- Most extensive CopilotKit documentation
- Same Python ecosystem (LangChain-based)
- Minimal frontend changes required

---

## High-Level Changes

### 1. Python Backend (`src/ncl/api/`)

#### Dependencies
```toml
# Remove
pydantic-ai-slim[openai,ag-ui]

# Add
langgraph
copilotkit
langchain-openai
```

#### Agent Structure (conceptual)

**Current (PydanticAI):**
```python
agent = Agent("openai:gpt-4o", deps_type=StateDeps[RAGState])

@agent.tool
async def query_email_documents(...) -> ToolReturn:
    state.is_searching = True
    # ... do work ...
    return ToolReturn(metadata=[StateSnapshotEvent(...)])  # Only emits here
```

**Target (LangGraph):**
```python
from copilotkit import CopilotKitState
from copilotkit.langgraph import copilotkit_emit_state

class AgentState(CopilotKitState):
    is_searching: bool = False
    search_progress: str = ""
    # ...

async def search_node(state: AgentState, config: RunnableConfig):
    state["is_searching"] = True
    state["search_progress"] = "Searching vectors..."
    await copilotkit_emit_state(config, state)  # Instant UI update!

    # Vector search...
    state["search_progress"] = "Reranking results..."
    await copilotkit_emit_state(config, state)  # Another update!

    # Reranking...
    state["search_progress"] = "Generating answer..."
    await copilotkit_emit_state(config, state)  # Another update!

    # LLM call...
    return {"messages": [...], "is_searching": False}
```

#### FastAPI Integration

**Current:** `agent.to_ag_ui()` mounted at `/copilotkit`

**Target:** Use `copilotkit_router` or `add_langgraph_fastapi_endpoint`:
```python
from copilotkit.fastapi import copilotkit_router
# or
from ag_ui_langgraph import add_langgraph_fastapi_endpoint
```

### 2. Next.js Runtime (`web/src/app/api/copilotkit/route.ts`)

**Current:** `HttpAgent` (AG-UI protocol)

**Target:** `LangGraphHttpAgent` (LangGraph protocol):
```typescript
import { LangGraphHttpAgent } from "@copilotkit/runtime";

const runtime = new CopilotRuntime({
  agents: {
    default: new LangGraphHttpAgent({
      url: AGENT_URL,
    }),
  },
});
```

### 3. Frontend (`web/src/`)

**Minimal changes** - add state rendering hook:

```tsx
import { useCoAgentStateRender } from "@copilotkit/react-core";

useCoAgentStateRender<RAGState>({
  name: "default",
  render: ({ state, status }) => {
    if (state.search_progress) {
      return <SearchProgress message={state.search_progress} />;
    }
    return null;
  },
});
```

---

## Files to Modify

| File | Change |
|------|--------|
| `pyproject.toml` | Swap dependencies |
| `src/ncl/api/agent.py` | Rewrite as LangGraph graph |
| `src/ncl/api/main.py` | Update router mounting |
| `web/src/app/api/copilotkit/route.ts` | Use `LangGraphHttpAgent` |
| `web/src/types/rag.ts` | Add `search_progress` field |
| `web/src/components/ChatContainer.tsx` | Add `useCoAgentStateRender` |
| `README.md` | Update architecture & dependencies |

---

## Key CopilotKit LangGraph APIs

Reference: Context7 `/copilotkit/copilotkit`

| Function | Purpose |
|----------|---------|
| `CopilotKitState` | Base state class (extends MessagesState) |
| `copilotkit_emit_state(config, state)` | Emit state update to frontend immediately |
| `copilotkit_emit_message(config, msg)` | Emit intermediate message |
| `copilotkit_customize_config(config, ...)` | Configure streaming behavior |
| `copilotkit_router(graph)` | FastAPI router for LangGraph |

---

## Migration Steps (when ready)

1. **Add LangGraph dependencies** to `pyproject.toml`
2. **Create LangGraph graph** with nodes for chat and RAG search
3. **Update state class** to extend `CopilotKitState`
4. **Add `copilotkit_emit_state()` calls** at key progress points
5. **Update FastAPI mounting** to use LangGraph integration
6. **Update Next.js runtime** to use `LangGraphHttpAgent`
7. **Add frontend state renderer** for progress display
8. **Test streaming updates** end-to-end

---

## Security Considerations

- **Auth unchanged**: Both Next.js gateway JWT validation and Python backend `AuthMiddleware` remain in place (defense-in-depth)
- **State exposure**: Only expose non-sensitive fields to frontend (`is_searching`, `search_progress`, `error_message`) - avoid leaking internal data like raw embeddings or full document content in state
- **Input validation**: Keep existing question sanitization (truncate to 2000 chars)

---

## Documentation

- **Update README.md**: Document new LangGraph dependency and streaming capability
- **Update architecture section**: Note the switch from PydanticAI/AG-UI to LangGraph protocol

---

## Notes

- Keep existing RAG query engine (`RAGQueryEngine`) - only agent wrapper changes
- Authentication middleware remains unchanged
- System prompt can be reused in LangGraph system message
- Consider adding progress stages: "Searching" → "Reranking" → "Generating"
