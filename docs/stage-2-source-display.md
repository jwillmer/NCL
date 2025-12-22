# Stage 2: Source Display Implementation Plan

## Overview

Stage 2 adds source display functionality to the NCL chat interface, enabling users to view and explore the documents used to generate answers. This builds user confidence by showing the provenance of information.

## Goals

1. Display retrieved sources alongside chat responses
2. Show confidence scores (similarity + rerank) for transparency
3. Allow users to expand/view source content
4. Enable navigation through document hierarchy (email → attachment → section)

---

## Architecture

### State Flow

```
User Query → Backend RAG Search → Sources + Answer
                                        ↓
                    Frontend receives via CopilotKit action result
                                        ↓
                    RAGState updated (sources, answer, is_searching)
                                        ↓
                    UI renders answer + SourceList component
```

### Key Components

| Component | Purpose |
|-----------|---------|
| `SourceList` | Container for displaying all sources |
| `SourceCard` | Individual source with metadata and preview |
| `ConfidenceIndicator` | Visual score badge with tooltip |
| `SourceDetail` | Expandable/modal view of full source content |

---

## Implementation Steps

### Phase 1: Source Components

**1.1 Create `Sources.tsx`**

Single file containing all source-related components (KISS principle):

```tsx
// SourceList - renders array of SourceCard
// SourceCard - compact card with:
//   - Document type icon (email, pdf, image, etc.)
//   - Title (email_subject or file_path)
//   - Sender and date for emails
//   - Truncated content preview (first 150 chars)
//   - ConfidenceIndicator badge
// ConfidenceIndicator - color-coded score badge:
//   - Green: >=80%
//   - Blue: >=60%
//   - Yellow: >=40%
//   - Gray: <40%
//   - Checkmark icon if rerank_score exists
```

**1.2 Source Type Icons**

Map `document_type` to icons:
- `email` → Mail icon
- `pdf` → FileText icon
- `image` → Image icon
- `docx`/`xlsx`/`pptx` → File icon variants
- `zip` → Archive icon

**1.3 Content Preview**

- Truncate `chunk_content` to 150 characters
- Add ellipsis and "Show more" link
- Strip markdown/formatting for preview

### Phase 2: Integration with Chat

**2.1 Modify `ChatContainer.tsx`**

Add sources panel below chat:

```tsx
<div className="flex gap-4">
  <div className="flex-1">
    <CopilotChat ... />
  </div>
  <div className="w-80">
    <SourceList sources={state.sources} />
  </div>
</div>
```

**2.2 Use RAGState from Backend**

Two options for state access:

**Option A: Polling endpoint (simpler)**
```tsx
// Poll /rag-state every 2s while is_searching
useEffect(() => {
  const interval = setInterval(fetchState, 2000);
  return () => clearInterval(interval);
}, []);
```

**Option B: useCoAgent hook (recommended)**
```tsx
import { useCoAgent } from "@copilotkit/react-core";

const { state } = useCoAgent<RAGState>({
  name: "ncl_email_agent",
  initialState: initialRAGState,
});
```

Recommendation: Start with Option A, migrate to Option B when CopilotKit state sync is needed.

**2.3 Loading States**

Show skeleton cards while `is_searching === true`:

```tsx
{state.is_searching ? (
  <SourceListSkeleton count={3} />
) : (
  <SourceList sources={state.sources} />
)}
```

### Phase 3: Source Detail View

**3.1 Expandable Card**

Click source card to expand in-place:
- Show full `chunk_content`
- Show `heading_path` as breadcrumb
- Link to parent email if `root_file_path` exists

**3.2 Modal Alternative**

For complex sources, use a modal:
- Full content with syntax highlighting (for code)
- Document hierarchy visualization
- Copy button for content

### Phase 4: Document Hierarchy

**4.1 Breadcrumb Component**

Display path from root email to chunk:

```
📧 RE: Project Update → 📎 budget.pdf → 📄 Section 2.1
```

Parse from:
- `root_file_path`: Parent email path
- `file_path`: Attachment path
- `heading_path`: Section within document

**4.2 Source Grouping**

Group sources by parent email:

```tsx
const groupedSources = useMemo(() => {
  return sources.reduce((acc, source) => {
    const key = source.root_file_path || source.file_path;
    if (!acc[key]) acc[key] = [];
    acc[key].push(source);
    return acc;
  }, {} as Record<string, SourceReference[]>);
}, [sources]);
```

---

## UI/UX Specifications

### Layout

```
┌─────────────────────────────────────────────────────────┐
│  NCL Email Assistant                              [User]│
├───────────────────────────────────┬─────────────────────┤
│                                   │ Sources (3)         │
│  Chat Messages                    │ ┌─────────────────┐ │
│                                   │ │ 📧 Subject...   │ │
│  [User]: What about the budget?   │ │ From: John      │ │
│                                   │ │ Score: 92%  ✓   │ │
│  [Assistant]: Based on the email  │ └─────────────────┘ │
│  from John on Dec 15...           │ ┌─────────────────┐ │
│                                   │ │ 📎 budget.pdf   │ │
│                                   │ │ Section 2.1     │ │
│                                   │ │ Score: 85%  ✓   │ │
│                                   │ └─────────────────┘ │
│                                   │ ┌─────────────────┐ │
│                                   │ │ 📧 RE: Budget   │ │
│  ┌─────────────────────────────┐  │ │ From: Sarah     │ │
│  │ Ask about your emails...    │  │ │ Score: 71%      │ │
│  └─────────────────────────────┘  │ └─────────────────┘ │
└───────────────────────────────────┴─────────────────────┘
```

### Colors

Use NCL brand colors for confidence:
- High (>=80%): `#003A8F` (NCL Blue)
- Medium (>=60%): `#4F83CC` (Light Blue)
- Low (>=40%): `#6D6E71` (Gray)
- Very Low (<40%): `#D1D3D4` (Light Gray)

### Responsive

- Desktop: Side panel (as shown)
- Tablet: Collapsible panel
- Mobile: Sources below chat, accordion style

---

## Files to Create/Modify

### New Files

| File | Purpose |
|------|---------|
| `frontend/src/components/Sources.tsx` | All source display components |
| `frontend/src/types/rag.ts` | TypeScript types (already created) |

### Modified Files

| File | Changes |
|------|---------|
| `frontend/src/components/ChatContainer.tsx` | Add sources panel |
| `frontend/src/App.tsx` | Add state fetching/useCoAgent |

---

## Testing Checklist

- [ ] Sources display after query completes
- [ ] Confidence scores show correct colors
- [ ] Reranked sources show checkmark
- [ ] Source cards expand on click
- [ ] Content preview truncates correctly
- [ ] Document hierarchy breadcrumb works
- [ ] Loading skeleton shows during search
- [ ] Error state displays properly
- [ ] Responsive layout works on all screen sizes
- [ ] Sources clear on new query

---

## Future Enhancements (Post-Stage 2)

1. **Source Filtering**: Filter by document type, date range, sender
2. **Source Sorting**: Sort by relevance, date, or type
3. **Feedback Loop**: Allow users to mark sources as helpful/unhelpful
4. **Source Search**: Search within retrieved sources
5. **Export**: Download sources as PDF or citation list
