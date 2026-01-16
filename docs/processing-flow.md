# MTSS Processing Flow

This document describes the data flow through the MTSS email RAG pipeline.

## Data Folder Structure

```
data/
├── source/                    # User-provided data (read-only)
│   ├── inbox/                 # Users can organize by folder
│   │   └── project-x/
│   │       └── email1.eml
│   ├── archive/
│   │   └── 2024/
│   │       └── email2.eml
│   └── email3.eml            # Or flat structure
│
└── processed/                 # MTSS-generated data
    ├── attachments/           # Extracted email attachments
    │   └── {email_hash}/
    │       ├── document.pdf
    │       └── image.png
    └── extracted/             # Extracted ZIP contents
        └── {zip_hash}_extracted/
            ├── file1.pdf
            └── nested_folder/
                └── file2.docx
```

## High-Level Architecture

```mermaid
flowchart TB
    subgraph Input["Source Data"]
        EML[EML Files]
    end

    subgraph Parsing["Parsing Layer"]
        EP[EML Parser]
        PP[Preprocessor]
        PR[Parser Registry]
        LP[LlamaParse]
        ZIP[ZIP Extractor]
    end

    subgraph Processing["Processing Layer"]
        IP[Image Processor<br/>OpenAI Vision]
        CH[Document Chunker<br/>LangChain]
        EMB[Embedding Generator]
    end

    subgraph Storage["Storage Layer"]
        DB[(Supabase<br/>pgvector)]
        UF[Unsupported Files Log]
    end

    EML --> EP
    EP --> |Attachments| PP
    PP --> |Route| PR
    PP --> |Images| IP
    PP --> |ZIP| ZIP
    PR --> LP
    ZIP --> PP
    LP --> CH
    IP --> CH
    CH --> EMB
    EMB --> DB
    PP --> |Unsupported| UF
```

## Ingest Pipeline Flow

```mermaid
flowchart TD
    START([MTSS ingest]) --> LOAD_VESSELS[Load vessel registry]
    LOAD_VESSELS --> SCAN[Scan source directory]
    SCAN --> HASH{Hash exists?}

    HASH --> |Completed| SKIP[Skip file]
    HASH --> |New/Failed| PARSE[Parse EML]

    PARSE --> BODY[Extract body text]
    PARSE --> ATTACH[Save attachments]

    BODY --> VESSEL_MATCH[Match vessels in<br/>subject + body]
    VESSEL_MATCH --> CHUNK_BODY[Create body chunks<br/>with vessel_ids]

    ATTACH --> PREPROCESS[Preprocess attachment]

    subgraph Preprocessor["Preprocessor Decision"]
        PREPROCESS --> TYPE{File type?}
        TYPE --> |ZIP| IS_ZIP[is_zip=true]
        TYPE --> |Image| CLASSIFY{Classify image}
        TYPE --> |Document| FIND_PARSER[Find parser]

        CLASSIFY --> |Logo/Banner| SKIP_IMG[should_process=false]
        CLASSIFY --> |Meaningful| DESCRIBE[Get description]

        FIND_PARSER --> |Found| HAS_PARSER[parser_name=llamaparse]
        FIND_PARSER --> |None| NO_PARSER[should_process=false]
    end

    IS_ZIP --> EXTRACT[Extract ZIP]
    EXTRACT --> PREPROCESS

    DESCRIBE --> CREATE_IMG_CHUNK[Create image chunk<br/>inherit vessel_ids]
    HAS_PARSER --> LLAMAPARSE[Parse with LlamaParse]
    LLAMAPARSE --> CHUNK_DOC[Chunk with LangChain<br/>inherit vessel_ids]

    SKIP_IMG --> LOG_SKIP[Log as non-content]
    NO_PARSER --> LOG_UNSUP[Log as unsupported]

    CHUNK_BODY --> EMBED
    CREATE_IMG_CHUNK --> EMBED
    CHUNK_DOC --> EMBED

    EMBED[Generate embeddings] --> STORE[Store in Supabase<br/>with vessel_ids metadata]
    STORE --> COMPLETE[Mark completed]

    SKIP --> END([Next file])
    LOG_SKIP --> END
    LOG_UNSUP --> END
    COMPLETE --> END
```

## Component Responsibilities

### Preprocessor (`DocumentPreprocessor`)

Routes files to appropriate handlers and filters non-content images.

```mermaid
flowchart LR
    subgraph Preprocess["preprocess(file, classify_images)"]
        INPUT[File] --> CHECK_ZIP{ZIP file?}

        CHECK_ZIP --> |Yes| ZIP_RESULT[PreprocessResult<br/>is_zip=true]
        CHECK_ZIP --> |No| CHECK_IMG{Image?}

        CHECK_IMG --> |Yes| CLASSIFY{classify_images?}
        CHECK_IMG --> |No| CHECK_PARSER{Parser exists?}

        CLASSIFY --> |True| VISION[Vision API classify]
        CLASSIFY --> |False| IMG_RESULT[PreprocessResult<br/>is_image=true]

        VISION --> |Logo/Banner| SKIP_RESULT[PreprocessResult<br/>should_process=false]
        VISION --> |Meaningful| DESC_RESULT[PreprocessResult<br/>image_description=...]

        CHECK_PARSER --> |Found| PARSER_RESULT[PreprocessResult<br/>parser_name=...]
        CHECK_PARSER --> |None| UNSUP_RESULT[PreprocessResult<br/>should_process=false]
    end
```

### Parser Registry (`ParserRegistry`)

Simple lookup for file type → parser mapping.

| MIME Type / Extension | Parser |
|-----------------------|--------|
| `application/pdf`, `.pdf` | LlamaParse |
| `application/vnd.openxmlformats-*`, `.docx`, `.pptx`, `.xlsx` | LlamaParse |
| `application/msword`, `.doc`, `.xls`, `.ppt` | LlamaParse |
| `text/csv`, `.csv` | LlamaParse |
| `application/rtf`, `.rtf` | LlamaParse |
| `text/html`, `.html` | LlamaParse |
| `application/epub+zip`, `.epub` | LlamaParse |

### Image Processor (`ImageProcessor`)

Uses OpenAI Vision API for:
1. **Classification**: Detect logos, banners, signatures (skip these)
2. **Description**: Generate text description of meaningful images

```mermaid
flowchart LR
    IMG[Image] --> CLASSIFY[classify_and_describe]
    CLASSIFY --> |API Call| VISION[OpenAI Vision]
    VISION --> RESULT{Classification}
    RESULT --> |logo/banner/signature| SKIP[should_skip=true]
    RESULT --> |meaningful| DESC[description=...]
```

### Document Chunker (`DocumentChunker`)

Uses LangChain text splitters with tiktoken tokenization.

```mermaid
flowchart LR
    TEXT[Markdown Text] --> SPLITTER{Content type?}
    SPLITTER --> |Markdown| MD[MarkdownTextSplitter]
    SPLITTER --> |Plain| REC[RecursiveCharacterTextSplitter]
    MD --> CHUNKS[Chunk objects]
    REC --> CHUNKS
    CHUNKS --> HEADERS[Extract heading_path]
```

Configuration:
- `chunk_size_tokens`: 512 (default)
- `chunk_overlap_tokens`: 50 (default)
- Encoding: `cl100k_base` (matches OpenAI embeddings)

## ZIP Extraction Security

```mermaid
flowchart TD
    ZIP[ZIP File] --> VALID{Valid ZIP?}
    VALID --> |No| ERROR[Raise ValueError]
    VALID --> |Yes| DEPTH{Depth > 3?}

    DEPTH --> |Yes| DEPTH_ERROR[Raise ZipExtractionError]
    DEPTH --> |No| ITERATE[Iterate members]

    ITERATE --> PATH{Path traversal?}
    PATH --> |../ or absolute| SKIP_DANGEROUS[Skip file]
    PATH --> |Safe| HIDDEN{Hidden file?}

    HIDDEN --> |.file or __MACOSX| SKIP_HIDDEN[Skip file]
    HIDDEN --> |No| SIZE{Total size OK?}

    SIZE --> |> 500MB| SIZE_ERROR[Raise ZipExtractionError]
    SIZE --> |OK| COUNT{File count OK?}

    COUNT --> |> 100 files| COUNT_ERROR[Raise ZipExtractionError]
    COUNT --> |OK| EXTRACT[Extract file]

    EXTRACT --> NESTED{Nested ZIP?}
    NESTED --> |Yes| RECURSE[Recursive extract]
    RECURSE --> DEPTH
    NESTED --> |No| ADD[Add to results]
```

Limits (configurable via environment):
- `ZIP_MAX_DEPTH`: 3
- `ZIP_MAX_FILES`: 100
- `ZIP_MAX_TOTAL_SIZE_MB`: 500

## Query Flow

```mermaid
flowchart TD
    Q([User Query]) --> VESSEL{Vessel selected?}
    VESSEL --> |Yes| BUILD_FILTER[Build metadata filter<br/>vessel_ids contains uuid]
    VESSEL --> |No| EMBED

    BUILD_FILTER --> EMBED[Generate embedding]
    EMBED --> SEARCH[Vector search<br/>top_k=20<br/>+ metadata filter]

    SEARCH --> RERANK{Reranking?}
    RERANK --> |Enabled| COHERE[Cohere rerank<br/>top_n=5]
    RERANK --> |Disabled| RESULTS
    COHERE --> RESULTS[Final results]

    RESULTS --> CONTEXT[Build context]
    CONTEXT --> LLM[GPT-4o-mini]
    LLM --> ANSWER[Answer + Sources]
```

### Vessel Filtering

When a user selects a vessel in the UI, the search is filtered to only return chunks from documents tagged with that vessel.

```mermaid
flowchart LR
    subgraph Frontend
        UI[Vessel Dropdown] --> |vessel_id| COPILOT[CopilotKit properties]
    end

    subgraph Agent
        COPILOT --> |selected_vessel_id| SEARCH_NODE[search_node]
        SEARCH_NODE --> |metadata_filter| QUERY[query_engine.search_only]
    end

    subgraph Database
        QUERY --> |vessel_ids contains uuid| MATCH[match_chunks function]
        MATCH --> |chunks.metadata @> filter| RESULTS[Filtered results]
    end
```

**Tagging scope:** Only email subject and body are scanned for vessel names during ingest. Attachments inherit vessel tags from their parent email. See [features.md](features.md#11-vessel-filtering) for details.

## Data Flow Summary

| Stage | Component | Input | Output |
|-------|-----------|-------|--------|
| 1. Load | VesselMatcher | Vessel registry | Name/alias lookup |
| 2. Scan | CLI | Source directory | EML file list |
| 3. Dedupe | ProgressTracker | File hash | Skip or process |
| 4. Parse | EMLParser | EML file | Body + attachments |
| 5. Match | VesselMatcher | Subject + body | vessel_ids list |
| 6. Route | Preprocessor | File + MIME type | PreprocessResult |
| 7. Extract | AttachmentProcessor | ZIP file | Extracted files |
| 8. Classify | ImageProcessor | Image file | Skip or description |
| 9. Parse | LlamaParse | Document | Markdown text |
| 10. Chunk | DocumentChunker | Markdown | Chunk objects |
| 11. Embed | EmbeddingGenerator | Chunks | 1536-dim vectors |
| 12. Store | SupabaseClient | Chunks + vessel_ids | Database records |

## Database Schema

| Table | Purpose |
|-------|---------|
| `documents` | Email/attachment hierarchy with deduplication |
| `chunks` | Text chunks with embeddings (pgvector), vessel_ids in metadata |
| `vessels` | Vessel registry (name, IMO, type, aliases) |
| `conversations` | Chat conversations with vessel_id filter |
| `unsupported_files` | Logged unsupported/skipped files |
| `processing_log` | Progress tracking for resume |
