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

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              SOURCE DATA                                     │
│                                                                             │
│                             [ EML Files ]                                   │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            PARSING LAYER                                     │
│                                                                             │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────┐                  │
│  │ EML Parser  │───▶│ Preprocessor │───▶│Parser Registry│                  │
│  └─────────────┘    └──────┬───────┘    └───────┬───────┘                  │
│                            │                    │                           │
│                     ┌──────┴──────┐      ┌──────┴──────┐                   │
│                     ▼             ▼      ▼             ▼                   │
│              ┌───────────┐  ┌─────────┐  ┌───────────────┐                 │
│              │ZIP Extract│  │ Images  │  │  LlamaParse   │                 │
│              └───────────┘  └─────────┘  └───────────────┘                 │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PROCESSING LAYER                                    │
│                                                                             │
│  ┌──────────────────┐    ┌──────────────────┐    ┌─────────────────────┐   │
│  │  Image Processor │    │  Document Chunker│    │ Embedding Generator │   │
│  │  (OpenAI Vision) │───▶│   (LangChain)    │───▶│   (OpenAI + retry)  │   │
│  └──────────────────┘    └──────────────────┘    └─────────────────────┘   │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           STORAGE LAYER                                      │
│                                                                             │
│         ┌───────────────────────┐    ┌─────────────────────────┐           │
│         │   Supabase + pgvector │    │   Ingest Events Log     │           │
│         │   (documents, chunks) │    │   (errors, warnings)    │           │
│         └───────────────────────┘    └─────────────────────────┘           │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Ingest Pipeline Flow

```
┌──────────────────┐
│   MTSS ingest    │
│   [--lenient]    │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────┐
│ 1. Load vessel registry              │
└────────┬─────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│ 2. Scan source directory for .eml    │
└────────┬─────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│ 3. Check hash - already processed?   │
├──────────────────────────────────────┤
│ YES (Completed) ──▶ Skip file        │
│ NO  (New/Failed) ──▶ Continue        │
└────────┬─────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│ 4. Parse EML file                    │
│    • Extract body text (multi-charset│
│      decode with fallback)           │
│    • Save attachments to temp folder │
└────────┬─────────────────────────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐  ┌─────────────────────────────┐
│ BODY   │  │ ATTACHMENTS                 │
└───┬────┘  └─────────────┬───────────────┘
    │                     │
    ▼                     ▼
┌────────────────┐  ┌─────────────────────────────────────────┐
│Match vessels in│  │ 5. Preprocess each attachment           │
│subject + body  │  │                                         │
└───┬────────────┘  │    ┌─────────────────────────────────┐  │
    │               │    │ File Type?                      │  │
    ▼               │    ├─────────────────────────────────┤  │
┌────────────────┐  │    │                                 │  │
│Create body     │  │    │ ZIP ──────▶ Extract contents    │  │
│chunks with     │  │    │            (recurse to preproc) │  │
│vessel_ids      │  │    │                                 │  │
└───┬────────────┘  │    │ Image ───▶ Classify via Vision  │  │
    │               │    │            Logo/Banner? Skip    │  │
    │               │    │            Meaningful? Describe │  │
    │               │    │                                 │  │
    │               │    │ Document ▶ Find parser          │  │
    │               │    │            Found? LlamaParse    │  │
    │               │    │            None? Log unsupported│  │
    │               │    └─────────────────────────────────┘  │
    │               └─────────────┬───────────────────────────┘
    │                             │
    │    ┌────────────────────────┤
    │    │                        │
    │    ▼                        ▼
    │ ┌──────────────┐    ┌───────────────────┐
    │ │Image chunk   │    │Document chunks    │
    │ │(description) │    │(LangChain split)  │
    │ │+ vessel_ids  │    │+ vessel_ids       │
    │ └──────┬───────┘    └─────────┬─────────┘
    │        │                      │
    └────────┴──────────┬───────────┘
                        │
                        ▼
         ┌──────────────────────────────┐
         │ 6. Generate embeddings       │
         │    (with retry + backoff)    │
         └──────────────┬───────────────┘
                        │
                        ▼
         ┌──────────────────────────────┐
         │ 7. Store in Supabase         │
         │    • Insert chunks           │
         │    • vessel_ids in metadata  │
         └──────────────┬───────────────┘
                        │
                        ▼
         ┌──────────────────────────────┐
         │ 8. Mark document COMPLETED   │
         └──────────────────────────────┘
```

## Component Responsibilities

### Preprocessor (`DocumentPreprocessor`)

Routes files to appropriate handlers and filters non-content images.

```
                    ┌─────────────────┐
                    │ preprocess(file,│
                    │ classify_images)│
                    └────────┬────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │  ZIP file?     │
                    └───┬────────┬───┘
                        │        │
                       YES       NO
                        │        │
                        ▼        ▼
            ┌───────────────┐  ┌──────────────┐
            │PreprocessResult│  │  Image?      │
            │is_zip=true     │  └──┬───────┬──┘
            └───────────────┘     │       │
                                 YES      NO
                                  │       │
                                  ▼       ▼
                    ┌─────────────────┐  ┌──────────────┐
                    │classify_images? │  │Parser exists?│
                    └──┬──────────┬──┘  └──┬───────┬──┘
                      YES         NO      YES      NO
                       │          │        │       │
                       ▼          ▼        ▼       ▼
              ┌─────────────┐ ┌──────┐ ┌──────┐ ┌──────────┐
              │Vision API   │ │Result│ │Result│ │Result    │
              │classify     │ │image │ │parser│ │should_   │
              └──┬──────┬──┘ │=true │ │=name │ │process=  │
                 │      │    └──────┘ └──────┘ │false     │
           Logo/Banner  Meaningful             └──────────┘
                 │      │
                 ▼      ▼
          ┌──────────┐ ┌────────────┐
          │should_   │ │image_      │
          │process=  │ │description │
          │false     │ │=...        │
          └──────────┘ └────────────┘
```

### Parser Registry (`ParserRegistry`)

Simple lookup for file type to parser mapping.

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

```
┌─────────┐     ┌─────────────────────┐     ┌───────────────┐
│ Image   │────▶│ classify_and_describe│────▶│ OpenAI Vision │
└─────────┘     └─────────────────────┘     └───────┬───────┘
                                                    │
                                           ┌────────┴────────┐
                                           │                 │
                                           ▼                 ▼
                                  ┌─────────────────┐ ┌────────────────┐
                                  │logo/banner/sig  │ │ meaningful     │
                                  │should_skip=true │ │ description=...│
                                  └─────────────────┘ └────────────────┘
```

### Document Chunker (`DocumentChunker`)

Uses LangChain text splitters with tiktoken tokenization.

```
┌───────────────┐
│ Markdown Text │
└───────┬───────┘
        │
        ▼
┌───────────────────┐
│ Content type?     │
├─────────┬─────────┤
│Markdown │ Plain   │
└────┬────┴────┬────┘
     │         │
     ▼         ▼
┌──────────┐ ┌────────────────────────┐
│Markdown  │ │RecursiveCharacter      │
│TextSplit │ │TextSplitter            │
└────┬─────┘ └───────────┬────────────┘
     │                   │
     └─────────┬─────────┘
               │
               ▼
     ┌─────────────────┐
     │ Chunk objects   │
     │ + heading_path  │
     └─────────────────┘
```

Configuration:
- `chunk_size_tokens`: 512 (default)
- `chunk_overlap_tokens`: 50 (default)
- Encoding: `cl100k_base` (matches OpenAI embeddings)

## ZIP Extraction Security

```
┌──────────┐
│ ZIP File │
└────┬─────┘
     │
     ▼
┌────────────────┐
│ Valid ZIP?     │
├────────┬───────┤
│   NO   │  YES  │
└────┬───┴───┬───┘
     │       │
     ▼       ▼
┌─────────┐ ┌──────────────────┐
│ Raise   │ │ Depth > 3?       │
│ValueError│ ├──────────┬───────┤
└─────────┘ │   YES    │  NO   │
            └────┬─────┴───┬───┘
                 │         │
                 ▼         ▼
       ┌──────────────┐ ┌────────────────────┐
       │ZipExtraction │ │ For each member:   │
       │Error         │ └─────────┬──────────┘
       └──────────────┘           │
                                  ▼
                     ┌────────────────────────┐
                     │ Path traversal?        │
                     │ (../ or absolute)      │
                     ├────────────┬───────────┤
                     │    YES     │    NO     │
                     └─────┬──────┴─────┬─────┘
                           │            │
                           ▼            ▼
                    ┌───────────┐ ┌───────────────┐
                    │Skip file  │ │Hidden file?   │
                    └───────────┘ │(. or __MACOSX)│
                                  ├───────┬───────┤
                                  │  YES  │  NO   │
                                  └───┬───┴───┬───┘
                                      │       │
                                      ▼       ▼
                               ┌──────────┐ ┌──────────────┐
                               │Skip file │ │Total size OK?│
                               └──────────┘ │(< 500MB)     │
                                            ├──────┬───────┤
                                            │  NO  │  YES  │
                                            └──┬───┴───┬───┘
                                               │       │
                                               ▼       ▼
                                  ┌──────────────┐ ┌─────────────┐
                                  │ZipExtraction │ │File count OK│
                                  │Error         │ │(< 100)      │
                                  └──────────────┘ ├──────┬──────┤
                                                   │  NO  │ YES  │
                                                   └──┬───┴──┬───┘
                                                      │      │
                                                      ▼      ▼
                                         ┌──────────────┐ ┌────────────┐
                                         │ZipExtraction │ │Extract file│
                                         │Error         │ └─────┬──────┘
                                         └──────────────┘       │
                                                                ▼
                                                    ┌───────────────────┐
                                                    │Nested ZIP?        │
                                                    ├─────────┬─────────┤
                                                    │   YES   │   NO    │
                                                    └────┬────┴────┬────┘
                                                         │         │
                                                         ▼         ▼
                                                ┌─────────────┐ ┌────────────┐
                                                │Recurse with │ │Add to      │
                                                │depth + 1    │ │results     │
                                                └─────────────┘ └────────────┘
```

**Limits (configurable via environment):**
- `ZIP_MAX_DEPTH`: 3
- `ZIP_MAX_FILES`: 100
- `ZIP_MAX_TOTAL_SIZE_MB`: 500

## Query Flow

```
┌─────────────────┐
│   User Query    │
└────────┬────────┘
         │
         ▼
┌──────────────────────┐
│ Vessel selected?     │
├──────────┬───────────┤
│   YES    │    NO     │
└────┬─────┴─────┬─────┘
     │           │
     ▼           │
┌────────────────┐  │
│Build metadata  │  │
│filter:         │  │
│vessel_ids      │  │
│contains uuid   │  │
└────────┬───────┘  │
         │          │
         └────┬─────┘
              │
              ▼
    ┌─────────────────────┐
    │ Generate embedding  │
    └──────────┬──────────┘
               │
               ▼
    ┌─────────────────────┐
    │ Vector search       │
    │ top_k=20            │
    │ + metadata filter   │
    └──────────┬──────────┘
               │
               ▼
    ┌─────────────────────┐
    │ Reranking enabled?  │
    ├──────────┬──────────┤
    │   YES    │    NO    │
    └────┬─────┴─────┬────┘
         │           │
         ▼           │
  ┌─────────────┐    │
  │Cohere rerank│    │
  │top_n=5      │    │
  └──────┬──────┘    │
         │           │
         └─────┬─────┘
               │
               ▼
    ┌─────────────────────┐
    │ Build context from  │
    │ final results       │
    └──────────┬──────────┘
               │
               ▼
    ┌─────────────────────┐
    │ GPT-4o generate     │
    │ answer              │
    └──────────┬──────────┘
               │
               ▼
    ┌─────────────────────┐
    │ Answer + Sources    │
    └─────────────────────┘
```

### Vessel Filtering

When a user selects a vessel in the UI, the search is filtered to only return chunks from documents tagged with that vessel.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ FRONTEND                                                                     │
│                                                                             │
│   ┌────────────────┐      ┌─────────────────────┐                          │
│   │ Vessel Dropdown│─────▶│ CopilotKit props    │                          │
│   │ (UI component) │      │ selected_vessel_id  │                          │
│   └────────────────┘      └──────────┬──────────┘                          │
└──────────────────────────────────────┼──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ AGENT                                                                        │
│                                                                             │
│   ┌─────────────────┐      ┌─────────────────────┐                         │
│   │ search_node     │─────▶│ query_engine.search │                         │
│   │ (LangGraph)     │      │ with metadata_filter│                         │
│   └─────────────────┘      └──────────┬──────────┘                         │
└──────────────────────────────────────┼──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ DATABASE                                                                     │
│                                                                             │
│   ┌─────────────────────┐      ┌─────────────────────┐                     │
│   │ match_chunks()      │─────▶│ chunks.metadata     │                     │
│   │ vessel_ids contains │      │ @> filter           │                     │
│   │ uuid                │      │                     │                     │
│   └─────────────────────┘      └──────────┬──────────┘                     │
│                                           │                                 │
│                                           ▼                                 │
│                                ┌─────────────────────┐                     │
│                                │ Filtered results    │                     │
│                                └─────────────────────┘                     │
└─────────────────────────────────────────────────────────────────────────────┘
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
| `ingest_events` | Processing events (errors, warnings, unsupported files) |
| `processing_log` | Progress tracking for resume |
