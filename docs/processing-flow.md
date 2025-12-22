# NCL Processing Flow

This document describes the data flow through the NCL email RAG pipeline.

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
└── processed/                 # NCL-generated data
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

## High-Level Overview

```mermaid
flowchart TB
    subgraph Input["Source Data (read-only)"]
        EML[EML Files<br/>with subdirectories]
    end

    subgraph Parsing["Parsing Layer"]
        EP[EML Parser]
        AP[Attachment Processor]
        ZIP[ZIP Extractor]
    end

    subgraph Processing["Processing Layer"]
        HM[Hierarchy Manager]
        EMB[Embedding Generator]
        RR[Reranker]
    end

    subgraph Storage["Storage Layer"]
        DB[(Supabase)]
        FR[File Registry]
        UF[Unsupported Files Log]
    end

    subgraph Query["Query Layer"]
        QE[Query Engine]
        LLM[LLM / GPT-4o-mini]
    end

    EML --> EP
    EP --> |Email Body| HM
    EP --> |Attachments| AP
    AP --> |ZIP files| ZIP
    ZIP --> |Extracted files| AP
    AP --> |Unsupported| UF
    AP --> |Chunks| HM
    HM --> |Documents| DB
    HM --> |Chunks| EMB
    EMB --> |Embedded Chunks| DB
    FR --> |Track Files| DB

    User([User Query]) --> QE
    QE --> |Vector Search| DB
    DB --> |Results| RR
    RR --> |Reranked| QE
    QE --> |Context| LLM
    LLM --> |Answer| User
```

## Detailed Ingestion Flow

```mermaid
flowchart TD
    subgraph Ingest["ncl ingest command"]
        START([Start]) --> SCAN[Scan source directory<br/>including subdirectories]
        SCAN --> REGISTRY{In file registry?}

        REGISTRY --> |Hash match + completed| SKIP[Skip file]
        REGISTRY --> |New or changed| REGISTER[Register in file_registry]

        REGISTER --> PARSE[Parse EML]

        PARSE --> BODY[Extract email body]
        PARSE --> ATTACH[Extract attachments<br/>to processed/attachments/]

        BODY --> CHUNK1[Create body chunk]

        ATTACH --> ZIPCHECK{Is ZIP file?}
        ZIPCHECK --> |Yes| EXTRACT[Extract to<br/>processed/extracted/]
        EXTRACT --> NESTED{Nested ZIP?}
        NESTED --> |Yes| EXTRACT
        NESTED --> |No| CHECKEXT[Check extracted files]

        ZIPCHECK --> |No| SUPPORTED{Supported format?}
        CHECKEXT --> SUPPORTED

        SUPPORTED --> |Yes| PROCESS[Process with Docling]
        SUPPORTED --> |No| LOGUNSUP[Log to unsupported_files]

        LOGUNSUP --> NEXTATT([Next attachment])

        PROCESS --> CHUNK2[Create attachment chunks]

        CHUNK1 --> EMBED[Generate embeddings]
        CHUNK2 --> EMBED

        EMBED --> STORE[Store in Supabase]
        STORE --> UPDATE[Update file_registry<br/>status = completed]
        UPDATE --> END([Next file])

        SKIP --> END
    end
```

## ZIP Extraction Flow

```mermaid
flowchart TD
    subgraph ZIPFlow["ZIP Extraction Process"]
        ZIPFILE[ZIP Attachment] --> VALIDATE{Valid ZIP?}

        VALIDATE --> |No| ERROR[Log error]
        VALIDATE --> |Yes| SECURITY[Security checks]

        SECURITY --> PATHCHECK{Path traversal?}
        PATHCHECK --> |Yes| SKIPFILE[Skip dangerous file]
        PATHCHECK --> |No| HIDDEN{Hidden file?}

        HIDDEN --> |.file or __MACOSX| SKIPFILE
        HIDDEN --> |No| EXTRACT[Extract to<br/>processed/extracted/]

        EXTRACT --> CHECKTYPE{File type?}

        CHECKTYPE --> |Nested ZIP| RECURSE[Recursively extract]
        RECURSE --> SECURITY

        CHECKTYPE --> |Supported| ADDLIST[Add to processing list]
        CHECKTYPE --> |Unsupported| LOGUNSUP[Log to unsupported_files<br/>with ZIP context]

        ADDLIST --> RETURN[Return extracted files]
        LOGUNSUP --> RETURN
        SKIPFILE --> RETURN
        ERROR --> RETURN
    end
```

## Attachment Processing Detail

```mermaid
flowchart LR
    subgraph AttachProc["Attachment Processing"]
        INPUT[Attachment] --> TYPECHECK{Type?}

        TYPECHECK --> |ZIP| ZIPPROC[ZIP Extractor]
        ZIPPROC --> |Extracted files| TYPECHECK

        TYPECHECK --> |PDF/DOCX/PPTX/XLSX| DOCLING[Docling Converter]
        TYPECHECK --> |Image| IMGPROC[Image Processor]
        TYPECHECK --> |HTML| DOCLING
        TYPECHECK --> |Unsupported| UNSUP[Log unsupported]

        subgraph Docling["Docling Processing"]
            DOCLING --> OCR{OCR Enabled?}
            OCR --> |Yes| EASYOCR[EasyOCR]
            OCR --> |No| DOC[Document Object]
            EASYOCR --> DOC
        end

        subgraph ImageProc["Image Processing"]
            IMGPROC --> DESC{Picture Description?}
            DESC --> |Yes| SMOLVLM[SmolVLM Model]
            DESC --> |No| IMGDOC[Image Document]
            SMOLVLM --> IMGDOC
        end

        DOC --> CHUNKER[HybridChunker]
        IMGDOC --> CHUNKER
        CHUNKER --> |OpenAI Tokenizer| CHUNKS[Semantic Chunks]
    end
```

## File Registry Flow

```mermaid
flowchart TD
    subgraph FileRegistry["File Registry System"]
        NEWFILE[New EML file] --> HASH[Compute SHA-256 hash]
        HASH --> LOOKUP[Query file_registry]

        LOOKUP --> EXISTS{Exists?}

        EXISTS --> |No| INSERT[Insert new record<br/>status = pending]
        INSERT --> PROCESS[Process file]

        EXISTS --> |Yes| CHECKSTATUS{Status?}

        CHECKSTATUS --> |completed| CHECKHASH{Hash changed?}
        CHECKHASH --> |No| SKIPFILE[Skip - already processed]
        CHECKHASH --> |Yes| REPROCESS[Reprocess file]

        CHECKSTATUS --> |failed| RETRY{Attempts < 3?}
        RETRY --> |Yes| REPROCESS
        RETRY --> |No| SKIPFILE

        CHECKSTATUS --> |pending/processing| REPROCESS

        PROCESS --> SUCCESS{Success?}
        REPROCESS --> SUCCESS

        SUCCESS --> |Yes| COMPLETE[Update status = completed<br/>Link document_id]
        SUCCESS --> |No| FAIL[Update status = failed<br/>Increment attempts]
    end
```

## Query Flow with Two-Stage Retrieval

```mermaid
flowchart TD
    subgraph Query["ncl query command"]
        Q([User Question]) --> EMBED[Generate Query Embedding]

        EMBED --> VECTOR[Vector Search]
        VECTOR --> |top_k=20 candidates| STAGE1[Stage 1 Results]

        STAGE1 --> RERANK{Reranking Enabled?}
        RERANK --> |Yes| CROSS[Cross-Encoder Reranking]
        CROSS --> |LiteLLM rerank| STAGE2[Stage 2: Top 5 Results]
        RERANK --> |No| STAGE2

        STAGE2 --> CONTEXT[Build Context]
        CONTEXT --> |Sources + Headers| PROMPT[Create LLM Prompt]

        PROMPT --> LLM[GPT-4o-mini]
        LLM --> ANSWER[Generated Answer]

        ANSWER --> FORMAT[Format with Sources]
        FORMAT --> OUTPUT([Display to User])
    end
```

## Document Hierarchy

```mermaid
flowchart TD
    subgraph Hierarchy["Document Hierarchy Example"]
        EMAIL[Email Document<br/>depth=0, root_id=self]

        EMAIL --> PDF[PDF Attachment<br/>depth=1]
        EMAIL --> DOCX[DOCX Attachment<br/>depth=1]
        EMAIL --> ZIP[ZIP Attachment<br/>depth=1]

        ZIP --> IMG1[Image from ZIP<br/>depth=2]
        ZIP --> IMG2[Another Image<br/>depth=2]
        ZIP --> NESTED[Nested ZIP<br/>depth=2]

        NESTED --> PDF2[PDF from nested ZIP<br/>depth=3]

        PDF --> C1[Chunk 1]
        PDF --> C2[Chunk 2]
        DOCX --> C3[Chunk 1]
        IMG1 --> C4[Chunk 1<br/>Image Description]
        PDF2 --> C5[Chunk 1]
    end
```

## Unsupported Files Tracking

```mermaid
flowchart LR
    subgraph UnsupportedFlow["Unsupported Files Logging"]
        UNSUP[Unsupported File] --> RECORD[Record in unsupported_files]

        RECORD --> INFO[Store metadata:]
        INFO --> F1[file_name]
        INFO --> F2[file_path]
        INFO --> F3[mime_type]
        INFO --> F4[file_extension]
        INFO --> F5[source_eml_path]
        INFO --> F6[source_zip_path]
        INFO --> F7[reason]

        F7 --> R1[unsupported_format]
        F7 --> R2[too_large]
        F7 --> R3[corrupted]
        F7 --> R4[extraction_failed]
    end
```

## Data Flow Summary

| Stage | Input | Output | Component |
|-------|-------|--------|-----------|
| 1. Scan | Source directory | EML file list | CLI |
| 2. Check | File path + hash | Need processing? | File Registry |
| 3. Parse | EML file | ParsedEmail | EMLParser |
| 4. Extract | Attachments | Saved files | EMLParser → processed/attachments/ |
| 5. Unzip | ZIP files | Extracted files | AttachmentProcessor → processed/extracted/ |
| 6. Log | Unsupported files | Database record | Unsupported Files Table |
| 7. Convert | Documents | Docling Document | AttachmentProcessor |
| 8. Chunk | Document | Semantic chunks | HybridChunker |
| 9. Embed | Chunks | 1536-dim vectors | EmbeddingGenerator |
| 10. Store | Embedded chunks | Database records | SupabaseClient |
| 11. Update | Processing result | File registry update | File Registry |
| 12. Search | Query embedding | Similar chunks | pgvector |
| 13. Rerank | Candidates | Ranked results | Reranker |
| 14. Generate | Context | Answer | LiteLLM/GPT-4o |

## Database Tables

| Table | Purpose |
|-------|---------|
| `documents` | Document hierarchy (emails, attachments) |
| `chunks` | Text chunks with embeddings |
| `file_registry` | Track all files for quick processing lookup |
| `unsupported_files` | Log unsupported files for visibility |
| `processing_log` | Legacy progress tracking (deprecated) |
