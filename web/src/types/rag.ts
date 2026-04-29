/**
 * RAG State types for state sharing between Python agent and frontend.
 */

export interface RAGState {
  search_progress: string;
  error_message: string | null;
  selected_vessel_id: string | null;
}

export const initialRAGState: RAGState = {
  search_progress: "",
  error_message: null,
  selected_vessel_id: null,
};

/**
 * Citation details returned from /citations/{chunk_id} API.
 */
export interface Citation {
  chunk_id: string;
  source_title: string | null;
  page: number | null;
  lines: [number, number] | null;
  archive_browse_uri: string | null;
  archive_download_uri: string | null;
  archive_download_signed_url: string | null;
  content: string | null;
  origin_email: OriginEmail | null;
  // Document type of the citation's source (e.g. "email", "attachment_pdf").
  // Null when the backend could not resolve it.
  document_type: string | null;
  // For email documents: number of direct attachments. Null for non-email
  // sources or when lookup failed.
  attachment_count: number | null;
}

export interface OriginEmail {
  subject: string | null;
  chunk_id: string;
}

/**
 * Props for citation elements rendered from <cite> tags.
 * Note: All attributes come through as strings from HTML parsing.
 */
export interface CiteProps {
  id?: string;
  doc?: string;  // Stable source/document id — used by the UI to dedupe chunks of the same source
  title?: string;
  page?: string;
  lines?: string;
  download?: string;
  children?: React.ReactNode;
}

/**
 * Single citation entry on the wire (v1 `data-citations` SSE frame).
 * Mirrors `CitationProcessor.serialize_citations_payload` in
 * `src/mtss/rag/citation_processor.py` — keep the field set in sync.
 */
export interface CitationPayload {
  chunk_id: string;
  index: number;
  source_id: string | null;
  source_title: string | null;
  page: number | null;
  lines: [number, number] | null;
  archive_browse_uri: string | null;
  archive_download_uri: string | null;
  archive_verified: boolean;
}

/**
 * Wire shape for the `data-citations` SSE frame emitted after the LLM
 * stream completes. Consumed by `applyCitationsToMarkdown` in `lib/utils.ts`
 * to swap raw `[C:chunk_id]` markers for fully-attributed `<cite>` tags.
 */
export interface CitationsFrame {
  version: 1;
  citations: CitationPayload[];
  invalid_chunk_ids: string[];
}
