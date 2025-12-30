/**
 * RAG State types for state sharing between Python agent and frontend.
 */

export interface RAGState {
  search_progress: string;
  error_message: string | null;
}

export const initialRAGState: RAGState = {
  search_progress: "",
  error_message: null,
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
  content: string | null;
}

/**
 * Props for citation elements rendered from <cite> tags.
 * Note: All attributes come through as strings from HTML parsing.
 */
export interface CiteProps {
  id?: string;
  title?: string;
  page?: string;
  lines?: string;
  download?: string;
  children?: React.ReactNode;
}
