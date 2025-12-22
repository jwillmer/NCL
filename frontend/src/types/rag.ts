/**
 * RAG State types matching the Python backend models.
 * Keep in sync with src/ncl/api/state.py
 */

export interface SourceReference {
  chunk_id: string;
  file_path: string;
  document_type: string;
  email_subject: string | null;
  email_initiator: string | null;
  email_participants: string[] | null;
  email_date: string | null;
  chunk_content: string;
  similarity_score: number;
  rerank_score: number | null;
  heading_path: string | null;
  root_file_path: string | null;
}

export interface RAGState {
  sources: SourceReference[];
  current_query: string | null;
  answer: string | null;
  is_searching: boolean;
  error_message: string | null;
}

export const initialRAGState: RAGState = {
  sources: [],
  current_query: null,
  answer: null,
  is_searching: false,
  error_message: null,
};
