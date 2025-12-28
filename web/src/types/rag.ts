/**
 * RAG State types for state sharing between Python agent and frontend.
 */

export interface RAGState {
  is_searching: boolean;
  search_progress: string;
  current_query: string | null;
  error_message: string | null;
}

export const initialRAGState: RAGState = {
  is_searching: false,
  search_progress: "",
  current_query: null,
  error_message: null,
};
