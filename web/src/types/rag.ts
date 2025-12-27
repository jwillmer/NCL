/**
 * RAG State types for state sharing between Python agent and frontend.
 */

export interface RAGState {
  is_searching: boolean;
  current_query: string | null;
  error_message: string | null;
}

export const initialRAGState: RAGState = {
  is_searching: false,
  current_query: null,
  error_message: null,
};
