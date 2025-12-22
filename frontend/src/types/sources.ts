/**
 * Source reference from RAG query response.
 */
export interface Source {
  file_path: string;
  document_type:
    | "email"
    | "attachment_pdf"
    | "attachment_image"
    | "attachment_docx"
    | "attachment_pptx"
    | "attachment_xlsx"
    | "attachment_other";
  email_subject: string | null;
  email_initiator: string | null;
  email_participants: string[] | null;
  email_date: string | null;
  chunk_content: string;
  similarity_score: number;
  rerank_score: number | null;
  heading_path: string[];
  root_file_path?: string | null;
}

/**
 * RAG query response from the API.
 */
export interface RAGResponse {
  answer: string;
  sources: Source[];
}
