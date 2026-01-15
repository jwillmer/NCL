/**
 * Conversations API client for managing chat history.
 */

import { supabase } from "./supabase";

// ============================================
// Types
// ============================================

export interface Conversation {
  id: string;
  thread_id: string;
  user_id: string;
  title: string | null;
  vessel_id: string | null;  // UUID of selected vessel (null = all vessels)
  is_archived: boolean;
  created_at: string;
  updated_at: string;
  last_message_at: string | null;
}

export interface ConversationListResponse {
  items: Conversation[];
  total: number;
  has_more: boolean;
}

export interface CreateConversationParams {
  thread_id?: string;
  title?: string;
  vessel_id?: string;  // UUID of vessel (null = all vessels)
}

export interface UpdateConversationParams {
  title?: string;
  vessel_id?: string | null;  // UUID of vessel, or null to clear filter
  is_archived?: boolean;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  vessel_id?: string | null;  // Vessel filter active when message was sent
}

export interface Vessel {
  id: string;
  name: string;
  imo: string | null;
  vessel_type: string | null;
}

// ============================================
// API Error Handling
// ============================================

export class ConversationApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public code?: string
  ) {
    super(message);
    this.name = "ConversationApiError";
  }
}

async function getAuthHeaders(): Promise<HeadersInit> {
  const {
    data: { session },
  } = await supabase.auth.getSession();
  if (!session?.access_token) {
    throw new ConversationApiError("Not authenticated", 401, "UNAUTHENTICATED");
  }
  return {
    "Content-Type": "application/json",
    Authorization: `Bearer ${session.access_token}`,
  };
}

export function getApiBaseUrl(): string {
  return process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let message = "An error occurred";
    let code: string | undefined;

    try {
      const error = await response.json();
      message = error.detail || error.message || message;
      code = error.code;
    } catch {
      message = response.statusText || message;
    }

    throw new ConversationApiError(message, response.status, code);
  }

  // Handle 204 No Content
  if (response.status === 204) {
    return undefined as T;
  }

  return response.json();
}

// ============================================
// API Functions
// ============================================

/**
 * List user's conversations with optional search and pagination.
 */
export async function listConversations(params?: {
  q?: string;
  archived?: boolean;
  limit?: number;
  offset?: number;
}): Promise<ConversationListResponse> {
  const headers = await getAuthHeaders();
  const searchParams = new URLSearchParams();

  if (params?.q) searchParams.set("q", params.q);
  if (params?.archived !== undefined)
    searchParams.set("archived", String(params.archived));
  if (params?.limit !== undefined)
    searchParams.set("limit", String(params.limit));
  if (params?.offset !== undefined)
    searchParams.set("offset", String(params.offset));

  const queryString = searchParams.toString();
  const url = `${getApiBaseUrl()}/conversations${queryString ? `?${queryString}` : ""}`;

  const response = await fetch(url, { headers });
  return handleResponse<ConversationListResponse>(response);
}

/**
 * Create a new conversation.
 */
export async function createConversation(
  params?: CreateConversationParams
): Promise<Conversation> {
  const headers = await getAuthHeaders();

  const response = await fetch(`${getApiBaseUrl()}/conversations`, {
    method: "POST",
    headers,
    body: JSON.stringify(params || {}),
  });

  return handleResponse<Conversation>(response);
}

/**
 * Get a conversation by thread_id.
 */
export async function getConversation(
  threadId: string
): Promise<Conversation> {
  const headers = await getAuthHeaders();

  const response = await fetch(
    `${getApiBaseUrl()}/conversations/${threadId}`,
    { headers }
  );

  return handleResponse<Conversation>(response);
}

/**
 * Update a conversation's metadata.
 */
export async function updateConversation(
  threadId: string,
  params: UpdateConversationParams
): Promise<Conversation> {
  const headers = await getAuthHeaders();

  const response = await fetch(
    `${getApiBaseUrl()}/conversations/${threadId}`,
    {
      method: "PATCH",
      headers,
      body: JSON.stringify(params),
    }
  );

  return handleResponse<Conversation>(response);
}

/**
 * Delete a conversation.
 */
export async function deleteConversation(threadId: string): Promise<void> {
  const headers = await getAuthHeaders();

  const response = await fetch(
    `${getApiBaseUrl()}/conversations/${threadId}`,
    {
      method: "DELETE",
      headers,
    }
  );

  return handleResponse<void>(response);
}

/**
 * Update last_message_at timestamp (called when a new message is sent).
 */
export async function touchConversation(
  threadId: string
): Promise<Conversation> {
  const headers = await getAuthHeaders();

  const response = await fetch(
    `${getApiBaseUrl()}/conversations/${threadId}/touch`,
    {
      method: "POST",
      headers,
    }
  );

  return handleResponse<Conversation>(response);
}

/**
 * Generate and set conversation title from first message content.
 * @param force - Force regeneration even if title exists
 */
export async function generateTitle(
  threadId: string,
  content: string,
  force?: boolean
): Promise<Conversation> {
  const headers = await getAuthHeaders();

  const response = await fetch(
    `${getApiBaseUrl()}/conversations/${threadId}/generate-title`,
    {
      method: "POST",
      headers,
      body: JSON.stringify({ content, force: force ?? false }),
    }
  );

  return handleResponse<Conversation>(response);
}

/**
 * Get all messages for a conversation from LangGraph checkpoints.
 * Used by frontend to load conversation history on page mount.
 */
export async function getMessages(threadId: string): Promise<ChatMessage[]> {
  const headers = await getAuthHeaders();

  const response = await fetch(
    `${getApiBaseUrl()}/conversations/${threadId}/messages`,
    { headers }
  );

  const data = await handleResponse<{ messages: ChatMessage[] }>(response);
  return data.messages;
}

/**
 * Get all vessels from the registry for dropdown selection.
 */
export async function listVessels(): Promise<Vessel[]> {
  const headers = await getAuthHeaders();

  const response = await fetch(`${getApiBaseUrl()}/vessels`, { headers });

  return handleResponse<Vessel[]>(response);
}

/**
 * Submit user feedback for a chat message.
 * Feedback is stored in Langfuse linked to the conversation trace.
 *
 * @param threadId - The conversation thread ID
 * @param messageId - The message ID that received feedback
 * @param value - 1 for positive (thumbs up), 0 for negative (thumbs down)
 */
export async function submitFeedback(
  threadId: string,
  messageId: string,
  value: 0 | 1
): Promise<void> {
  const headers = await getAuthHeaders();

  const response = await fetch(`${getApiBaseUrl()}/feedback`, {
    method: "POST",
    headers,
    body: JSON.stringify({
      thread_id: threadId,
      message_id: messageId,
      value,
    }),
  });

  await handleResponse<{ status: string }>(response);
}
