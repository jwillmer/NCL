/**
 * Conversations API client for managing chat history.
 */

import { getConfig } from "./config";
import { getAuthHeaders as _getAuthHeaders, clearAuthCache } from "./authCache";

// ============================================
// Types
// ============================================

export interface Conversation {
  id: string;
  thread_id: string;
  user_id: string;
  title: string | null;
  vessel_id: string | null;
  vessel_type: string | null;
  vessel_class: string | null;
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
  vessel_id?: string;
  vessel_type?: string;
  vessel_class?: string;
}

export interface UpdateConversationParams {
  title?: string;
  vessel_id?: string | null;
  vessel_type?: string | null;
  vessel_class?: string | null;
  is_archived?: boolean;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  vessel_id?: string | null;
}

export interface Vessel {
  id: string;
  name: string;
  vessel_type: string;
  vessel_class: string;
}

// ============================================
// API Helpers
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

export function getApiBaseUrl(): string {
  const config = getConfig();
  const baseUrl = config.API_URL || "";
  return baseUrl.endsWith("/api") ? baseUrl : `${baseUrl}/api`;
}

/**
 * Get auth headers from Supabase session.
 * Exported for use by useChat transport.
 *
 * Implementation lives in `./authCache` to allow session-change hooks
 * (onAuthStateChange) to clear the cached token without creating a
 * circular import with this module. Re-exported here to preserve the
 * public API surface.
 */
export const getAuthHeaders = _getAuthHeaders;

/**
 * Typed fetch helper — handles auth, JSON, and error wrapping.
 * On 401 the auth-token cache is cleared and the request is retried once.
 */
async function typedFetch<T>(
  method: string,
  path: string,
  body?: unknown,
  retried = false
): Promise<T> {
  const headers = await getAuthHeaders();
  const url = `${getApiBaseUrl()}${path}`;

  const response = await fetch(url, {
    method,
    headers,
    ...(body !== undefined ? { body: JSON.stringify(body) } : {}),
  });

  if (response.status === 401 && !retried) {
    clearAuthCache();
    return typedFetch<T>(method, path, body, true);
  }

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

  if (response.status === 204) return undefined as T;
  return response.json();
}

// ============================================
// API Functions
// ============================================

export async function listConversations(params?: {
  q?: string;
  archived?: boolean;
  limit?: number;
  offset?: number;
}): Promise<ConversationListResponse> {
  const searchParams = new URLSearchParams();
  if (params?.q) searchParams.set("q", params.q);
  if (params?.archived !== undefined) searchParams.set("archived", String(params.archived));
  if (params?.limit !== undefined) searchParams.set("limit", String(params.limit));
  if (params?.offset !== undefined) searchParams.set("offset", String(params.offset));
  const qs = searchParams.toString();
  return typedFetch("GET", `/conversations${qs ? `?${qs}` : ""}`);
}

export async function createConversation(params?: CreateConversationParams): Promise<Conversation> {
  return typedFetch("POST", "/conversations", params || {});
}

export async function getConversation(threadId: string): Promise<Conversation> {
  return typedFetch("GET", `/conversations/${threadId}`);
}

export async function updateConversation(threadId: string, params: UpdateConversationParams): Promise<Conversation> {
  return typedFetch("PATCH", `/conversations/${threadId}`, params);
}

export async function deleteConversation(threadId: string): Promise<void> {
  return typedFetch("DELETE", `/conversations/${threadId}`);
}

export async function touchConversation(threadId: string): Promise<Conversation> {
  return typedFetch("POST", `/conversations/${threadId}/touch`);
}

export async function generateTitle(threadId: string, content: string, force?: boolean): Promise<Conversation> {
  return typedFetch("POST", `/conversations/${threadId}/generate-title`, { content, force: force ?? false });
}

export async function getMessages(threadId: string): Promise<ChatMessage[]> {
  const data = await typedFetch<{ messages: ChatMessage[] }>("GET", `/conversations/${threadId}/messages`);
  return data.messages;
}

export async function listVessels(): Promise<Vessel[]> {
  return typedFetch("GET", "/vessels");
}

export async function listVesselTypes(): Promise<string[]> {
  return typedFetch("GET", "/vessel-types");
}

export async function listVesselClasses(): Promise<string[]> {
  return typedFetch("GET", "/vessel-classes");
}

export async function submitFeedback(threadId: string, messageId: string, value: 0 | 1): Promise<void> {
  await typedFetch<{ status: string }>("POST", "/feedback", {
    thread_id: threadId,
    message_id: messageId,
    value,
  });
}
