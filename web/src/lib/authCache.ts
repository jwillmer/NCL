/**
 * In-memory auth-token cache.
 *
 * Caches the Supabase session access token to avoid calling
 * `supabase.auth.getSession()` (localStorage + occasional network refresh)
 * on every API request.
 *
 * Lives in its own module to avoid a circular import between
 * `auth.tsx` (which calls `clearAuthCache` on state change) and
 * `conversations.ts` (which reads from the cache for every fetch).
 */

import { getSupabase } from "./supabase";
import { ConversationApiError } from "./conversations";

let _cached: { token: string; exp: number } | null = null;

export async function getAuthHeaders(): Promise<Record<string, string>> {
  const now = Date.now() / 1000;
  if (_cached && _cached.exp - 30 > now) {
    return {
      "Content-Type": "application/json",
      Authorization: `Bearer ${_cached.token}`,
    };
  }
  const {
    data: { session },
  } = await getSupabase().auth.getSession();
  if (!session?.access_token) {
    throw new ConversationApiError("Not authenticated", 401, "UNAUTHENTICATED");
  }
  _cached = {
    token: session.access_token,
    exp: session.expires_at ?? now + 3000,
  };
  return {
    "Content-Type": "application/json",
    Authorization: `Bearer ${session.access_token}`,
  };
}

export function clearAuthCache(): void {
  _cached = null;
}
