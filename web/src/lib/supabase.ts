import { createBrowserClient } from "@supabase/ssr";
import { SupabaseClient } from "@supabase/supabase-js";

import { getConfig } from "./config";

let _supabase: SupabaseClient | null = null;

/**
 * Get the Supabase client (lazy initialization).
 * This avoids creating the client at module load time, which would fail
 * during Next.js static export when env vars are not available.
 *
 * Uses runtime config from /config.js in Docker deployments,
 * falls back to process.env in development.
 */
export function getSupabase(): SupabaseClient {
  if (_supabase) {
    return _supabase;
  }

  const config = getConfig();
  const supabaseUrl = config.SUPABASE_URL;
  const supabaseAnonKey = config.SUPABASE_ANON_KEY;

  if (!supabaseUrl || !supabaseAnonKey) {
    throw new Error(
      "Missing Supabase configuration. Set NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY."
    );
  }

  // Use @supabase/ssr's createBrowserClient which stores session in cookies
  // This enables server-side routes to access the session via cookies
  _supabase = createBrowserClient(supabaseUrl, supabaseAnonKey);
  return _supabase;
}

/**
 * @deprecated Use getSupabase() instead for lazy initialization.
 * This export is kept for backwards compatibility but may cause build issues.
 */
export const supabase = typeof window !== "undefined" ? getSupabase() : (null as unknown as SupabaseClient);
