import { createBrowserClient } from "@supabase/ssr";
import { SupabaseClient } from "@supabase/supabase-js";

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

function createSupabaseClient(): SupabaseClient {
  if (!supabaseUrl || !supabaseAnonKey) {
    // In development, provide a helpful error. In production, this will
    // fail at runtime when auth is attempted rather than at build time.
    console.error(
      "Missing Supabase environment variables. Set NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY."
    );
  }

  // Use @supabase/ssr's createBrowserClient which stores session in cookies
  // This enables server-side routes to access the session via cookies
  return createBrowserClient(supabaseUrl || "", supabaseAnonKey || "");
}

export const supabase = createSupabaseClient();
