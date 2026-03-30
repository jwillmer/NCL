/**
 * Runtime configuration for the frontend.
 *
 * In Docker deployments, config is loaded from /config.js served by FastAPI.
 * In development, falls back to process.env values.
 *
 * The /config.js endpoint returns a script that sets window.__MTSS_CONFIG__.
 */

export interface RuntimeConfig {
  SUPABASE_URL: string;
  SUPABASE_ANON_KEY: string;
  API_URL: string;
  LANGFUSE_PUBLIC_KEY?: string;
  LANGFUSE_BASE_URL?: string;
}

declare global {
  interface Window {
    __MTSS_CONFIG__?: RuntimeConfig;
  }
}

/**
 * Get runtime configuration.
 * Prefers window.__MTSS_CONFIG__ (set by /config.js in Docker),
 * falls back to process.env for development.
 */
export function getConfig(): RuntimeConfig {
  // Check for runtime config (Docker deployment)
  if (typeof window !== "undefined" && window.__MTSS_CONFIG__) {
    return window.__MTSS_CONFIG__;
  }

  // Fallback to build-time env vars (development)
  return {
    SUPABASE_URL: import.meta.env.VITE_SUPABASE_URL || "",
    SUPABASE_ANON_KEY: import.meta.env.VITE_SUPABASE_ANON_KEY || "",
    API_URL: import.meta.env.VITE_API_URL || "",
    LANGFUSE_PUBLIC_KEY: import.meta.env.VITE_LANGFUSE_PUBLIC_KEY,
    LANGFUSE_BASE_URL: import.meta.env.VITE_LANGFUSE_BASE_URL,
  };
}
