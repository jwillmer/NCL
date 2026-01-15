/**
 * Langfuse Browser SDK wrapper for client-side user interaction tracking.
 *
 * Uses the same session ID (thread_id) as the backend for consistent trace grouping.
 * Only public key is used - safe for browser environments.
 */

import { LangfuseWeb } from "langfuse";

let langfuseWeb: LangfuseWeb | null = null;

/**
 * Initialize Langfuse browser SDK.
 * Call this once on app startup.
 */
export function initLangfuse(): void {
  const publicKey = process.env.NEXT_PUBLIC_LANGFUSE_PUBLIC_KEY;
  const baseUrl = process.env.NEXT_PUBLIC_LANGFUSE_BASE_URL;

  if (!publicKey) {
    console.debug("Langfuse not configured for browser (missing NEXT_PUBLIC_LANGFUSE_PUBLIC_KEY)");
    return;
  }

  try {
    langfuseWeb = new LangfuseWeb({
      publicKey,
      baseUrl: baseUrl || "https://cloud.langfuse.com",
    });
    console.debug("Langfuse browser SDK initialized");
  } catch (error) {
    console.error("Failed to initialize Langfuse browser SDK:", error);
  }
}

/**
 * Track user feedback (thumbs up/down) for a message.
 *
 * @param sessionId - The conversation thread_id (matches backend session)
 * @param messageId - The message ID being rated
 * @param value - 0 for thumbs down, 1 for thumbs up
 */
export function trackFeedback(sessionId: string, messageId: string, value: 0 | 1): void {
  if (!langfuseWeb) {
    console.debug("Langfuse not initialized, skipping feedback tracking");
    return;
  }

  try {
    langfuseWeb.score({
      name: "user_feedback_browser",
      value,
      sessionId,
      comment: `message_id: ${messageId}`,
    });
  } catch (error) {
    // Silent failure - don't block user interactions
    console.error("Failed to track feedback in Langfuse:", error);
  }
}
