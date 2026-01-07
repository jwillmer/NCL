/**
 * Custom React hook for AG-UI agent communication.
 *
 * Replaces CopilotKit's useCoAgent, useCoAgentStateRender, and useCopilotChatInternal
 * with direct AG-UI SDK usage via HttpAgent.
 *
 * State synchronization:
 * - Uses onStateChanged callback which receives the already-merged state
 * - Backend emits state via copilotkit_emit_state() which sends STATE_SNAPSHOT events
 * - The AG-UI SDK automatically merges state and calls onStateChanged
 */

import { useState, useCallback, useRef, useEffect } from "react";
import { HttpAgent, Message, AgentSubscriber } from "@ag-ui/client";
import { RAGState, initialRAGState } from "@/types/rag";

const DEBUG = process.env.NODE_ENV === "development";

export type ConnectionStatus = "connected" | "disconnected" | "reconnecting";

export interface UseAgentChatOptions {
  agentUrl: string;
  threadId: string;
  authToken: string;
  vesselId?: string | null;
  initialState?: RAGState;
  initialMessages?: Message[];
}

export interface UseAgentChatReturn {
  messages: Message[];
  setMessages: (messages: Message[]) => void;
  state: RAGState;
  setState: (state: Partial<RAGState>) => void;
  isLoading: boolean;
  isStreaming: boolean;
  streamingContent: string;
  streamingMessageId: string | null;
  error: string | null;
  connectionStatus: ConnectionStatus;
  sendMessage: (content: string) => Promise<void>;
  abortRun: () => void;
}

const MAX_RETRY_ATTEMPTS = 3;
const RETRY_DELAY_BASE = 1000; // 1 second base delay for exponential backoff

/**
 * Hook for managing AG-UI agent communication with React state.
 */
export function useAgentChat(options: UseAgentChatOptions): UseAgentChatReturn {
  const { agentUrl, threadId, authToken, vesselId, initialState, initialMessages } = options;

  // Core state
  const [messages, setMessages] = useState<Message[]>(initialMessages || []);
  const [state, setStateInternal] = useState<RAGState>(initialState || initialRAGState);
  const [isLoading, setIsLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingContent, setStreamingContent] = useState("");
  const [streamingMessageId, setStreamingMessageId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>("connected");

  // Refs for agent instance and retry logic
  const agentRef = useRef<HttpAgent | null>(null);
  const retryCountRef = useRef(0);

  // Initialize agent on mount or when config changes
  useEffect(() => {
    if (DEBUG) console.log("[AG-UI] Initializing HttpAgent for thread:", threadId);

    const agent = new HttpAgent({
      url: agentUrl,
      threadId,
      headers: {
        Authorization: `Bearer ${authToken}`,
      },
      initialMessages: initialMessages || [],
      initialState: initialState || initialRAGState,
      debug: DEBUG, // Enable AG-UI SDK debug logging
    });

    agentRef.current = agent;
    setConnectionStatus("connected");

    return () => {
      if (DEBUG) console.log("[AG-UI] Cleaning up HttpAgent for thread:", threadId);
      agent.abortRun();
      agentRef.current = null;
    };
  }, [agentUrl, threadId, authToken]);

  // Update agent state when initialMessages change (for history loading)
  useEffect(() => {
    if (agentRef.current && initialMessages && initialMessages.length > 0) {
      agentRef.current.setMessages(initialMessages);
      setMessages(initialMessages);
    }
  }, [initialMessages]);

  // Sync vesselId to state when it changes
  useEffect(() => {
    if (vesselId !== undefined) {
      setStateInternal((prev) => ({ ...prev, selected_vessel_id: vesselId }));
    }
  }, [vesselId]);

  // Partial state update function
  const setState = useCallback((partialState: Partial<RAGState>) => {
    setStateInternal((prev) => ({ ...prev, ...partialState }));
  }, []);

  // Send a message to the agent
  const sendMessage = useCallback(
    async (content: string) => {
      const agent = agentRef.current;
      if (!agent) {
        setError("Agent not initialized");
        return;
      }

      // Reset error and set loading state
      setError(null);
      setIsLoading(true);
      setStreamingContent("");
      setStreamingMessageId(null);

      // Add user message immediately for optimistic UI
      const userMessageId = `user-${Date.now()}`;
      const userMessage: Message = {
        id: userMessageId,
        role: "user",
        content,
      };

      setMessages((prev) => [...prev, userMessage]);
      agent.addMessage(userMessage);

      // Create subscriber to handle events
      const subscriber: AgentSubscriber = {
        onRunStartedEvent: ({ event }) => {
          if (DEBUG) console.log("[AG-UI] Run started:", event.runId);
          setIsLoading(true);
          setConnectionStatus("connected");
          retryCountRef.current = 0;
        },

        onTextMessageStartEvent: ({ event }) => {
          if (DEBUG) console.log("[AG-UI] Text message start:", event.messageId);
          setIsStreaming(true);
          setStreamingMessageId(event.messageId);
          setStreamingContent("");
        },

        onTextMessageContentEvent: ({ textMessageBuffer }) => {
          setStreamingContent(textMessageBuffer);
        },

        onTextMessageEndEvent: ({ textMessageBuffer, event }) => {
          if (DEBUG) console.log("[AG-UI] Text message end:", event.messageId);
          setIsStreaming(false);
          // Add the completed assistant message only if not already present
          // (prevents duplicates when backend also sends messagesSnapshot)
          setMessages((prev) => {
            const exists = prev.some((m) => m.id === event.messageId);
            if (exists) {
              if (DEBUG) console.log("[AG-UI] Message already exists, skipping:", event.messageId);
              return prev;
            }
            return [...prev, {
              id: event.messageId,
              role: "assistant" as const,
              content: textMessageBuffer,
            }];
          });
          setStreamingContent("");
          setStreamingMessageId(null);
        },

        // Use onStateChanged for simpler state handling - AG-UI merges state automatically
        onStateChanged: ({ state: newState }) => {
          if (DEBUG) console.log("[AG-UI] State changed:", newState);
          if (newState) {
            const ragState = newState as RAGState;
            if (DEBUG && ragState.search_progress) {
              console.log("[AG-UI] Progress update:", ragState.search_progress);
            }
            setStateInternal(ragState);
          }
        },

        // Also handle explicit state events for completeness
        onStateSnapshotEvent: ({ event }) => {
          if (DEBUG) console.log("[AG-UI] State snapshot:", event.snapshot);
          const snapshot = event.snapshot as RAGState;
          if (snapshot) {
            setStateInternal(snapshot);
          }
        },

        onStateDeltaEvent: ({ event, state: newState }) => {
          if (DEBUG) console.log("[AG-UI] State delta:", event.delta, "merged:", newState);
          if (newState) {
            setStateInternal(newState as RAGState);
          }
        },

        // Handle messages snapshot for conversation history sync
        // This replaces all messages with the backend's authoritative version
        // (includes properly formatted citations from backend processing)
        onMessagesSnapshotEvent: ({ event }) => {
          if (DEBUG) console.log("[AG-UI] Messages snapshot:", event.messages?.length, "messages");
          if (event.messages && event.messages.length > 0) {
            // Filter out messages with only user's optimistically added message ID
            // Backend messages are authoritative and have processed citations
            setMessages(event.messages);
          }
        },

        onRunFinishedEvent: ({ event }) => {
          if (DEBUG) console.log("[AG-UI] Run finished:", event.runId);
          setIsLoading(false);
          setIsStreaming(false);
        },

        onRunErrorEvent: ({ event }) => {
          if (DEBUG) console.log("[AG-UI] Run error:", event.message);
          setError(event.message || "An error occurred");
          setIsLoading(false);
          setIsStreaming(false);
        },

        onRunFailed: async ({ error: err }) => {
          if (DEBUG) console.log("[AG-UI] Run failed:", err.message);
          // Handle connection failures with retry logic
          if (retryCountRef.current < MAX_RETRY_ATTEMPTS) {
            retryCountRef.current++;
            setConnectionStatus("reconnecting");
            const delay = RETRY_DELAY_BASE * Math.pow(2, retryCountRef.current - 1);
            await new Promise((resolve) => setTimeout(resolve, delay));
            // Retry will be handled by the outer try/catch
            throw err;
          }
          setError(err.message || "Connection failed after multiple attempts");
          setConnectionStatus("disconnected");
          setIsLoading(false);
          setIsStreaming(false);
        },

        // Debug: Log all events in development
        onEvent: DEBUG ? ({ event }) => {
          console.log("[AG-UI] Event:", event.type, event);
        } : undefined,
      };

      // Run the agent with retry logic
      const attemptRun = async (): Promise<void> => {
        try {
          await agent.runAgent(
            {
              forwardedProps: {
                state: {
                  selected_vessel_id: state.selected_vessel_id,
                },
              },
            },
            subscriber
          );
        } catch (err) {
          // If we haven't exceeded retries and it's a connection error, retry
          if (retryCountRef.current < MAX_RETRY_ATTEMPTS) {
            retryCountRef.current++;
            setConnectionStatus("reconnecting");
            const delay = RETRY_DELAY_BASE * Math.pow(2, retryCountRef.current - 1);
            await new Promise((resolve) => setTimeout(resolve, delay));
            return attemptRun();
          }
          // Max retries exceeded
          const errorMessage = err instanceof Error ? err.message : "Unknown error";
          setError(errorMessage);
          setConnectionStatus("disconnected");
          setIsLoading(false);
          setIsStreaming(false);
        }
      };

      await attemptRun();
    },
    [state.selected_vessel_id]
  );

  // Abort the current run
  const abortRun = useCallback(() => {
    const agent = agentRef.current;
    if (agent) {
      agent.abortRun();
      setIsLoading(false);
      setIsStreaming(false);
    }
  }, []);

  return {
    messages,
    setMessages,
    state,
    setState,
    isLoading,
    isStreaming,
    streamingContent,
    streamingMessageId,
    error,
    connectionStatus,
    sendMessage,
    abortRun,
  };
}
