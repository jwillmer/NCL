"use client";

/**
 * AgentChat - Custom chat UI component using AG-UI SDK directly.
 *
 * Replaces CopilotChat and CopilotKit wrapper with a single component.
 * Provides the same functionality: message list, input, streaming, and context.
 */

import { createContext, useContext, useRef, useEffect, useState, FormEvent, KeyboardEvent } from "react";
import { Message } from "@ag-ui/client";
import { useAgentChat, UseAgentChatReturn, ConnectionStatus } from "@/hooks/useAgentChat";
import { RAGState } from "@/types/rag";
import { Send, Loader2, AlertCircle, WifiOff, Ship } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";
import {
  sourceTagRenderers,
  MessageCitationProvider,
  SourcesAccordion,
  useCitationContext,
} from "./Sources";

// Extended Message type with vessel_id metadata
export type ExtendedMessage = Message & {
  vessel_id?: string | null;
};

/**
 * Extract text content from a Message's content field.
 * Handles string, array of content parts, or record types.
 */
function getMessageText(content: Message["content"]): string {
  if (typeof content === "string") {
    return content;
  }
  if (Array.isArray(content)) {
    return content
      .filter((part): part is { type: "text"; text: string } => part.type === "text")
      .map((part) => part.text)
      .join("");
  }
  // Record type - try to extract text
  if (content && typeof content === "object") {
    const textContent = (content as Record<string, unknown>)["text"];
    if (typeof textContent === "string") {
      return textContent;
    }
  }
  return "";
}

// Vessel lookup map for displaying vessel names
export type VesselLookup = Record<string, string>; // vessel_id -> vessel_name

// Context for sharing agent chat state with child components
interface AgentChatContextType extends UseAgentChatReturn {
  threadId: string;
  disabled: boolean;
}

const AgentChatContext = createContext<AgentChatContextType | null>(null);

export function useAgentChatContext() {
  const context = useContext(AgentChatContext);
  if (!context) {
    throw new Error("useAgentChatContext must be used within AgentChat");
  }
  return context;
}

// Labels for the chat UI
interface ChatLabels {
  title?: string;
  initial?: string;
  placeholder?: string;
}

interface AgentChatProps {
  agentUrl: string;
  threadId: string;
  authToken: string;
  vesselId?: string | null;
  labels?: ChatLabels;
  disabled?: boolean;
  className?: string;
  initialMessages?: ExtendedMessage[];
  onMessagesChange?: (messages: Message[]) => void;
  renderAssistantMessage?: (props: AssistantMessageRenderProps) => React.ReactNode;
  vesselLookup?: VesselLookup;
  children?: React.ReactNode;
}

/**
 * Vessel badge component to show which vessel filter was active for a message.
 */
function VesselBadge({ vesselId, vesselLookup }: { vesselId?: string | null; vesselLookup?: VesselLookup }) {
  if (!vesselId) return null;

  const vesselName = vesselLookup?.[vesselId];
  const displayText = vesselName || `Vessel ${vesselId.slice(0, 8)}...`;

  return (
    <div className="flex justify-end mt-1">
      <span
        className="inline-flex items-center gap-1 text-[10px] uppercase tracking-wide text-white/60 hover:text-white/90 transition-colors cursor-default"
        title={vesselName ? `Filtered to: ${vesselName}` : `Vessel ID: ${vesselId}`}
      >
        <Ship className="h-3 w-3" />
        <span className="truncate max-w-[150px]">{displayText}</span>
      </span>
    </div>
  );
}

export interface AssistantMessageRenderProps {
  message: Message;
  isStreaming: boolean;
  isLastMessage: boolean;
  streamingContent?: string;
}

/**
 * Transform raw [C:chunk_id] citation markers to <cite> tags.
 */
function transformRawCitations(content: string): string {
  const citationPattern = /\[C:([a-f0-9]+)\]/gi;
  let index = 1;
  const indexMap = new Map<string, number>();

  return content.replace(citationPattern, (_, chunkId) => {
    const lowerChunkId = chunkId.toLowerCase();
    if (!indexMap.has(lowerChunkId)) {
      indexMap.set(lowerChunkId, index++);
    }
    return `<cite id="${lowerChunkId}">${indexMap.get(lowerChunkId)}</cite>`;
  });
}

/**
 * Default assistant message renderer with markdown and citations.
 */
function DefaultAssistantMessage({ message, isStreaming, streamingContent }: AssistantMessageRenderProps) {
  const { onViewCitation } = useCitationContext();
  const content = isStreaming && streamingContent ? streamingContent : getMessageText(message.content);

  return (
    <MessageCitationProvider onViewCitation={onViewCitation}>
      <div>
        <div className="prose prose-sm max-w-none prose-p:my-2 prose-headings:my-3 [&_ul]:list-disc [&_ul]:pl-5 [&_ol]:list-decimal [&_ol]:pl-5">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            rehypePlugins={[rehypeRaw as any]}
            components={sourceTagRenderers}
          >
            {transformRawCitations(content)}
          </ReactMarkdown>
        </div>
        {isStreaming && (
          <div className="mt-2 flex items-center gap-2">
            <Loader2 className="h-4 w-4 animate-spin text-gray-400" />
          </div>
        )}
        {!isStreaming && content && (
          <div className="mt-3 mb-2">
            <SourcesAccordion />
          </div>
        )}
      </div>
    </MessageCitationProvider>
  );
}

/**
 * Connection status indicator.
 */
function ConnectionStatusIndicator({ status }: { status: ConnectionStatus }) {
  if (status === "connected") return null;

  return (
    <div className={`flex items-center gap-2 px-4 py-2 text-sm ${
      status === "reconnecting" ? "bg-yellow-50 text-yellow-700" : "bg-red-50 text-red-700"
    }`}>
      {status === "reconnecting" ? (
        <>
          <Loader2 className="h-4 w-4 animate-spin" />
          <span>Reconnecting...</span>
        </>
      ) : (
        <>
          <WifiOff className="h-4 w-4" />
          <span>Connection lost. Please refresh the page.</span>
        </>
      )}
    </div>
  );
}

/**
 * Wrapper component to properly render assistant message with hooks.
 * This ensures hooks are called consistently regardless of conditional rendering.
 */
function RenderAssistantMessage({
  message,
  isStreaming,
  isLastMessage,
  streamingContent,
  Component,
}: AssistantMessageRenderProps & {
  Component: (props: AssistantMessageRenderProps) => React.ReactNode;
}) {
  return <>{Component({ message, isStreaming, isLastMessage, streamingContent })}</>;
}

/**
 * Message list component with auto-scroll.
 */
function MessageList({
  messages,
  isStreaming,
  streamingContent,
  streamingMessageId,
  renderAssistantMessage,
  initialMessage,
  vesselLookup,
}: {
  messages: ExtendedMessage[];
  isStreaming: boolean;
  streamingContent: string;
  streamingMessageId: string | null;
  renderAssistantMessage: (props: AssistantMessageRenderProps) => React.ReactNode;
  initialMessage?: string;
  vesselLookup?: VesselLookup;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom on new messages or streaming content
  useEffect(() => {
    const scrollToBottom = () => {
      if (scrollRef.current) {
        scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
      }
    };

    // Immediate scroll
    scrollToBottom();

    // Delayed scroll to handle layout shifts (e.g. images loading or markdown rendering)
    const timeoutId = setTimeout(scrollToBottom, 50);
    
    // Also use RAF for good measure
    const rafId = requestAnimationFrame(scrollToBottom);

    return () => {
      clearTimeout(timeoutId);
      cancelAnimationFrame(rafId);
    };
  }, [messages.length, streamingContent, isStreaming]);

  // Filter to only show user and assistant messages (exclude tool calls, system messages)
  // Also filter out assistant messages that are raw tool results (JSON context dumps) or empty
  const visibleMessages = messages.filter((m) => {
    if (m.role !== "user" && m.role !== "assistant") return false;
    if (m.role === "assistant" && typeof m.content === "string") {
      const content = m.content.trim();
      // Filter out empty messages
      if (!content) return false;
      // Filter out raw tool result messages (JSON with context field)
      if (content.startsWith('{"context":') || content.startsWith('{"available_chunk_ids":')) {
        return false;
      }
    }
    return true;
  });
  const showInitialMessage = visibleMessages.length === 0 && !isStreaming && initialMessage;

  return (
    <div ref={scrollRef} className="flex-1 overflow-y-auto min-h-0 p-4 pb-24 space-y-6 scroll-smooth">
      {showInitialMessage && (
        <div className="flex gap-4 max-w-3xl mx-auto">
          <div className="flex-shrink-0 w-8 h-8 rounded-full bg-MTSS-blue flex items-center justify-center shadow-sm">
            <span className="text-white text-xs font-semibold">AI</span>
          </div>
          <div className="flex-1 bg-white border border-gray-100 shadow-sm rounded-2xl rounded-tl-none p-5">
            <p className="text-gray-600 leading-relaxed">{initialMessage}</p>
          </div>
        </div>
      )}

      {visibleMessages.map((message, index) => {
        const isLastMessage = index === visibleMessages.length - 1;
        const isAssistant = message.role === "assistant";
        const isCurrentStreaming = isStreaming && streamingMessageId === message.id;

        return (
          <div key={message.id} className={`flex gap-4 max-w-3xl mx-auto ${isAssistant ? "" : "flex-row-reverse"}`}>
            <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center shadow-sm ${
              isAssistant ? "bg-MTSS-blue" : "bg-gray-700"
            }`}>
              <span className="text-white text-xs font-semibold">
                {isAssistant ? "AI" : "U"}
              </span>
            </div>
            <div className={`flex-1 max-w-[85%] rounded-2xl p-5 shadow-sm ${
              isAssistant 
                ? "bg-white border border-gray-100 rounded-tl-none" 
                : "bg-MTSS-blue text-white rounded-tr-none ml-auto"
            }`}>
              {isAssistant ? (
                <RenderAssistantMessage
                  message={message}
                  isStreaming={isCurrentStreaming}
                  isLastMessage={isLastMessage}
                  streamingContent={isCurrentStreaming ? streamingContent : undefined}
                  Component={renderAssistantMessage}
                />
              ) : (
                <div>
                  <p className="whitespace-pre-wrap leading-relaxed">{getMessageText(message.content)}</p>
                  {message.vessel_id && (
                    <VesselBadge vesselId={message.vessel_id} vesselLookup={vesselLookup} />
                  )}
                </div>
              )}
            </div>
          </div>
        );
      })}

      {/* Show streaming message as content arrives - displays chunks in real-time */}
      {isStreaming && (
        <div className="flex gap-4 max-w-3xl mx-auto">
          <div className="flex-shrink-0 w-8 h-8 rounded-full bg-MTSS-blue flex items-center justify-center shadow-sm">
            <span className="text-white text-xs font-semibold">AI</span>
          </div>
          <div className="flex-1 max-w-[85%] bg-white border border-gray-100 shadow-sm rounded-2xl rounded-tl-none p-5">
            <RenderAssistantMessage
              message={{ id: streamingMessageId || "streaming", role: "assistant", content: streamingContent }}
              isStreaming={true}
              isLastMessage={true}
              streamingContent={streamingContent}
              Component={renderAssistantMessage}
            />
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * Component to render animated trailing dots
 */
function AnimatedDots() {
  const [dots, setDots] = useState("");

  useEffect(() => {
    const interval = setInterval(() => {
      setDots(prev => prev.length >= 3 ? "" : prev + ".");
    }, 500);
    return () => clearInterval(interval);
  }, []);

  return <span className="inline-block w-6">{dots}</span>;
}

/**
 * Chat input component with integrated progress indicator.
 */
function ChatInput({
  onSend,
  disabled,
  isLoading,
  placeholder,
  progressMessage,
}: {
  onSend: (content: string) => void;
  disabled: boolean;
  isLoading: boolean;
  placeholder?: string;
  progressMessage?: string;
}) {
  const [input, setInput] = useState("");

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (input.trim() && !disabled && !isLoading) {
      onSend(input.trim());
      setInput("");
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  // Show progress message whenever we have one (backend is working)
  const showProgress = !!progressMessage;

  return (
    <form onSubmit={handleSubmit} className="p-4 bg-white/80 backdrop-blur-md border-t border-gray-200">
      <div className="max-w-3xl mx-auto flex gap-3 relative">
        {showProgress ? (
          // Modern progress indicator - text only with animated dots
          <div className="flex-1 flex items-center px-4 py-3 bg-gray-50/50 border border-gray-200 rounded-xl">
            <span className="text-sm font-medium text-MTSS-blue animate-pulse">
              {progressMessage}
            </span>
            <span className="text-sm font-medium text-MTSS-blue ml-0.5">
              <AnimatedDots />
            </span>
          </div>
        ) : (
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={placeholder || "Type a message..."}
            disabled={disabled || isLoading}
            className={`flex-1 resize-none rounded-xl border border-gray-200 px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-MTSS-blue/20 focus:border-MTSS-blue transition-all shadow-sm ${
              disabled || isLoading ? "opacity-50 cursor-not-allowed bg-gray-50" : "bg-white"
            }`}
            rows={1}
            style={{ minHeight: "46px" }}
          />
        )}
        <button
          type="submit"
          disabled={disabled || isLoading || !input.trim()}
          className={`px-4 rounded-xl bg-MTSS-blue text-white transition-all shadow-sm hover:shadow active:scale-95 ${
            disabled || isLoading || !input.trim()
              ? "opacity-50 cursor-not-allowed"
              : "hover:bg-MTSS-blue-dark"
          }`}
        >
          {isLoading ? (
            <Loader2 className="h-5 w-5 animate-spin" />
          ) : (
            <Send className="h-5 w-5" />
          )}
        </button>
      </div>
    </form>
  );
}

/**
 * Main AgentChat component.
 */
export function AgentChat({
  agentUrl,
  threadId,
  authToken,
  vesselId,
  labels = {},
  disabled = false,
  className = "",
  initialMessages,
  onMessagesChange,
  renderAssistantMessage,
  vesselLookup,
  children,
}: AgentChatProps) {
  const chat = useAgentChat({
    agentUrl,
    threadId,
    authToken,
    vesselId,
    initialMessages,
  });

  // Notify parent of message changes
  useEffect(() => {
    if (onMessagesChange) {
      onMessagesChange(chat.messages);
    }
  }, [chat.messages, onMessagesChange]);

  const contextValue: AgentChatContextType = {
    ...chat,
    threadId,
    disabled,
  };

  const assistantMessageRenderer = renderAssistantMessage || DefaultAssistantMessage;

  // Get progress message from state
  const progressMessage = chat.state?.search_progress || undefined;

  return (
    <AgentChatContext.Provider value={contextValue}>
      <div className={`flex flex-col overflow-hidden ${className}`}>
        {/* Connection status */}
        <ConnectionStatusIndicator status={chat.connectionStatus} />

        {/* Error display */}
        {chat.error && (
          <div className="flex-shrink-0 flex items-center gap-2 px-4 py-2 bg-red-50 border-b border-red-200">
            <AlertCircle className="h-4 w-4 text-red-500" />
            <span className="text-sm text-red-700">{chat.error}</span>
          </div>
        )}

        {/* Main content area with messages - scrollable, takes all available space */}
        <div className="relative flex-1 min-h-0 overflow-hidden flex flex-col">
          {children}

          {/* Message list - scrollable */}
          <MessageList
            messages={chat.messages as ExtendedMessage[]}
            isStreaming={chat.isStreaming}
            streamingContent={chat.streamingContent}
            streamingMessageId={chat.streamingMessageId}
            renderAssistantMessage={assistantMessageRenderer}
            initialMessage={labels.initial}
            vesselLookup={vesselLookup}
          />
        </div>

        {/* Input - always visible at bottom */}
        <div className="flex-shrink-0 border-t border-gray-200 bg-white fixed bottom-0 left-0 right-0 z-10">
          <ChatInput
            onSend={chat.sendMessage}
            disabled={disabled}
            isLoading={chat.isLoading}
            placeholder={labels.placeholder}
            progressMessage={progressMessage}
          />
        </div>
      </div>
    </AgentChatContext.Provider>
  );
}
