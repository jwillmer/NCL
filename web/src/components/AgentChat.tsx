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
import { Send, Loader2, AlertCircle, WifiOff } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";
import {
  sourceTagRenderers,
  MessageCitationProvider,
  SourcesAccordion,
  useCitationContext,
} from "./Sources";

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
  initialMessages?: Message[];
  onMessagesChange?: (messages: Message[]) => void;
  renderAssistantMessage?: (props: AssistantMessageRenderProps) => React.ReactNode;
  children?: React.ReactNode;
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
  const content = isStreaming && streamingContent ? streamingContent : (message.content || "");

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
 * Message list component with auto-scroll.
 */
function MessageList({
  messages,
  isStreaming,
  streamingContent,
  streamingMessageId,
  renderAssistantMessage,
  initialMessage,
}: {
  messages: Message[];
  isStreaming: boolean;
  streamingContent: string;
  streamingMessageId: string | null;
  renderAssistantMessage: (props: AssistantMessageRenderProps) => React.ReactNode;
  initialMessage?: string;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom on new messages or streaming content
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages.length, streamingContent]);

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
    <div ref={scrollRef} className="flex-1 overflow-y-auto min-h-0 p-4 space-y-4">
      {showInitialMessage && (
        <div className="flex gap-3">
          <div className="flex-shrink-0 w-8 h-8 rounded-full bg-ncl-blue flex items-center justify-center">
            <span className="text-white text-sm font-medium">AI</span>
          </div>
          <div className="flex-1 bg-gray-50 rounded-lg p-4">
            <p className="text-gray-700">{initialMessage}</p>
          </div>
        </div>
      )}

      {visibleMessages.map((message, index) => {
        const isLastMessage = index === visibleMessages.length - 1;
        const isAssistant = message.role === "assistant";
        const isCurrentStreaming = isStreaming && streamingMessageId === message.id;

        return (
          <div key={message.id} className={`flex gap-3 ${isAssistant ? "" : "flex-row-reverse"}`}>
            <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
              isAssistant ? "bg-ncl-blue" : "bg-gray-600"
            }`}>
              <span className="text-white text-sm font-medium">
                {isAssistant ? "AI" : "U"}
              </span>
            </div>
            <div className={`flex-1 max-w-[80%] rounded-lg p-4 ${
              isAssistant ? "bg-gray-50" : "bg-ncl-blue text-white ml-auto"
            }`}>
              {isAssistant ? (
                renderAssistantMessage({
                  message,
                  isStreaming: isCurrentStreaming,
                  isLastMessage,
                  streamingContent: isCurrentStreaming ? streamingContent : undefined,
                })
              ) : (
                <p className="whitespace-pre-wrap">{message.content}</p>
              )}
            </div>
          </div>
        );
      })}

      {/* Show streaming message as content arrives - displays chunks in real-time */}
      {isStreaming && (
        <div className="flex gap-3">
          <div className="flex-shrink-0 w-8 h-8 rounded-full bg-ncl-blue flex items-center justify-center">
            <span className="text-white text-sm font-medium">AI</span>
          </div>
          <div className="flex-1 max-w-[80%] bg-gray-50 rounded-lg p-4">
            {renderAssistantMessage({
              message: { id: streamingMessageId || "streaming", role: "assistant", content: streamingContent },
              isStreaming: true,
              isLastMessage: true,
              streamingContent,
            })}
          </div>
        </div>
      )}
    </div>
  );
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
    <form onSubmit={handleSubmit} className="p-4">
      <div className="flex gap-2">
        {showProgress ? (
          // Progress indicator replaces input when loading
          <div className="flex-1 flex items-center gap-2 px-4 py-2 bg-blue-50 border border-blue-200 rounded-lg">
            <div className="animate-spin h-4 w-4 border-2 border-blue-500 border-t-transparent rounded-full" />
            <span className="text-sm text-blue-700">{progressMessage}</span>
          </div>
        ) : (
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={placeholder || "Type a message..."}
            disabled={disabled || isLoading}
            className={`flex-1 resize-none rounded-lg border border-gray-300 px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ncl-blue focus:border-transparent ${
              disabled || isLoading ? "opacity-50 cursor-not-allowed bg-gray-50" : ""
            }`}
            rows={1}
          />
        )}
        <button
          type="submit"
          disabled={disabled || isLoading || !input.trim()}
          className={`px-4 py-2 rounded-lg bg-ncl-blue text-white transition-colors ${
            disabled || isLoading || !input.trim()
              ? "opacity-50 cursor-not-allowed"
              : "hover:bg-ncl-blue-dark"
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
        <div className="relative flex-1 min-h-0 overflow-hidden">
          {children}

          {/* Message list - scrollable */}
          <MessageList
            messages={chat.messages}
            isStreaming={chat.isStreaming}
            streamingContent={chat.streamingContent}
            streamingMessageId={chat.streamingMessageId}
            renderAssistantMessage={assistantMessageRenderer}
            initialMessage={labels.initial}
          />
        </div>

        {/* Input - always visible at bottom */}
        <div className="flex-shrink-0 border-t border-gray-200 bg-white">
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
