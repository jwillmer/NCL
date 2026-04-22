/**
 * Chat page — merged chat UI using Vercel AI SDK's useChat.
 * Combines former AgentChat, ChatContainer, and chat/page into one file.
 */

import { useState, useEffect, useCallback, useRef, useMemo, type FormEvent, type KeyboardEvent } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { useChat, type UIMessage } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";
import rehypeSanitize, { defaultSchema } from "rehype-sanitize";
import { ArrowLeft, Archive, Send, Loader2, AlertCircle, Ship, ChevronDown, ThumbsUp, ThumbsDown } from "lucide-react";
import * as DropdownMenu from "@radix-ui/react-dropdown-menu";

import { useAuth, LoginForm } from "@/components/auth";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import {
  CitationProvider,
  MessageCitationProvider,
  SourcesAccordion,
  SourceViewDialog,
  sourceTagRenderers,
  useCitationContext,
} from "@/components/Sources";
import { Button } from "@/components/ui";
import { cn, transformRawCitations } from "@/lib/utils";
import { initLangfuse, trackFeedback } from "@/lib/langfuse";
import {
  Conversation,
  Vessel,
  getConversation,
  createConversation,
  updateConversation,
  getMessages,
  touchConversation,
  generateTitle,
  submitFeedback,
  listVessels,
  listVesselTypes,
  listVesselClasses,
  getApiBaseUrl,
  getAuthHeaders,
  ConversationApiError,
} from "@/lib/conversations";

// Initialize Langfuse on load
if (typeof window !== "undefined") {
  initLangfuse();
}

// Allow <cite>, <img-cite>, and <mark> through sanitizer.
// clobberPrefix is disabled so <cite id="..."> round-trips the backend's
// 12-char hex chunk_id to CiteRenderer / SourceViewDialog untouched.
// rehype-sanitize's default prepends "user-content-" to every id, which
// made /api/citations/user-content-<hex> 400 on the backend.
const sanitizeSchema = {
  ...defaultSchema,
  clobberPrefix: "",
  tagNames: [...(defaultSchema.tagNames || []), "cite", "img-cite", "mark"],
  attributes: {
    ...defaultSchema.attributes,
    cite: ["id", "title", "page", "lines", "download"],
    "img-cite": ["src", "id"],
    mark: ["className", "class"],
  },
};

// ============================================
// Filter Dropdown
// ============================================

function FilterDropdown({
  label, value, options, loading, onChange, disabled, searchable,
}: {
  label: string;
  value: string | null;
  options: { id: string | null; name: string }[];
  loading?: boolean;
  onChange: (value: string | null) => void;
  disabled?: boolean;
  searchable?: boolean;
}) {
  const [search, setSearch] = useState("");
  const filtered = searchable && search
    ? options.filter((v) => v.name.toLowerCase().includes(search.toLowerCase()))
    : options;
  const selected = options.find((v) => v.id === value) || options[0];
  const hasSelection = value !== null;

  return (
    <DropdownMenu.Root onOpenChange={(open) => !open && setSearch("")}>
      <DropdownMenu.Trigger asChild>
        <button
          className={cn(
            "flex items-center gap-2 px-3 py-1.5 text-sm rounded-lg border border-gray-200 bg-white hover:bg-gray-50 transition-colors",
            disabled && "opacity-50 cursor-not-allowed",
            hasSelection && "border-blue-300 bg-blue-50"
          )}
          disabled={disabled || loading}
        >
          <Ship className="h-4 w-4 text-gray-500" />
          <span className="max-w-[250px] truncate">
            {loading ? "Loading..." : hasSelection
              ? <><span className="font-medium text-gray-600">{label}:</span> {selected.name}</>
              : selected.name}
          </span>
          <ChevronDown className="h-3 w-3 opacity-50" />
        </button>
      </DropdownMenu.Trigger>
      <DropdownMenu.Portal>
        <DropdownMenu.Content className="min-w-[180px] bg-white rounded-md shadow-lg border border-gray-200 z-50" align="start">
          {searchable && (
            <div className="p-2 border-b border-gray-200">
              <input
                type="text"
                placeholder={`Search ${label.toLowerCase()}...`}
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="w-full px-2 py-1 text-sm border border-gray-200 rounded outline-none focus:border-blue-500"
                onClick={(e) => e.stopPropagation()}
                onKeyDown={(e) => e.stopPropagation()}
              />
            </div>
          )}
          <div className="max-h-[250px] overflow-y-auto p-1">
            {filtered.length > 0 ? filtered.map((opt) => (
              <DropdownMenu.Item
                key={opt.id || "all"}
                className={cn(
                  "flex items-center gap-2 px-3 py-2 text-sm cursor-pointer rounded outline-none",
                  opt.id === value ? "bg-blue-50 text-blue-600" : "hover:bg-gray-100"
                )}
                onSelect={() => onChange(opt.id)}
              >
                {opt.name}
              </DropdownMenu.Item>
            )) : (
              <div className="px-3 py-2 text-sm text-gray-500">No results found</div>
            )}
          </div>
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  );
}

// ============================================
// Assistant Message (local subcomponent)
// ============================================

function AssistantMessage({ content, messageId, threadId, showFeedback }: {
  content: string;
  messageId: string;
  threadId: string;
  showFeedback: boolean;
}) {
  const { onViewCitation } = useCitationContext();
  const [feedback, setFeedback] = useState<"up" | "down" | null>(null);

  const handleFeedback = (value: 0 | 1) => {
    if (feedback) return;
    setFeedback(value === 1 ? "up" : "down");
    submitFeedback(threadId, messageId, value).catch(console.error);
    trackFeedback(threadId, messageId, value);
  };

  return (
    <MessageCitationProvider onViewCitation={onViewCitation}>
      <div>
        <div className="prose prose-sm max-w-none prose-p:my-2 prose-headings:my-3 [&_ul]:list-disc [&_ul]:pl-5 [&_ol]:list-decimal [&_ol]:pl-5">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            rehypePlugins={[rehypeRaw as any, [rehypeSanitize, sanitizeSchema]]}
            components={sourceTagRenderers}
          >
            {transformRawCitations(content)}
          </ReactMarkdown>
        </div>
        {showFeedback && content && (
          <>
            <div className="mt-3 mb-2"><SourcesAccordion /></div>
            <div className="flex items-center gap-1 mt-2">
              <button
                onClick={() => handleFeedback(1)}
                className={cn("p-1.5 rounded hover:bg-gray-100 transition-colors",
                  feedback === "up" ? "text-green-600 bg-green-50" : "text-gray-400 hover:text-gray-600")}
                disabled={feedback !== null}
              ><ThumbsUp className="h-4 w-4" /></button>
              <button
                onClick={() => handleFeedback(0)}
                className={cn("p-1.5 rounded hover:bg-gray-100 transition-colors",
                  feedback === "down" ? "text-red-600 bg-red-50" : "text-gray-400 hover:text-gray-600")}
                disabled={feedback !== null}
              ><ThumbsDown className="h-4 w-4" /></button>
            </div>
          </>
        )}
      </div>
    </MessageCitationProvider>
  );
}

// ============================================
// ChatPage Content
// ============================================

function ChatPageContent() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const threadId = searchParams.get("threadId") || "";
  const { session, loading: authLoading } = useAuth();
  const userId = session?.user?.id;

  // Conversation state
  const [conversation, setConversation] = useState<Conversation | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Filter state
  const [vesselId, setVesselId] = useState<string | null>(null);
  const [vesselTypeId, setVesselTypeId] = useState<string | null>(null);
  const [vesselClassId, setVesselClassId] = useState<string | null>(null);
  const [vessels, setVessels] = useState<Vessel[]>([]);
  const [vesselTypes, setVesselTypes] = useState<string[]>([]);
  const [vesselClasses, setVesselClasses] = useState<string[]>([]);
  const [vesselsLoading, setVesselsLoading] = useState(true);

  // Citation dialog state
  const [selectedChunkId, setSelectedChunkId] = useState<string | null>(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [linesToHighlight, setLinesToHighlight] = useState<[number, number][] | undefined>();

  // History/title tracking
  const [historyLoaded, setHistoryLoaded] = useState(false);
  const prevMessageCount = useRef(0);
  const titleGenerated = useRef(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Vessel lookup for badges
  const vesselLookup = useMemo(() => {
    const map: Record<string, string> = {};
    for (const v of vessels) map[v.id] = v.name;
    return map;
  }, [vessels]);

  const transport = useMemo(
    () => new DefaultChatTransport({
      api: `${getApiBaseUrl()}/agent`,
      headers: async () => getAuthHeaders(),
      body: () => ({
        thread_id: threadId,
        filters: { vessel_id: vesselId, vessel_type: vesselTypeId, vessel_class: vesselClassId },
      }),
    }),
    [threadId, vesselId, vesselTypeId, vesselClassId]
  );

  const { messages, sendMessage, status, setMessages, error: chatError } = useChat({
    id: threadId,
    transport,
    onError: () => {
      // Reload from DB on error to resync
      getMessages(threadId)
        .then((history) => setMessages(history.map(toUIMessage)))
        .catch(() => {});
    },
  });

  const isStreaming = status === "streaming" || status === "submitted";

  // Load filters
  useEffect(() => {
    if (!userId) return;
    Promise.all([listVessels(), listVesselTypes(), listVesselClasses()])
      .then(([v, t, c]) => { setVessels(v); setVesselTypes(t); setVesselClasses(c); })
      .catch(console.error)
      .finally(() => setVesselsLoading(false));
  }, [userId]);

  // Load or create conversation
  useEffect(() => {
    if (!userId || !threadId) return;
    setLoading(true);
    setError(null);

    getConversation(threadId)
      .then((conv) => {
        setConversation(conv);
        setVesselId(conv.vessel_id);
        setVesselTypeId(conv.vessel_type);
        setVesselClassId(conv.vessel_class);
      })
      .catch(async (err) => {
        if (err instanceof ConversationApiError && err.status === 404) {
          try {
            setConversation(await createConversation({ thread_id: threadId }));
          } catch { setError("Failed to create conversation"); }
        } else {
          setError("Failed to load conversation");
        }
      })
      .finally(() => setLoading(false));
  }, [userId, threadId]);

  // Load message history
  useEffect(() => {
    if (!userId || !threadId || historyLoaded) return;
    getMessages(threadId)
      .then((history) => {
        if (history.length > 0) {
          setMessages(history.map(toUIMessage));
          titleGenerated.current = history.some((m) => m.role === "user");
        }
        prevMessageCount.current = history.length;
      })
      .catch(console.error)
      .finally(() => setHistoryLoaded(true));
  }, [userId, threadId, historyLoaded, setMessages]);

  // Track messages for title gen and timestamp update
  useEffect(() => {
    if (!threadId || !historyLoaded || messages.length <= prevMessageCount.current) return;
    touchConversation(threadId).catch(console.error);
    if (!titleGenerated.current) {
      const userMsg = messages.find((m) => m.role === "user");
      if (userMsg) {
        const text = userMsg.parts.filter((p): p is { type: "text"; text: string } => p.type === "text").map((p) => p.text).join("");
        if (text) {
          titleGenerated.current = true;
          generateTitle(threadId, text).catch(console.error);
        }
      }
    }
    prevMessageCount.current = messages.length;
  }, [messages, threadId, historyLoaded]);

  // Auto-scroll
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, isStreaming]);

  // Filter change handler (unified)
  const handleFilterChange = useCallback(async (key: "vessel_id" | "vessel_type" | "vessel_class", value: string | null) => {
    const setters = { vessel_id: setVesselId, vessel_type: setVesselTypeId, vessel_class: setVesselClassId };
    // Set selected, clear others (mutual exclusivity)
    for (const [k, setter] of Object.entries(setters)) setter(k === key ? value : null);
    if (conversation) {
      const update = { vessel_id: null as string | null, vessel_type: null as string | null, vessel_class: null as string | null };
      update[key] = value;
      updateConversation(threadId, update).catch(console.error);
    }
  }, [conversation, threadId]);

  // React to filter changes pushed by the agent mid-stream.
  // The server emits `2:["filter", {...}]` via emit_filter_update in set_filter_node;
  // the Vercel AI SDK exposes that as a `data-filter` part on the last assistant
  // message. We diff against current UI state to avoid re-triggering the DB
  // write loop when our own handleFilterChange caused the update.
  const lastAppliedAgentFilterRef = useRef<string>("");
  useEffect(() => {
    if (!messages.length) return;
    const last = messages[messages.length - 1];
    if (last.role !== "assistant") return;
    const parts = (last as unknown as { parts?: Array<{ type: string; data?: unknown }> }).parts;
    if (!parts) return;
    // Walk parts in order; the most recent filter update wins.
    type FilterPayload = { vessel_id: string | null; vessel_type: string | null; vessel_class: string | null };
    let latest: FilterPayload | null = null;
    for (const part of parts) {
      if (part.type === "data-filter" && part.data) {
        latest = part.data as FilterPayload;
      }
    }
    if (!latest) return;
    const key = JSON.stringify(latest);
    if (key === lastAppliedAgentFilterRef.current) return;
    lastAppliedAgentFilterRef.current = key;
    // Exactly one of the three should be non-null (server enforces mutual exclusion);
    // a full-null payload means "clear".
    if (latest.vessel_id) handleFilterChange("vessel_id", latest.vessel_id);
    else if (latest.vessel_type) handleFilterChange("vessel_type", latest.vessel_type);
    else if (latest.vessel_class) handleFilterChange("vessel_class", latest.vessel_class);
    else handleFilterChange("vessel_id", null);
  }, [messages, handleFilterChange]);

  const handleViewCitation = useCallback((chunkId: string, lines?: [number, number][]) => {
    setSelectedChunkId(chunkId);
    setLinesToHighlight(lines);
    setDialogOpen(true);
  }, []);

  // Chat input state
  const [input, setInput] = useState("");
  const handleSend = useCallback(() => {
    if (!input.trim() || isStreaming) return;
    sendMessage({ text: input.trim() });
    setInput("");
  }, [input, isStreaming, sendMessage]);

  // Auth/loading/error guards
  if (authLoading) return <div className="min-h-screen flex items-center justify-center bg-gray-50"><div className="text-MTSS-gray">Loading...</div></div>;
  if (!session) return <LoginForm />;
  if (loading) return <div className="min-h-screen flex items-center justify-center bg-gray-50"><div className="text-MTSS-gray">Loading conversation...</div></div>;
  if (error) return <div className="min-h-screen flex flex-col items-center justify-center bg-gray-50 gap-4"><p className="text-red-600">{error}</p><Button onClick={() => navigate("/conversations")}>Back to conversations</Button></div>;

  const isArchived = conversation?.is_archived ?? false;

  // Build filter options
  const vesselOptions = [{ id: null, name: "All Vessels" }, ...vessels.map((v) => ({ id: v.id, name: v.name }))];
  const typeOptions = [{ id: null, name: "All Types" }, ...vesselTypes.map((t) => ({ id: t, name: t }))];
  const classOptions = [{ id: null, name: "All Classes" }, ...vesselClasses.map((c) => ({ id: c, name: c }))];

  // Visible messages (filter out tool/system/empty)
  const visibleMessages = messages.filter((m) => {
    if (m.role !== "user" && m.role !== "assistant") return false;
    const text = m.parts.filter((p): p is { type: "text"; text: string } => p.type === "text").map((p) => p.text).join("");
    if (!text.trim()) return false;
    if (m.role === "assistant" && (text.startsWith('{"context":') || text.startsWith('{"available_chunk_ids":'))) return false;
    return true;
  });

  return (
    <CitationProvider onViewCitation={handleViewCitation}>
      <div className="flex h-screen overflow-hidden flex-col bg-gray-50">
        {/* Header */}
        <header className="sticky top-0 z-40 w-full border-b border-MTSS-gray-light bg-white">
          <div className="flex h-14 items-center justify-between px-4">
            <div className="flex items-center gap-3">
              <Button variant="ghost" size="sm" onClick={() => navigate("/conversations")} className="gap-1 -ml-2">
                <ArrowLeft className="h-4 w-4" /> Back
              </Button>
              <div className="h-6 w-px bg-MTSS-gray-light" />
              <h1 className="text-sm font-medium text-MTSS-blue-dark truncate max-w-[300px]">{conversation?.title || "New conversation"}</h1>
              {isArchived && <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-amber-100 text-amber-800"><Archive className="h-3 w-3" />Archived</span>}
            </div>
          </div>
        </header>

        {/* Error display */}
        {chatError && (
          <div className="flex-shrink-0 flex items-center gap-2 px-4 py-2 bg-red-50 border-b border-red-200">
            <AlertCircle className="h-4 w-4 text-red-500" />
            <span className="text-sm text-red-700">{chatError.message}</span>
          </div>
        )}

        {/* History loading overlay */}
        {!historyLoaded && (
          <div className="absolute inset-0 z-10 flex items-center justify-center bg-white/50 backdrop-blur-sm">
            <div className="flex flex-col items-center gap-2">
              <div className="animate-spin rounded-full border-2 border-MTSS-blue border-t-transparent h-8 w-8" />
              <span className="text-sm text-MTSS-blue font-medium">Loading history...</span>
            </div>
          </div>
        )}

        {/* Messages */}
        <div ref={scrollRef} className="flex-1 overflow-y-auto min-h-0 p-4 pb-36 space-y-6 scroll-smooth">
          {visibleMessages.length === 0 && !isStreaming && (
            <div className="flex gap-4 max-w-3xl mx-auto">
              <div className="flex-shrink-0 w-8 h-8 rounded-full bg-MTSS-blue flex items-center justify-center shadow-sm"><span className="text-white text-xs font-semibold">AI</span></div>
              <div className="flex-1 bg-white border border-gray-100 shadow-sm rounded-2xl rounded-tl-none p-5">
                <p className="text-gray-600 leading-relaxed">Welcome to MTSS. Describe any vessel issue and I'll search the knowledge base for solutions.</p>
              </div>
            </div>
          )}

          {visibleMessages.map((msg, idx) => {
            const isAssistant = msg.role === "assistant";
            const text = msg.parts.filter((p): p is { type: "text"; text: string } => p.type === "text").map((p) => p.text).join("");
            const isLast = idx === visibleMessages.length - 1;
            const isCurrentStreaming = isStreaming && isLast && isAssistant;

            return (
              <div key={msg.id} className={`flex gap-4 max-w-3xl mx-auto ${isAssistant ? "" : "flex-row-reverse"}`}>
                <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center shadow-sm ${isAssistant ? "bg-MTSS-blue" : "bg-gray-700"}`}>
                  <span className="text-white text-xs font-semibold">{isAssistant ? "AI" : "U"}</span>
                </div>
                <div className={`flex-1 max-w-[85%] rounded-2xl p-5 shadow-sm ${isAssistant ? "bg-white border border-gray-100 rounded-tl-none" : "bg-MTSS-blue text-white rounded-tr-none ml-auto"}`}>
                  {isAssistant ? (
                    <AssistantMessage content={text} messageId={msg.id} threadId={threadId} showFeedback={!isCurrentStreaming} />
                  ) : (
                    <p className="whitespace-pre-wrap leading-relaxed">{text}</p>
                  )}
                  {isCurrentStreaming && (
                    <div className="mt-2 flex items-center gap-2"><Loader2 className="h-4 w-4 animate-spin text-gray-400" /></div>
                  )}
                </div>
              </div>
            );
          })}

          {/* Skeleton while waiting for first token */}
          {status === "submitted" && (!messages.length || messages[messages.length - 1].role !== "assistant") && (
            <div className="flex gap-4 max-w-3xl mx-auto">
              <div className="flex-shrink-0 w-8 h-8 rounded-full bg-MTSS-blue flex items-center justify-center shadow-sm"><span className="text-white text-xs font-semibold">AI</span></div>
              <div className="flex-1 max-w-[85%] bg-white border border-gray-100 shadow-sm rounded-2xl rounded-tl-none p-5 space-y-2">
                <div className="h-4 w-48 bg-gray-200 rounded animate-pulse" />
                <div className="h-4 w-64 bg-gray-200 rounded animate-pulse" />
                <div className="h-4 w-32 bg-gray-200 rounded animate-pulse" />
              </div>
            </div>
          )}
        </div>

        {/* Input area — fixed at bottom */}
        <div className="flex-shrink-0 border-t border-gray-200 bg-white fixed bottom-0 left-0 right-0 z-10">
          {/* Filter row */}
          <div className="px-4 py-2 bg-gray-50/80 border-b border-gray-200">
            <div className="max-w-3xl mx-auto flex items-center gap-2">
              <FilterDropdown label="Vessel" value={vesselId} options={vesselOptions} loading={vesselsLoading} onChange={(v) => handleFilterChange("vessel_id", v)} disabled={isArchived} searchable />
              <FilterDropdown label="Type" value={vesselTypeId} options={typeOptions} onChange={(v) => handleFilterChange("vessel_type", v)} disabled={isArchived} />
              <FilterDropdown label="Class" value={vesselClassId} options={classOptions} onChange={(v) => handleFilterChange("vessel_class", v)} disabled={isArchived} />
            </div>
          </div>

          {/* Input */}
          <form onSubmit={(e: FormEvent) => { e.preventDefault(); handleSend(); }} className="p-4 bg-white/80 backdrop-blur-md">
            <div className="max-w-3xl mx-auto flex gap-3">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e: KeyboardEvent<HTMLTextAreaElement>) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); } }}
                placeholder={isArchived ? "This conversation is archived (read-only)" : "Describe an issue or search for technical information..."}
                disabled={isArchived || isStreaming}
                className={cn("flex-1 resize-none rounded-xl border border-gray-200 px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-MTSS-blue/20 focus:border-MTSS-blue transition-all shadow-sm",
                  (isArchived || isStreaming) ? "opacity-50 cursor-not-allowed bg-gray-50" : "bg-white")}
                rows={1}
                style={{ minHeight: "46px" }}
              />
              <button
                type="submit"
                disabled={isArchived || isStreaming || !input.trim()}
                className={cn("px-4 rounded-xl bg-MTSS-blue text-white transition-all shadow-sm hover:shadow active:scale-95",
                  (isArchived || isStreaming || !input.trim()) ? "opacity-50 cursor-not-allowed" : "hover:bg-MTSS-blue-dark")}
              >
                {isStreaming ? <Loader2 className="h-5 w-5 animate-spin" /> : <Send className="h-5 w-5" />}
              </button>
            </div>
          </form>
        </div>

        {/* Citation dialog */}
        <SourceViewDialog chunkId={selectedChunkId} open={dialogOpen} onOpenChange={setDialogOpen} linesToHighlight={linesToHighlight} />
      </div>
    </CitationProvider>
  );
}

// Helper to convert backend ChatMessage to UI SDK format
function toUIMessage(msg: { id: string; role: string; content: string }): UIMessage {
  return {
    id: msg.id,
    role: msg.role as "user" | "assistant",
    parts: [{ type: "text" as const, text: msg.content }],
  };
}

// ============================================
// Page Export
// ============================================

export default function ChatPage() {
  return (
    <ErrorBoundary>
      <ChatPageContent />
    </ErrorBoundary>
  );
}
