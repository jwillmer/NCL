"use client";

/**
 * Conversations list page - main landing after login.
 * Shows all user conversations with search and grouping by date.
 */

import { useState, useEffect, useCallback, useRef } from "react";
import { useRouter } from "next/navigation";
import { Search, Plus, MessageSquare, Archive, Trash2, MoreVertical } from "lucide-react";
import * as DropdownMenu from "@radix-ui/react-dropdown-menu";
import { v4 as uuidv4 } from "uuid";

import { useAuth, LoginForm } from "@/components/auth";
import { MainLayout } from "@/components/Layout";
import { Button, Input, ScrollArea, Skeleton } from "@/components/ui";
import { cn, formatRelativeTime, groupByDate } from "@/lib/utils";
import {
  Conversation,
  Vessel,
  listConversations,
  deleteConversation,
  updateConversation,
  createConversation,
  listVessels,
} from "@/lib/conversations";

// ============================================
// Conversation Item Component
// ============================================

interface ConversationItemProps {
  conversation: Conversation;
  vesselLookup: Record<string, string>;  // vessel_id -> vessel name
  onSelect: (conv: Conversation) => void;
  onArchive: (conv: Conversation) => void;
  onDelete: (conv: Conversation) => void;
}

/**
 * Get filter badge text based on which filter is active.
 */
function getFilterBadge(conversation: Conversation, vesselLookup: Record<string, string>): string | null {
  if (conversation.vessel_id) {
    const vesselName = vesselLookup[conversation.vessel_id];
    return vesselName ? `Vessel: ${vesselName}` : null;
  }
  if (conversation.vessel_type) {
    return `Type: ${conversation.vessel_type}`;
  }
  if (conversation.vessel_class) {
    return `Class: ${conversation.vessel_class}`;
  }
  return null;
}

function ConversationItem({
  conversation,
  vesselLookup,
  onSelect,
  onArchive,
  onDelete,
}: ConversationItemProps) {
  const filterBadge = getFilterBadge(conversation, vesselLookup);

  return (
    <div
      className={cn(
        "group flex items-start gap-3 p-3 rounded-lg cursor-pointer transition-colors",
        "hover:bg-MTSS-gray-light/50 border border-transparent hover:border-MTSS-gray-light"
      )}
      onClick={() => onSelect(conversation)}
    >
      <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-MTSS-blue/10 shrink-0">
        <MessageSquare className="h-4 w-4 text-MTSS-blue" />
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium text-MTSS-blue-dark truncate">
          {conversation.title || "New conversation"}
        </p>
        <div className="flex items-center gap-2 mt-0.5">
          <span className="text-xs text-MTSS-gray">
            {formatRelativeTime(conversation.last_message_at || conversation.created_at)}
          </span>
          {filterBadge && (
            <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium bg-MTSS-blue/10 text-MTSS-blue">
              {filterBadge}
            </span>
          )}
        </div>
      </div>

      {/* Actions dropdown */}
      <DropdownMenu.Root>
        <DropdownMenu.Trigger asChild>
          <button
            className="opacity-0 group-hover:opacity-100 p-1.5 rounded hover:bg-MTSS-gray-light transition-opacity"
            onClick={(e) => e.stopPropagation()}
          >
            <MoreVertical className="h-4 w-4 text-MTSS-gray" />
          </button>
        </DropdownMenu.Trigger>
        <DropdownMenu.Portal>
          <DropdownMenu.Content
            className="min-w-[140px] bg-white rounded-md shadow-lg border border-MTSS-gray-light p-1 z-50"
            align="end"
            onClick={(e) => e.stopPropagation()}
          >
            <DropdownMenu.Item
              className="flex items-center gap-2 px-3 py-2 text-sm cursor-pointer hover:bg-MTSS-gray-light/50 rounded outline-none"
              onSelect={() => onArchive(conversation)}
            >
              <Archive className="h-4 w-4" />
              {conversation.is_archived ? "Unarchive" : "Archive"}
            </DropdownMenu.Item>
            <DropdownMenu.Item
              className="flex items-center gap-2 px-3 py-2 text-sm text-red-600 cursor-pointer hover:bg-red-50 rounded outline-none"
              onSelect={() => onDelete(conversation)}
            >
              <Trash2 className="h-4 w-4" />
              Delete
            </DropdownMenu.Item>
          </DropdownMenu.Content>
        </DropdownMenu.Portal>
      </DropdownMenu.Root>
    </div>
  );
}

// ============================================
// Loading Skeleton
// ============================================

function ConversationSkeleton() {
  return (
    <div className="flex items-start gap-3 p-3">
      <Skeleton className="h-9 w-9 rounded-lg" />
      <div className="flex-1">
        <Skeleton className="h-4 w-3/4 mb-2" />
        <Skeleton className="h-3 w-1/3" />
      </div>
    </div>
  );
}

// ============================================
// Empty State
// ============================================

function EmptyState({ isSearch }: { isSearch: boolean }) {
  return (
    <div className="flex flex-col items-center justify-center py-16 text-center">
      <div className="flex h-16 w-16 items-center justify-center rounded-full bg-MTSS-gray-light mb-4">
        <MessageSquare className="h-8 w-8 text-MTSS-gray" />
      </div>
      <h3 className="text-lg font-medium text-MTSS-blue-dark mb-1">
        {isSearch ? "No conversations found" : "No conversations yet"}
      </h3>
      <p className="text-sm text-MTSS-gray max-w-sm">
        {isSearch
          ? "Try adjusting your search query"
          : "Start a new conversation to search the knowledge base"}
      </p>
    </div>
  );
}

// ============================================
// Main Conversations Page
// ============================================

function ConversationsPage() {
  const router = useRouter();
  const { session, loading: authLoading } = useAuth();

  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [vesselLookup, setVesselLookup] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(true);
  const [searchInput, setSearchInput] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [showArchived, setShowArchived] = useState(false);
  const [hasMore, setHasMore] = useState(false);
  const [offset, setOffset] = useState(0);
  const [loadingMore, setLoadingMore] = useState(false);
  const searchDebounceRef = useRef<NodeJS.Timeout | null>(null);

  // Load vessels for name lookup
  useEffect(() => {
    if (!session) return;

    async function loadVessels() {
      try {
        const vessels = await listVessels();
        const lookup: Record<string, string> = {};
        for (const vessel of vessels) {
          lookup[vessel.id] = vessel.name;
        }
        setVesselLookup(lookup);
      } catch (error) {
        console.error("Failed to load vessels:", error);
      }
    }

    loadVessels();
  }, [session]);

  const LIMIT = 50;

  // Debounce search input
  useEffect(() => {
    if (searchDebounceRef.current) {
      clearTimeout(searchDebounceRef.current);
    }
    searchDebounceRef.current = setTimeout(() => {
      setSearchQuery(searchInput);
    }, 300);

    return () => {
      if (searchDebounceRef.current) {
        clearTimeout(searchDebounceRef.current);
      }
    };
  }, [searchInput]);

  // Load conversations
  const loadConversations = useCallback(
    async (reset = false) => {
      if (!session) return;

      const currentOffset = reset ? 0 : offset;
      if (reset) {
        setLoading(true);
      } else {
        setLoadingMore(true);
      }

      try {
        const result = await listConversations({
          q: searchQuery || undefined,
          archived: showArchived,
          limit: LIMIT,
          offset: currentOffset,
        });

        if (reset) {
          setConversations(result.items);
          setOffset(result.items.length);
        } else {
          setConversations((prev) => [...prev, ...result.items]);
          setOffset(currentOffset + result.items.length);
        }
        setHasMore(result.has_more);
      } catch (error) {
        console.error("Failed to load conversations:", error);
      } finally {
        setLoading(false);
        setLoadingMore(false);
      }
    },
    [session, searchQuery, showArchived, offset]
  );

  // Initial load and reload on filter changes
  useEffect(() => {
    if (session) {
      loadConversations(true);
    }
  }, [session, searchQuery, showArchived]); // eslint-disable-line react-hooks/exhaustive-deps

  // Handle new conversation
  const handleNewConversation = async () => {
    const threadId = uuidv4();
    try {
      await createConversation({ thread_id: threadId });
      router.push(`/chat?threadId=${threadId}`);
    } catch (error) {
      console.error("Failed to create conversation:", error);
      // Still navigate - conversation will be created on first message
      router.push(`/chat?threadId=${threadId}`);
    }
  };

  // Handle select conversation
  const handleSelectConversation = (conv: Conversation) => {
    router.push(`/chat?threadId=${conv.thread_id}`);
  };

  // Handle archive/unarchive
  const handleArchive = async (conv: Conversation) => {
    try {
      await updateConversation(conv.thread_id, { is_archived: !conv.is_archived });
      loadConversations(true);
    } catch (error) {
      console.error("Failed to archive conversation:", error);
    }
  };

  // Handle delete
  const handleDelete = async (conv: Conversation) => {
    if (!confirm("Are you sure you want to delete this conversation?")) return;

    try {
      await deleteConversation(conv.thread_id);
      setConversations((prev) => prev.filter((c) => c.id !== conv.id));
    } catch (error) {
      console.error("Failed to delete conversation:", error);
    }
  };

  // Group conversations by date
  const groupedConversations = groupByDate(conversations);

  if (authLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-MTSS-gray">Loading...</div>
      </div>
    );
  }

  if (!session) {
    return <LoginForm />;
  }

  return (
    <MainLayout>
      <div className="flex-1 flex flex-col max-w-3xl mx-auto w-full px-4 py-6">
        {/* Header */}
        <div className="flex items-center gap-4 mb-6">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-MTSS-gray" />
            <Input
              type="text"
              placeholder="Search conversations..."
              value={searchInput}
              onChange={(e) => setSearchInput(e.target.value)}
              className="pl-10"
            />
          </div>
          <Button onClick={handleNewConversation} className="gap-2">
            <Plus className="h-4 w-4" />
            New Chat
          </Button>
        </div>

        {/* Archive toggle */}
        <div className="flex items-center gap-2 mb-4">
          <Button
            variant={showArchived ? "secondary" : "ghost"}
            size="sm"
            onClick={() => setShowArchived(!showArchived)}
            className="gap-2"
          >
            <Archive className="h-4 w-4" />
            {showArchived ? "Showing Archived" : "Show Archived"}
          </Button>
        </div>

        {/* Conversations List */}
        <ScrollArea className="flex-1">
          {loading ? (
            <div className="space-y-2">
              {Array.from({ length: 5 }).map((_, i) => (
                <ConversationSkeleton key={i} />
              ))}
            </div>
          ) : conversations.length === 0 ? (
            <EmptyState isSearch={!!searchQuery} />
          ) : (
            <div className="space-y-6">
              {groupedConversations.map((group) => (
                <div key={group.label}>
                  <h3 className="text-xs font-medium text-MTSS-gray uppercase tracking-wider mb-2 px-3">
                    {group.label}
                  </h3>
                  <div className="space-y-1">
                    {group.items.map((conv) => (
                      <ConversationItem
                        key={conv.id}
                        conversation={conv}
                        vesselLookup={vesselLookup}
                        onSelect={handleSelectConversation}
                        onArchive={handleArchive}
                        onDelete={handleDelete}
                      />
                    ))}
                  </div>
                </div>
              ))}

              {/* Load more */}
              {hasMore && (
                <div className="flex justify-center pt-4">
                  <Button
                    variant="outline"
                    onClick={() => loadConversations(false)}
                    disabled={loadingMore}
                  >
                    {loadingMore ? "Loading..." : "Load more"}
                  </Button>
                </div>
              )}
            </div>
          )}
        </ScrollArea>
      </div>
    </MainLayout>
  );
}

export default ConversationsPage;
