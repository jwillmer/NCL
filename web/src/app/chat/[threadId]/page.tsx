"use client";

/**
 * Chat page - individual conversation view.
 * Full-screen chat with back navigation and vessel selector.
 */

import { useState, useEffect, useCallback, useRef } from "react";
import { useRouter, useParams } from "next/navigation";
import { ArrowLeft, Ship, ChevronDown, Archive } from "lucide-react";
import * as DropdownMenu from "@radix-ui/react-dropdown-menu";
import { CopilotKit } from "@copilotkit/react-core";

import { useAuth, LoginForm } from "@/components/auth";
import { ChatContainer } from "@/components/ChatContainer";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import { Button } from "@/components/ui";
import { cn } from "@/lib/utils";
import {
  Conversation,
  getConversation,
  createConversation,
  updateConversation,
  touchConversation,
  generateTitle,
  ConversationApiError,
} from "@/lib/conversations";

// ============================================
// Vessel Selector (UI Placeholder)
// ============================================

interface VesselSelectorProps {
  value: string | null;
  onChange: (value: string | null) => void;
}

function VesselSelector({ value, onChange }: VesselSelectorProps) {
  // Placeholder vessel list - will be replaced with actual data later
  const vessels = [
    { id: null, name: "All Vessels" },
    { id: "vessel-1", name: "MV Atlantic Star" },
    { id: "vessel-2", name: "MV Pacific Explorer" },
    { id: "vessel-3", name: "MV Nordic Spirit" },
  ];

  const selectedVessel = vessels.find((v) => v.id === value) || vessels[0];

  return (
    <DropdownMenu.Root>
      <DropdownMenu.Trigger asChild>
        <Button variant="outline" size="sm" className="gap-2">
          <Ship className="h-4 w-4" />
          <span className="max-w-[120px] truncate">{selectedVessel.name}</span>
          <ChevronDown className="h-3 w-3 opacity-50" />
        </Button>
      </DropdownMenu.Trigger>
      <DropdownMenu.Portal>
        <DropdownMenu.Content
          className="min-w-[180px] bg-white rounded-md shadow-lg border border-ncl-gray-light p-1 z-50"
          align="end"
        >
          {vessels.map((vessel) => (
            <DropdownMenu.Item
              key={vessel.id || "all"}
              className={cn(
                "flex items-center gap-2 px-3 py-2 text-sm cursor-pointer rounded outline-none",
                vessel.id === value
                  ? "bg-ncl-blue/10 text-ncl-blue"
                  : "hover:bg-ncl-gray-light/50"
              )}
              onSelect={() => onChange(vessel.id)}
            >
              {vessel.name}
            </DropdownMenu.Item>
          ))}
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  );
}

// ============================================
// Chat Header
// ============================================

interface ChatHeaderProps {
  conversation: Conversation | null;
  vesselFilter: string | null;
  onBack: () => void;
  onVesselChange: (value: string | null) => void;
}

function ChatHeader({
  conversation,
  vesselFilter,
  onBack,
  onVesselChange,
}: ChatHeaderProps) {
  const isArchived = conversation?.is_archived ?? false;

  return (
    <header className="sticky top-0 z-40 w-full border-b border-ncl-gray-light bg-white">
      <div className="flex h-14 items-center justify-between px-4">
        <div className="flex items-center gap-3">
          <Button
            variant="ghost"
            size="sm"
            onClick={onBack}
            className="gap-1 -ml-2"
          >
            <ArrowLeft className="h-4 w-4" />
            Back
          </Button>
          <div className="h-6 w-px bg-ncl-gray-light" />
          <h1 className="text-sm font-medium text-ncl-blue-dark truncate max-w-[300px]">
            {conversation?.title || "New conversation"}
          </h1>
          {isArchived && (
            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-amber-100 text-amber-800">
              <Archive className="h-3 w-3" />
              Archived
            </span>
          )}
        </div>
        {!isArchived && (
          <VesselSelector value={vesselFilter} onChange={onVesselChange} />
        )}
      </div>
    </header>
  );
}

// ============================================
// Main Chat Page Content
// ============================================

function ChatPageContent() {
  const router = useRouter();
  const params = useParams();
  const threadId = params.threadId as string;
  const { session, loading: authLoading } = useAuth();

  const [conversation, setConversation] = useState<Conversation | null>(null);
  const [vesselFilter, setVesselFilter] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const hasGeneratedTitle = useRef(false);
  const isFirstMessage = useRef(true);

  // Load or create conversation
  useEffect(() => {
    if (!session || !threadId) return;

    async function loadOrCreateConversation() {
      setLoading(true);
      setError(null);

      try {
        // Try to load existing conversation
        const conv = await getConversation(threadId);
        setConversation(conv);
        setVesselFilter(conv.vessel_filter);
        isFirstMessage.current = !conv.title; // No title means likely new
      } catch (err) {
        if (err instanceof ConversationApiError && err.status === 404) {
          // Conversation doesn't exist yet - create it
          try {
            const newConv = await createConversation({ thread_id: threadId });
            setConversation(newConv);
            isFirstMessage.current = true;
          } catch (createErr) {
            console.error("Failed to create conversation:", createErr);
            setError("Failed to create conversation");
          }
        } else {
          console.error("Failed to load conversation:", err);
          setError("Failed to load conversation");
        }
      } finally {
        setLoading(false);
      }
    }

    loadOrCreateConversation();
  }, [session, threadId]);

  // Handle back navigation
  const handleBack = useCallback(() => {
    router.push("/conversations");
  }, [router]);

  // Handle vessel filter change
  const handleVesselChange = useCallback(
    async (value: string | null) => {
      setVesselFilter(value);
      if (conversation) {
        try {
          await updateConversation(threadId, { vessel_filter: value || undefined });
        } catch (err) {
          console.error("Failed to update vessel filter:", err);
        }
      }
    },
    [conversation, threadId]
  );

  // Callback for when a message is sent (to generate title and touch timestamp)
  const handleMessageSent = useCallback(
    async (content: string) => {
      if (!conversation) return;

      try {
        // Touch conversation to update last_message_at
        const updated = await touchConversation(threadId);
        setConversation(updated);

        // Generate title on first user message if no title exists
        if (isFirstMessage.current && !hasGeneratedTitle.current && content) {
          hasGeneratedTitle.current = true;
          isFirstMessage.current = false;
          const withTitle = await generateTitle(threadId, content);
          setConversation(withTitle);
        }
      } catch (err) {
        console.error("Failed to update conversation:", err);
      }
    },
    [conversation, threadId]
  );

  // Auth loading
  if (authLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-ncl-gray">Loading...</div>
      </div>
    );
  }

  // Not authenticated
  if (!session) {
    return <LoginForm />;
  }

  // Conversation loading
  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-ncl-gray">Loading conversation...</div>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center bg-gray-50 gap-4">
        <p className="text-red-600">{error}</p>
        <Button onClick={handleBack}>Back to conversations</Button>
      </div>
    );
  }

  const isArchived = conversation?.is_archived ?? false;

  return (
    <CopilotKit
      runtimeUrl="/api/copilotkit"
      agent="default"
      threadId={threadId}
      headers={{
        Authorization: `Bearer ${session.access_token}`,
      }}
    >
      <div className="flex min-h-screen flex-col bg-gray-50">
        <ChatHeader
          conversation={conversation}
          vesselFilter={vesselFilter}
          onBack={handleBack}
          onVesselChange={handleVesselChange}
        />
        <main className="flex-1 flex flex-col">
          <ChatContainer
            onMessageSent={handleMessageSent}
            disabled={isArchived}
          />
        </main>
      </div>
    </CopilotKit>
  );
}

// ============================================
// Page Export with Error Boundary
// ============================================

export default function ChatPage() {
  return (
    <ErrorBoundary>
      <ChatPageContent />
    </ErrorBoundary>
  );
}
