"use client";

/**
 * Chat page - individual conversation view.
 * Full-screen chat with back navigation and vessel selector.
 */

import { useState, useEffect, useCallback } from "react";
import { useRouter, useParams } from "next/navigation";
import { ArrowLeft, Ship, ChevronDown, Archive } from "lucide-react";
import * as DropdownMenu from "@radix-ui/react-dropdown-menu";

import { useAuth, LoginForm } from "@/components/auth";
import { ChatContainer } from "@/components/ChatContainer";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import { Button } from "@/components/ui";
import { cn } from "@/lib/utils";
import { initLangfuse } from "@/lib/langfuse";
import {
  Conversation,
  Vessel,
  getConversation,
  createConversation,
  updateConversation,
  listVessels,
  ConversationApiError,
} from "@/lib/conversations";

// Initialize Langfuse browser SDK on module load
if (typeof window !== "undefined") {
  initLangfuse();
}

// ============================================
// Vessel Selector
// ============================================

interface VesselSelectorProps {
  value: string | null;
  vessels: Vessel[];
  loading?: boolean;
  onChange: (value: string | null) => void;
}

function VesselSelector({ value, vessels, loading, onChange }: VesselSelectorProps) {
  const [search, setSearch] = useState("");

  // Build options list with "All Vessels" at the top
  const options: { id: string | null; name: string }[] = [
    { id: null, name: "All Vessels" },
    ...vessels.map((v) => ({ id: v.id, name: v.name })),
  ];

  // Filter options by search term
  const filteredOptions = search
    ? options.filter((v) =>
        v.name.toLowerCase().includes(search.toLowerCase())
      )
    : options;

  const selectedOption = options.find((v) => v.id === value) || options[0];

  return (
    <DropdownMenu.Root onOpenChange={(open) => !open && setSearch("")}>
      <DropdownMenu.Trigger asChild>
        <Button variant="outline" size="sm" className="gap-2" disabled={loading}>
          <Ship className="h-4 w-4" />
          <span className="max-w-[120px] truncate">
            {loading ? "Loading..." : selectedOption.name}
          </span>
          <ChevronDown className="h-3 w-3 opacity-50" />
        </Button>
      </DropdownMenu.Trigger>
      <DropdownMenu.Portal>
        <DropdownMenu.Content
          className="min-w-[180px] bg-white rounded-md shadow-lg border border-ncl-gray-light z-50"
          align="end"
        >
          {/* Search input */}
          <div className="p-2 border-b border-ncl-gray-light">
            <input
              type="text"
              placeholder="Search vessels..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-full px-2 py-1 text-sm border border-ncl-gray-light rounded outline-none focus:border-ncl-blue"
              onClick={(e) => e.stopPropagation()}
              onKeyDown={(e) => e.stopPropagation()}
            />
          </div>
          {/* Filtered list */}
          <div className="max-h-[250px] overflow-y-auto p-1">
            {filteredOptions.length > 0 ? (
              filteredOptions.map((option) => (
                <DropdownMenu.Item
                  key={option.id || "all"}
                  className={cn(
                    "flex items-center gap-2 px-3 py-2 text-sm cursor-pointer rounded outline-none",
                    option.id === value
                      ? "bg-ncl-blue/10 text-ncl-blue"
                      : "hover:bg-ncl-gray-light/50"
                  )}
                  onSelect={() => onChange(option.id)}
                >
                  {option.name}
                </DropdownMenu.Item>
              ))
            ) : (
              <div className="px-3 py-2 text-sm text-gray-500">
                No vessels found
              </div>
            )}
          </div>
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
  vesselId: string | null;
  vessels: Vessel[];
  vesselsLoading: boolean;
  onBack: () => void;
  onVesselChange: (value: string | null) => void;
}

function ChatHeader({
  conversation,
  vesselId,
  vessels,
  vesselsLoading,
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
          <VesselSelector
            value={vesselId}
            vessels={vessels}
            loading={vesselsLoading}
            onChange={onVesselChange}
          />
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
  const [vesselId, setVesselId] = useState<string | null>(null);
  const [vessels, setVessels] = useState<Vessel[]>([]);
  const [vesselsLoading, setVesselsLoading] = useState(true);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Load vessels from API
  useEffect(() => {
    if (!session) return;

    async function loadVessels() {
      try {
        const vesselList = await listVessels();
        setVessels(vesselList);
      } catch (err) {
        console.error("Failed to load vessels:", err);
        // Non-fatal error - continue without vessels
      } finally {
        setVesselsLoading(false);
      }
    }

    loadVessels();
  }, [session]);

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
        setVesselId(conv.vessel_id);
      } catch (err) {
        if (err instanceof ConversationApiError && err.status === 404) {
          // Conversation doesn't exist yet - create it
          try {
            const newConv = await createConversation({ thread_id: threadId });
            setConversation(newConv);
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
      setVesselId(value);
      if (conversation) {
        try {
          // Pass null explicitly to clear vessel filter, or the vessel_id to set it
          await updateConversation(threadId, { vessel_id: value });
        } catch (err) {
          console.error("Failed to update vessel filter:", err);
        }
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
    <div className="flex h-screen overflow-hidden flex-col bg-gray-50">
      <ChatHeader
        conversation={conversation}
        vesselId={vesselId}
        vessels={vessels}
        vesselsLoading={vesselsLoading}
        onBack={handleBack}
        onVesselChange={handleVesselChange}
      />
      <main className="flex-1 flex flex-col min-h-0 overflow-hidden">
        <ChatContainer
          threadId={threadId}
          authToken={session.access_token}
          disabled={isArchived}
          vesselId={vesselId}
        />
      </main>
    </div>
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
