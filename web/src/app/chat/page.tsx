"use client";

/**
 * Chat page - individual conversation view.
 * Full-screen chat with back navigation and vessel selector.
 */

import { useState, useEffect, useCallback, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { ArrowLeft, Archive } from "lucide-react";

import { useAuth, LoginForm } from "@/components/auth";
import { ChatContainer } from "@/components/ChatContainer";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import { Button } from "@/components/ui";
import { initLangfuse } from "@/lib/langfuse";
import {
  Conversation,
  Vessel,
  getConversation,
  createConversation,
  updateConversation,
  listVessels,
  listVesselTypes,
  listVesselClasses,
  ConversationApiError,
} from "@/lib/conversations";

// Initialize Langfuse browser SDK on module load
if (typeof window !== "undefined") {
  initLangfuse();
}

// ============================================
// Chat Header
// ============================================

interface ChatHeaderProps {
  conversation: Conversation | null;
  onBack: () => void;
}

function ChatHeader({
  conversation,
  onBack,
}: ChatHeaderProps) {
  const isArchived = conversation?.is_archived ?? false;

  return (
    <header className="sticky top-0 z-40 w-full border-b border-MTSS-gray-light bg-white">
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
          <div className="h-6 w-px bg-MTSS-gray-light" />
          <h1 className="text-sm font-medium text-MTSS-blue-dark truncate max-w-[300px]">
            {conversation?.title || "New conversation"}
          </h1>
          {isArchived && (
            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-amber-100 text-amber-800">
              <Archive className="h-3 w-3" />
              Archived
            </span>
          )}
        </div>
      </div>
    </header>
  );
}

// ============================================
// Main Chat Page Content
// ============================================

function ChatPageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  // Get threadId from query parameter: /chat?threadId=xxx
  const threadId = searchParams.get("threadId") || "";
  const { session, loading: authLoading } = useAuth();

  const [conversation, setConversation] = useState<Conversation | null>(null);
  const [vesselId, setVesselId] = useState<string | null>(null);
  const [vesselClassId, setVesselClassId] = useState<string | null>(null);
  const [vesselTypeId, setVesselTypeId] = useState<string | null>(null);
  const [vessels, setVessels] = useState<Vessel[]>([]);
  const [vesselTypes, setVesselTypes] = useState<string[]>([]);
  const [vesselClasses, setVesselClasses] = useState<string[]>([]);
  const [vesselsLoading, setVesselsLoading] = useState(true);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Load vessels, types, and classes from API
  useEffect(() => {
    if (!session) return;

    async function loadFilters() {
      try {
        // Load all filter options in parallel
        const [vesselList, types, classes] = await Promise.all([
          listVessels(),
          listVesselTypes(),
          listVesselClasses(),
        ]);
        setVessels(vesselList);
        setVesselTypes(types);
        setVesselClasses(classes);
      } catch (err) {
        console.error("Failed to load filter options:", err);
        // Non-fatal error - continue without filters
      } finally {
        setVesselsLoading(false);
      }
    }

    loadFilters();
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
        // Restore filter state from conversation
        setVesselId(conv.vessel_id);
        setVesselTypeId(conv.vessel_type);
        setVesselClassId(conv.vessel_class);
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

  // Handle vessel filter change - clears other filters (mutual exclusivity)
  const handleVesselChange = useCallback(
    async (value: string | null) => {
      setVesselId(value);
      // Always clear other filters for mutual exclusivity
      setVesselClassId(null);
      setVesselTypeId(null);
      if (conversation) {
        try {
          // Update conversation with new filter (and clear others)
          await updateConversation(threadId, {
            vessel_id: value,
            vessel_type: null,
            vessel_class: null,
          });
        } catch (err) {
          console.error("Failed to update vessel filter:", err);
        }
      }
    },
    [conversation, threadId]
  );

  // Handle vessel type filter change - clears other filters (mutual exclusivity)
  const handleVesselTypeChange = useCallback(
    async (value: string | null) => {
      setVesselTypeId(value);
      // Always clear other filters for mutual exclusivity
      setVesselId(null);
      setVesselClassId(null);
      if (conversation) {
        try {
          // Update conversation with new filter (and clear others)
          await updateConversation(threadId, {
            vessel_id: null,
            vessel_type: value,
            vessel_class: null,
          });
        } catch (err) {
          console.error("Failed to update type filter:", err);
        }
      }
    },
    [conversation, threadId]
  );

  // Handle vessel class filter change - clears other filters (mutual exclusivity)
  const handleVesselClassChange = useCallback(
    async (value: string | null) => {
      setVesselClassId(value);
      // Always clear other filters for mutual exclusivity
      setVesselId(null);
      setVesselTypeId(null);
      if (conversation) {
        try {
          // Update conversation with new filter (and clear others)
          await updateConversation(threadId, {
            vessel_id: null,
            vessel_type: null,
            vessel_class: value,
          });
        } catch (err) {
          console.error("Failed to update class filter:", err);
        }
      }
    },
    [conversation, threadId]
  );

  // Auth loading
  if (authLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-MTSS-gray">Loading...</div>
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
        <div className="text-MTSS-gray">Loading conversation...</div>
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
        onBack={handleBack}
      />
      <main className="flex-1 flex flex-col min-h-0 overflow-hidden">
        <ChatContainer
          threadId={threadId}
          authToken={session.access_token}
          disabled={isArchived}
          vesselId={vesselId}
          vessels={vessels}
          vesselTypes={vesselTypes}
          vesselClasses={vesselClasses}
          vesselsLoading={vesselsLoading}
          vesselClassId={vesselClassId}
          vesselTypeId={vesselTypeId}
          onVesselChange={handleVesselChange}
          onVesselClassChange={handleVesselClassChange}
          onVesselTypeChange={handleVesselTypeChange}
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
      <Suspense fallback={
        <div className="min-h-screen flex items-center justify-center bg-gray-50">
          <div className="text-MTSS-gray">Loading...</div>
        </div>
      }>
        <ChatPageContent />
      </Suspense>
    </ErrorBoundary>
  );
}
