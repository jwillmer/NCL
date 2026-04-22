import { lazy, Suspense } from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { AuthProvider } from "@/components/auth";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import { VesselProvider } from "@/hooks/useVessels";
import { usePwaUpdate } from "@/hooks/usePwaUpdate";
import { Button } from "@/components/ui";

const ChatPage = lazy(() => import("@/pages/ChatPage"));
const ConversationsPage = lazy(() => import("@/pages/ConversationsPage"));

function LoadingSpinner() {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        minHeight: "100vh",
        color: "#6b7280",
        fontSize: "0.875rem",
      }}
    >
      Loading...
    </div>
  );
}

/**
 * Fixed-position toast that surfaces when the service worker has a new
 * build waiting. Click the button to call `updateServiceWorker(true)` —
 * that skips waiting, claims clients, and reloads.
 *
 * Kept inline (no toast library in the repo) and intentionally minimal.
 */
function UpdateToast() {
  const {
    needRefresh: [needRefresh, setNeedRefresh],
    updateServiceWorker,
  } = usePwaUpdate();

  if (!needRefresh) return null;

  return (
    <div
      className="fixed bottom-4 right-4 z-[60] flex items-center gap-3 rounded-lg border border-MTSS-gray-light bg-white px-4 py-3 shadow-lg"
      role="status"
      aria-live="polite"
    >
      <span className="text-sm text-MTSS-blue-dark">
        New version available — reload to update.
      </span>
      <Button
        size="sm"
        onClick={() => updateServiceWorker(true)}
      >
        Reload
      </Button>
      <Button
        size="sm"
        variant="ghost"
        onClick={() => setNeedRefresh(false)}
        aria-label="Dismiss update notice"
      >
        Later
      </Button>
    </div>
  );
}

export function App() {
  return (
    <ErrorBoundary>
      <BrowserRouter>
        <AuthProvider>
          <VesselProvider>
            <Suspense fallback={<LoadingSpinner />}>
              <Routes>
                <Route path="/" element={<Navigate to="/conversations" replace />} />
                <Route path="/conversations" element={<ConversationsPage />} />
                <Route path="/chat" element={<ChatPage />} />
              </Routes>
            </Suspense>
            <UpdateToast />
          </VesselProvider>
        </AuthProvider>
      </BrowserRouter>
    </ErrorBoundary>
  );
}
