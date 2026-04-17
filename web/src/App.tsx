import { lazy, Suspense } from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { AuthProvider } from "@/components/auth";
import { ErrorBoundary } from "@/components/ErrorBoundary";

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

export function App() {
  return (
    <ErrorBoundary>
      <BrowserRouter>
        <AuthProvider>
          <Suspense fallback={<LoadingSpinner />}>
            <Routes>
              <Route path="/" element={<Navigate to="/conversations" replace />} />
              <Route path="/conversations" element={<ConversationsPage />} />
              <Route path="/chat" element={<ChatPage />} />
            </Routes>
          </Suspense>
        </AuthProvider>
      </BrowserRouter>
    </ErrorBoundary>
  );
}
