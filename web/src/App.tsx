import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { AuthProvider } from "@/components/auth";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import ChatPage from "@/pages/ChatPage";
import ConversationsPage from "@/pages/ConversationsPage";

export function App() {
  return (
    <ErrorBoundary>
      <BrowserRouter>
        <AuthProvider>
          <Routes>
            <Route path="/" element={<Navigate to="/conversations" replace />} />
            <Route path="/conversations" element={<ConversationsPage />} />
            <Route path="/chat" element={<ChatPage />} />
          </Routes>
        </AuthProvider>
      </BrowserRouter>
    </ErrorBoundary>
  );
}
