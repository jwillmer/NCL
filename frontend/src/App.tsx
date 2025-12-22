/**
 * Root application component with routing and providers.
 * System prompt is fetched from backend to centralize AI instructions.
 */

import { useEffect, useState } from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { CopilotKit } from "@copilotkit/react-core";
import { AuthProvider, LoginForm, useAuth } from "@/components/auth";
import { ProtectedRoute } from "@/components/ProtectedRoute";
import { MainLayout } from "@/components/Layout";
import { ChatContainer } from "@/components/ChatContainer";

const API_URL = import.meta.env.VITE_API_URL || "";

function useSystemPrompt() {
  const [prompt, setPrompt] = useState<string>("");

  useEffect(() => {
    fetch(`${API_URL}/system-prompt`)
      .then((res) => res.json())
      .then((data) => setPrompt(data.prompt))
      .catch((err) => console.error("Failed to fetch system prompt:", err));
  }, []);

  return prompt;
}

function AppRoutes() {
  const { session } = useAuth();
  const systemPrompt = useSystemPrompt();

  return (
    <Routes>
      <Route
        path="/login"
        element={session ? <Navigate to="/" replace /> : <LoginForm />}
      />
      <Route
        path="/"
        element={
          <ProtectedRoute>
            <CopilotKit
              runtimeUrl={`${API_URL}/copilotkit`}
              headers={{
                Authorization: `Bearer ${session?.access_token || ""}`,
              }}
              instructions={systemPrompt}
            >
              <MainLayout>
                <ChatContainer />
              </MainLayout>
            </CopilotKit>
          </ProtectedRoute>
        }
      />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <AppRoutes />
      </AuthProvider>
    </BrowserRouter>
  );
}
