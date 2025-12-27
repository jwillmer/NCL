"use client";

/**
 * Main page with CopilotKit integration.
 * Handles authentication flow and renders the chat interface.
 */

import { CopilotKit } from "@copilotkit/react-core";
import { useAuth, LoginForm } from "@/components/auth";
import { MainLayout } from "@/components/Layout";
import { ChatContainer } from "@/components/ChatContainer";
import { ErrorBoundary } from "@/components/ErrorBoundary";

function AuthenticatedApp() {
  const { session, loading } = useAuth();

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-ncl-gray">Loading...</div>
      </div>
    );
  }

  if (!session) {
    return <LoginForm />;
  }

  return (
    <CopilotKit
      runtimeUrl="/api/copilotkit"
      agent="default"
      headers={{
        Authorization: `Bearer ${session.access_token}`,
      }}
    >
      <MainLayout>
        <ChatContainer />
      </MainLayout>
    </CopilotKit>
  );
}

export default function Home() {
  return (
    <ErrorBoundary>
      <AuthenticatedApp />
    </ErrorBoundary>
  );
}
