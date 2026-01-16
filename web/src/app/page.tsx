"use client";

/**
 * Main page - redirects to conversations list.
 * Unauthenticated users see the login form.
 */

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuth, LoginForm } from "@/components/auth";

export default function Home() {
  const router = useRouter();
  const { session, loading } = useAuth();

  useEffect(() => {
    if (!loading && session) {
      router.replace("/conversations");
    }
  }, [loading, session, router]);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-MTSS-gray">Loading...</div>
      </div>
    );
  }

  if (!session) {
    return <LoginForm />;
  }

  // Brief loading state while redirecting
  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50">
      <div className="text-MTSS-gray">Redirecting...</div>
    </div>
  );
}
