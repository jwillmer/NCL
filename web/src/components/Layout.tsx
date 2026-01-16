"use client";

/**
 * Layout components - Header and MainLayout.
 */

import { ReactNode } from "react";
import { Ship } from "lucide-react";
import { UserMenu } from "./auth";

// Git SHA injected at build time via NEXT_PUBLIC_GIT_SHA environment variable
const gitSha = process.env.NEXT_PUBLIC_GIT_SHA || "dev";
const gitShaShort = gitSha.length >= 8 ? gitSha.substring(0, 8) : gitSha;

// =============================================================================
// Header
// =============================================================================

function Header() {
  return (
    <header className="sticky top-0 z-40 w-full border-b border-MTSS-gray-light bg-white">
      <div className="flex h-16 items-center justify-between px-6">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-MTSS-blue">
            <Ship className="h-5 w-5 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-semibold text-MTSS-blue">Maritime Technical Support System (MTSS)</h1>
            <p className="text-xs text-MTSS-gray">Vessel Issue History & Knowledge Base</p>
          </div>
        </div>
        <UserMenu />
      </div>
    </header>
  );
}

// =============================================================================
// Main Layout
// =============================================================================

interface MainLayoutProps {
  children: ReactNode;
}

export function MainLayout({ children }: MainLayoutProps) {
  return (
    <div className="flex min-h-screen flex-col bg-gray-50">
      <Header />
      <main className="flex-1 flex flex-col">
        {children}
      </main>
      <footer className="border-t border-MTSS-gray-light bg-white py-4 px-6">
        <div className="flex items-center justify-between text-xs text-MTSS-gray">
          <span>MTSS v0.1.0</span>
          <span className="font-mono" title={`Git SHA: ${gitSha}`}>
            Build: {gitShaShort}
          </span>
        </div>
      </footer>
    </div>
  );
}
