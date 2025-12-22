/**
 * Layout components - Header and MainLayout.
 * Consolidated into a single file for simplicity (KISS principle).
 */

import { ReactNode } from "react";
import { Mail } from "lucide-react";
import { UserMenu } from "./auth";
import { Separator } from "./ui";

// =============================================================================
// Header
// =============================================================================

function Header() {
  return (
    <header className="sticky top-0 z-40 w-full border-b border-ncl-gray-light bg-white">
      <div className="flex h-16 items-center justify-between px-6">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-ncl-blue">
            <Mail className="h-5 w-5 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-semibold text-ncl-blue">NCL Email Archive</h1>
            <p className="text-xs text-ncl-gray">Intelligent Document Search</p>
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
      <footer className="border-t border-ncl-gray-light bg-white py-4 px-6">
        <div className="flex items-center justify-between text-xs text-ncl-gray">
          <span>NCL Email Archive v0.1.0</span>
          <span>Powered by AI with source attribution</span>
        </div>
      </footer>
    </div>
  );
}
