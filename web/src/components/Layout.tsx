/**
 * Layout components - Header and MainLayout.
 *
 * MainLayout uses a full-viewport flex column: Header (fixed) + main
 * (flex-1, min-h-0 so inner scrollers behave) + sticky footer. Pages that
 * need their own scroll container drop it inside <main> with `min-h-0`
 * + `overflow-y-auto` so scrolling stays bounded and the footer remains
 * pinned to the bottom of the viewport.
 */

import { ReactNode, useEffect, useState } from "react";
import { Mail, FileText, Tags, Ship } from "lucide-react";
import { UserMenu, useAuth } from "./auth";
import { getApiBaseUrl } from "@/lib/conversations";

// Git SHA + build time injected at image build via Vite env (VITE_GIT_SHA,
// VITE_BUILD_TIME). Dev builds fall back to placeholder values.
const gitSha = import.meta.env.VITE_GIT_SHA || "dev";
const gitShaShort = gitSha.length >= 8 ? gitSha.substring(0, 8) : gitSha;
const buildTime = import.meta.env.VITE_BUILD_TIME || "";

function formatBuildTime(iso: string): string {
  if (!iso) return "local build";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString();
}

// =============================================================================
// Header
// =============================================================================

function Header() {
  return (
    <header className="sticky top-0 z-40 w-full border-b border-MTSS-gray-light bg-white shrink-0">
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
// Corpus stats (footer left side)
// =============================================================================

interface CorpusStats {
  emails: number;
  documents: number;
  topics: number;
  vessels: number;
}

function useCorpusStats(): CorpusStats | null {
  const { session } = useAuth();
  const [stats, setStats] = useState<CorpusStats | null>(null);

  useEffect(() => {
    if (!session?.access_token) return;
    let cancelled = false;
    (async () => {
      try {
        const r = await fetch(`${getApiBaseUrl()}/stats`, {
          headers: { Authorization: `Bearer ${session.access_token}` },
        });
        if (!r.ok) return;
        const data = (await r.json()) as CorpusStats;
        if (!cancelled) setStats(data);
      } catch {
        // Footer stats are non-essential — silently drop on failure.
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [session?.access_token]);

  return stats;
}

interface StatItemProps {
  icon: ReactNode;
  label: string;
  value: number;
}

function StatItem({ icon, label, value }: StatItemProps) {
  return (
    <span
      title={label}
      aria-label={`${value.toLocaleString()} ${label}`}
      className="inline-flex items-center gap-1"
    >
      {icon}
      <span>{value.toLocaleString()}</span>
    </span>
  );
}

// =============================================================================
// Main Layout
// =============================================================================

interface MainLayoutProps {
  children: ReactNode;
}

export function MainLayout({ children }: MainLayoutProps) {
  const stats = useCorpusStats();
  return (
    <div className="flex h-screen flex-col bg-gray-50">
      <Header />
      <main className="flex-1 min-h-0 flex flex-col">
        {children}
      </main>
      <footer className="border-t border-MTSS-gray-light bg-white py-3 px-6 shrink-0">
        <div className="flex items-center justify-between text-xs text-MTSS-gray">
          <div className="flex items-center gap-4">
            {stats ? (
              <>
                <StatItem icon={<Mail className="h-3.5 w-3.5" />} label="emails" value={stats.emails} />
                <StatItem icon={<FileText className="h-3.5 w-3.5" />} label="documents" value={stats.documents} />
                <StatItem icon={<Tags className="h-3.5 w-3.5" />} label="topics" value={stats.topics} />
                <StatItem icon={<Ship className="h-3.5 w-3.5" />} label="vessels" value={stats.vessels} />
              </>
            ) : (
              <span className="opacity-0">placeholder</span>
            )}
          </div>
          <div className="flex flex-col items-end leading-tight">
            <span>MTSS v1.0.0</span>
            <span
              className="font-mono"
              title={`Git SHA: ${gitSha}\nBuild: ${formatBuildTime(buildTime)}`}
            >
              Build: {gitShaShort}
            </span>
          </div>
        </div>
      </footer>
    </div>
  );
}
