"use client";

/**
 * Sources components for displaying citations in the chat UI.
 *
 * - CiteRenderer: Renders <cite> tags as interactive inline badges
 * - SourcesAccordion: Collapsible list of all sources in the current message
 * - SourceViewDialog: Full-screen dialog to view source content
 * - CitationProvider: Context to collect citations as they render
 */

import {
  createContext,
  useContext,
  useState,
  useCallback,
  useEffect,
  type ReactNode,
} from "react";
import ReactMarkdown from "react-markdown";
import remarkBreaks from "remark-breaks";
import remarkGfm from "remark-gfm";
import { ChevronDown, ChevronRight, FileText, Download, ExternalLink } from "lucide-react";
import {
  Button,
  Card,
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  ScrollArea,
  Skeleton,
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "./ui";
import { useAuth } from "./auth";
import type { Citation, CiteProps } from "@/types/rag";
import { cn } from "@/lib/utils";

// =============================================================================
// Citation Context - Collects citations as they render
// =============================================================================

interface CitationEntry {
  id: string;
  index: number;
  title: string;
  page?: number;
  lines?: [number, number];
  download?: string;
}

interface CitationContextType {
  citations: Map<string, CitationEntry>;
  addCitation: (entry: CitationEntry) => void;
  clearCitations: () => void;
  onViewCitation: (id: string) => void;
}

const CitationContext = createContext<CitationContextType | undefined>(undefined);

interface CitationProviderProps {
  children: ReactNode;
  onViewCitation: (id: string) => void;
}

export function CitationProvider({ children, onViewCitation }: CitationProviderProps) {
  const [citations, setCitations] = useState<Map<string, CitationEntry>>(new Map());

  const addCitation = useCallback((entry: CitationEntry) => {
    setCitations((prev) => {
      const next = new Map(prev);
      next.set(entry.id, entry);
      return next;
    });
  }, []);

  const clearCitations = useCallback(() => {
    setCitations(new Map());
  }, []);

  return (
    <CitationContext.Provider value={{ citations, addCitation, clearCitations, onViewCitation }}>
      {children}
    </CitationContext.Provider>
  );
}

function useCitationContext() {
  const context = useContext(CitationContext);
  if (!context) {
    throw new Error("useCitationContext must be used within CitationProvider");
  }
  return context;
}

// =============================================================================
// CiteRenderer - Renders <cite> tags as inline citation badges
// =============================================================================

function CiteRenderer(props: CiteProps) {
  const { id, title, page, lines, download, children } = props;
  const { addCitation, onViewCitation } = useCitationContext();
  const index = typeof children === "string" ? parseInt(children, 10) : 0;

  // If no id, render children as-is (fallback for malformed tags)
  if (!id) {
    return <>{children}</>;
  }

  // Register this citation when it renders
  useEffect(() => {
    const parsedLines = lines
      ? (lines.split("-").map(Number) as [number, number])
      : undefined;

    addCitation({
      id,
      index,
      title: title || "Source",
      page: page ? parseInt(page, 10) : undefined,
      lines: parsedLines,
      download,
    });
  }, [id, index, title, page, lines, download, addCitation]);

  const displayTitle = title || "Source";

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <button
            onClick={() => onViewCitation(id)}
            className="inline-flex items-center justify-center min-w-[1.25rem] h-5 px-1 text-xs font-medium text-ncl-blue bg-ncl-blue/10 rounded hover:bg-ncl-blue/20 transition-colors cursor-pointer align-super"
            aria-label={`View source: ${displayTitle}`}
          >
            {children}
          </button>
        </TooltipTrigger>
        <TooltipContent>
          <p className="text-sm">{displayTitle}</p>
          {page && <p className="text-xs text-muted-foreground">Page {page}</p>}
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

// Export the markdown tag renderers for CopilotChat
// Using 'as const' and explicit typing to satisfy CopilotKit's ComponentsMap
export const sourceTagRenderers: Record<string, React.FC<Record<string, unknown>>> = {
  cite: CiteRenderer as React.FC<Record<string, unknown>>,
};

// =============================================================================
// SourcesAccordion - Collapsible list of sources
// =============================================================================

interface SourcesAccordionProps {
  className?: string;
}

export function SourcesAccordion({ className }: SourcesAccordionProps) {
  const { citations, onViewCitation } = useCitationContext();
  const [isOpen, setIsOpen] = useState(false);

  // Convert map to sorted array
  const citationList = Array.from(citations.values()).sort((a, b) => a.index - b.index);

  if (citationList.length === 0) {
    return null;
  }

  return (
    <Card className={cn("overflow-hidden", className)}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between px-4 py-3 text-left hover:bg-ncl-gray-light/20 transition-colors"
      >
        <div className="flex items-center gap-2">
          <FileText className="h-4 w-4 text-ncl-blue" />
          <span className="font-medium text-sm text-ncl-blue-dark">
            Sources ({citationList.length})
          </span>
        </div>
        {isOpen ? (
          <ChevronDown className="h-4 w-4 text-ncl-gray" />
        ) : (
          <ChevronRight className="h-4 w-4 text-ncl-gray" />
        )}
      </button>

      {isOpen && (
        <div className="border-t border-ncl-gray-light">
          <ScrollArea className="max-h-64">
            <div className="p-2 space-y-1">
              {citationList.map((citation) => (
                <SourceItem
                  key={citation.id}
                  citation={citation}
                  onView={() => onViewCitation(citation.id)}
                />
              ))}
            </div>
          </ScrollArea>
        </div>
      )}
    </Card>
  );
}

interface SourceItemProps {
  citation: CitationEntry;
  onView: () => void;
}

function SourceItem({ citation, onView }: SourceItemProps) {
  const locationParts: string[] = [];
  if (citation.page) locationParts.push(`p.${citation.page}`);
  if (citation.lines) locationParts.push(`lines ${citation.lines[0]}-${citation.lines[1]}`);

  return (
    <div className="flex items-center justify-between px-3 py-2 rounded-md hover:bg-ncl-gray-light/30 group">
      <div className="flex items-center gap-3 min-w-0 flex-1">
        <span className="flex-shrink-0 w-6 h-6 rounded-full bg-ncl-blue/10 text-ncl-blue text-xs font-medium flex items-center justify-center">
          {citation.index}
        </span>
        <div className="min-w-0 flex-1">
          <p className="text-sm font-medium text-ncl-blue-dark truncate">
            {citation.title}
          </p>
          {locationParts.length > 0 && (
            <p className="text-xs text-ncl-gray">{locationParts.join(" | ")}</p>
          )}
        </div>
      </div>

      <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
        <Button
          variant="ghost"
          size="sm"
          onClick={onView}
          className="h-8 px-2"
          title="View source"
        >
          <ExternalLink className="h-4 w-4" />
        </Button>
        {citation.download && (
          <a
            href={`/api/archive/${citation.download}`}
            download
            className="inline-flex items-center justify-center h-8 px-2 rounded-md text-sm font-medium hover:bg-ncl-gray-light/20 transition-colors"
            title="Download original"
          >
            <Download className="h-4 w-4" />
          </a>
        )}
      </div>
    </div>
  );
}

// =============================================================================
// SourceViewDialog - Full dialog to view source content
// =============================================================================

interface SourceViewDialogProps {
  chunkId: string | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function SourceViewDialog({ chunkId, open, onOpenChange }: SourceViewDialogProps) {
  const { session } = useAuth();
  const [citation, setCitation] = useState<Citation | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!open || !chunkId || !session?.access_token) {
      setCitation(null);
      setError(null);
      return;
    }

    const fetchCitation = async () => {
      setLoading(true);
      setError(null);

      try {
        const response = await fetch(`/api/citations/${chunkId}`, {
          headers: {
            Authorization: `Bearer ${session.access_token}`,
          },
        });

        if (!response.ok) {
          throw new Error(`Failed to fetch: ${response.statusText}`);
        }

        const data = await response.json();
        setCitation(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load content");
      } finally {
        setLoading(false);
      }
    };

    fetchCitation();
  }, [open, chunkId, session]);

  const locationParts: string[] = [];
  if (citation?.page) locationParts.push(`Page ${citation.page}`);
  if (citation?.lines) locationParts.push(`Lines ${citation.lines[0]}-${citation.lines[1]}`);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl h-[80vh] flex flex-col overflow-hidden">
        <DialogHeader className="flex-shrink-0">
          <div className="flex items-start justify-between gap-4 pr-8">
            <div>
              <DialogTitle className="text-lg">
                {citation?.source_title || "Source Document"}
              </DialogTitle>
              {locationParts.length > 0 && (
                <p className="text-sm text-ncl-gray mt-1">{locationParts.join(" | ")}</p>
              )}
            </div>
            {citation?.archive_download_uri && (
              <a
                href={`/api/archive/${citation.archive_download_uri}`}
                download
                className="inline-flex items-center justify-center h-9 px-3 rounded-md border border-ncl-gray-light bg-white text-sm font-medium hover:bg-ncl-gray-light/20 transition-colors"
              >
                <Download className="h-4 w-4 mr-2" />
                Download
              </a>
            )}
          </div>
        </DialogHeader>

        <ScrollArea className="flex-1 min-h-0 mt-4">
          <div className="pr-4 pb-4">
            {loading && (
              <div className="space-y-3">
                <Skeleton className="h-4 w-full" />
                <Skeleton className="h-4 w-3/4" />
                <Skeleton className="h-4 w-5/6" />
                <Skeleton className="h-4 w-2/3" />
              </div>
            )}

            {error && (
              <div className="text-red-600 bg-red-50 p-4 rounded-lg">{error}</div>
            )}

            {citation?.content && !loading && (
              <div className="prose prose-sm max-w-none prose-headings:text-ncl-blue-dark prose-p:text-ncl-blue-dark prose-li:text-ncl-blue-dark prose-a:text-ncl-blue prose-a:underline prose-hr:my-4 [&_p]:mb-4 [&_br+br]:block [&_br+br]:h-2">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm, remarkBreaks]}
                  components={{
                    // Rewrite relative URLs to use archive API path
                    a: ({ href, children, ...props }) => {
                      let resolvedHref = href || "";

                      // Fix malformed mailto links (e.g., "foo@bar.commailto:foo@bar.com")
                      if (resolvedHref.includes("mailto:") && !resolvedHref.startsWith("mailto:")) {
                        const mailtoIndex = resolvedHref.indexOf("mailto:");
                        resolvedHref = resolvedHref.slice(mailtoIndex);
                      }

                      // Skip mailto: and absolute URLs
                      if (!resolvedHref.startsWith("mailto:") && !resolvedHref.startsWith("http")) {
                        // Check if href is already an absolute archive path (starts with folder_id pattern)
                        // folder_id is 16 hex chars like "9c6aae7aa8c0b9a9"
                        const isAbsoluteArchivePath = /^[a-f0-9]{16}\//.test(resolvedHref);

                        if (isAbsoluteArchivePath) {
                          // Already absolute from archive root - just prepend API path
                          resolvedHref = `/api/archive/${resolvedHref}`;
                        } else {
                          // Relative path - resolve from current document's directory
                          const basePath = citation.archive_browse_uri
                            ? citation.archive_browse_uri
                                .replace(/^\/archive/, "")  // Strip leading /archive
                                .replace(/\/[^/]+$/, "")     // Get directory path
                            : "";
                          resolvedHref = `/api/archive${basePath}/${resolvedHref}`;
                        }
                      }

                      return (
                        <a href={resolvedHref} target="_blank" rel="noopener noreferrer" {...props}>
                          {children}
                        </a>
                      );
                    },
                    // Rewrite relative image URLs
                    img: ({ src, alt, ...props }) => {
                      let resolvedSrc = src || "";

                      if (!resolvedSrc.startsWith("http")) {
                        // Check if src is already an absolute archive path (starts with folder_id pattern)
                        const isAbsoluteArchivePath = /^[a-f0-9]{16}\//.test(resolvedSrc);

                        if (isAbsoluteArchivePath) {
                          // Already absolute from archive root - just prepend API path
                          resolvedSrc = `/api/archive/${resolvedSrc}`;
                        } else {
                          // Relative path - resolve from current document's directory
                          const basePath = citation.archive_browse_uri
                            ? citation.archive_browse_uri
                                .replace(/^\/archive/, "")  // Strip leading /archive
                                .replace(/\/[^/]+$/, "")     // Get directory path
                            : "";
                          resolvedSrc = `/api/archive${basePath}/${resolvedSrc}`;
                        }
                      }

                      return <img src={resolvedSrc} alt={alt || ""} {...props} />;
                    },
                  }}
                >
                  {citation.content}
                </ReactMarkdown>
              </div>
            )}

            {!citation?.content && !loading && !error && (
              <p className="text-ncl-gray text-sm">No content available for this source.</p>
            )}
          </div>
        </ScrollArea>
      </DialogContent>
    </Dialog>
  );
}
