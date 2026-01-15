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
  useRef,
  useMemo,
  type ReactNode,
} from "react";
import ReactMarkdown from "react-markdown";
import rehypeRaw from "rehype-raw";
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
import { getApiBaseUrl } from "@/lib/conversations";

// =============================================================================
// Citation Context - Collects citations as they render
// =============================================================================

/**
 * Reference to a specific chunk within a document.
 * Multiple chunks can belong to the same source document.
 */
interface ChunkRef {
  chunkId: string;
  page?: number;
  lines?: [number, number];
  download?: string;
}

/**
 * Citation entry representing a source document.
 * Multiple chunks from the same document are consolidated into one entry.
 */
interface CitationEntry {
  id: string;              // Primary chunk_id (first chunk from this document)
  index: number;           // Document index (shared across all chunks from same doc)
  title: string;
  titleLoading?: boolean;
  chunks: ChunkRef[];      // All chunks from this document
}

interface CitationContextType {
  citations: Map<string, CitationEntry>;
  addCitation: (entry: CitationEntry) => void;
  updateCitationTitle: (id: string, title: string) => void;
  clearCitations: () => void;
  onViewCitation: (id: string, linesToHighlight?: [number, number][]) => void;
}

const CitationContext = createContext<CitationContextType | undefined>(undefined);

interface CitationProviderProps {
  children: ReactNode;
  onViewCitation: (id: string, linesToHighlight?: [number, number][]) => void;
}

export function CitationProvider({ children, onViewCitation }: CitationProviderProps) {
  const [citations, setCitations] = useState<Map<string, CitationEntry>>(new Map());

  const addCitation = useCallback((entry: CitationEntry) => {
    setCitations((prev) => {
      const next = new Map(prev);

      // Find existing entry with same title to merge chunks
      const existingEntry = Array.from(next.values()).find(
        (e) => e.title === entry.title && e.title !== "Source"
      );

      if (existingEntry) {
        // Merge chunks if not already present
        const newChunk = entry.chunks[0];
        if (newChunk && !existingEntry.chunks.some((c) => c.chunkId === newChunk.chunkId)) {
          existingEntry.chunks.push(newChunk);
        }
        // Keep existing entry (already in map by its id)
      } else {
        // New document - add to map
        next.set(entry.id, entry);
      }

      return next;
    });
  }, []);

  const updateCitationTitle = useCallback((id: string, title: string) => {
    setCitations((prev) => {
      const existing = prev.get(id);
      if (!existing) return prev;
      const next = new Map(prev);
      next.set(id, { ...existing, title, titleLoading: false });
      return next;
    });
  }, []);

  const clearCitations = useCallback(() => {
    setCitations(new Map());
  }, []);

  return (
    <CitationContext.Provider value={{ citations, addCitation, updateCitationTitle, clearCitations, onViewCitation }}>
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

// Export the hook for use in custom AssistantMessage
export { useCitationContext };

// =============================================================================
// MessageCitationProvider - Per-message citation context
// =============================================================================

interface MessageCitationProviderProps {
  children: ReactNode;
  onViewCitation: (id: string, linesToHighlight?: [number, number][]) => void;
}

/**
 * Per-message citation context that collects citations only for a single message.
 * Each assistant message gets its own instance, isolating citations per response.
 */
export function MessageCitationProvider({ children, onViewCitation }: MessageCitationProviderProps) {
  const [citations, setCitations] = useState<Map<string, CitationEntry>>(new Map());

  const addCitation = useCallback((entry: CitationEntry) => {
    setCitations((prev) => {
      const next = new Map(prev);

      // Find existing entry with same title to merge chunks
      const existingEntry = Array.from(next.values()).find(
        (e) => e.title === entry.title && e.title !== "Source"
      );

      if (existingEntry) {
        // Merge chunks if not already present
        const newChunk = entry.chunks[0];
        if (newChunk && !existingEntry.chunks.some((c) => c.chunkId === newChunk.chunkId)) {
          existingEntry.chunks.push(newChunk);
        }
        // Keep existing entry (already in map by its id)
      } else {
        // New document - add to map
        next.set(entry.id, entry);
      }

      return next;
    });
  }, []);

  const updateCitationTitle = useCallback((id: string, title: string) => {
    setCitations((prev) => {
      const existing = prev.get(id);
      if (!existing) return prev;
      const next = new Map(prev);
      next.set(id, { ...existing, title, titleLoading: false });
      return next;
    });
  }, []);

  const clearCitations = useCallback(() => {
    setCitations(new Map());
  }, []);

  return (
    <CitationContext.Provider value={{ citations, addCitation, updateCitationTitle, clearCitations, onViewCitation }}>
      {children}
    </CitationContext.Provider>
  );
}

// =============================================================================
// CiteRenderer - Renders <cite> tags as inline citation badges
// =============================================================================

interface ImgCiteProps {
  src?: string;
  id?: string;  // Optional chunk ID for opening citation dialog
}

/**
 * ImgCiteRenderer - Renders <img-cite> tags as inline images.
 *
 * The agent extracts image URLs from the content (e.g., markdown attachment links)
 * and uses <img-cite src="path/to/image.jpg" id="chunk_id" /> to embed images inline.
 *
 * This component takes the src directly and renders the image via the archive API.
 * If an id is provided, clicking the image opens the citation dialog.
 */
function ImgCiteRenderer(props: ImgCiteProps) {
  const { src, id } = props;
  const { onViewCitation } = useCitationContext();

  if (!src) {
    return (
      <span className="inline-block text-xs text-ncl-gray bg-ncl-gray-light/30 px-2 py-1 rounded">
        [Image unavailable]
      </span>
    );
  }

  // Build the image URL - clean up various prefixes that might be included
  let cleanSrc = src;
  // Remove "img:" prefix if agent included it from context header
  cleanSrc = cleanSrc.replace(/^img:/, "");
  // Remove /archive/ prefix if present
  cleanSrc = cleanSrc.replace(/^\/archive\//, "");
  const imageUrl = `${getApiBaseUrl()}/archive/${cleanSrc}`;

  const handleClick = () => {
    if (id) {
      onViewCitation(id);
    } else {
      // If no citation ID, open image in new tab
      window.open(imageUrl, "_blank");
    }
  };

  return (
    <span
      role="button"
      tabIndex={0}
      onClick={handleClick}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          handleClick();
        }
      }}
      className="inline-block my-2 cursor-pointer rounded overflow-hidden border border-ncl-gray-light hover:border-ncl-blue transition-colors"
      title={id ? "Click to view source" : "Click to open full size"}
    >
      <img
        src={imageUrl}
        alt="Referenced image"
        className="max-w-full max-h-64 object-contain"
        loading="lazy"
      />
    </span>
  );
}

function CiteRenderer(props: CiteProps) {
  const { id, title, page, lines, download, children } = props;
  const { citations, addCitation, updateCitationTitle, onViewCitation } = useCitationContext();
  const { session } = useAuth();

  // Extract index from children - ReactMarkdown may pass string, number, or array
  let index = 0;
  if (typeof children === "string") {
    index = parseInt(children, 10) || 0;
  } else if (typeof children === "number") {
    index = children;
  } else if (Array.isArray(children) && children.length > 0) {
    const first = children[0];
    if (typeof first === "string") {
      index = parseInt(first, 10) || 0;
    } else if (typeof first === "number") {
      index = first;
    }
  }

  // If no id, render children as-is (fallback for malformed tags)
  if (!id) {
    return <>{children}</>;
  }

  // Track if title needs fetching
  const needsTitleFetch = !title;

  // Memoize parsed lines to prevent infinite re-render loop
  const parsedLines = useMemo(
    () => (lines ? (lines.split("-").map(Number) as [number, number]) : undefined),
    [lines]
  );

  // Register this citation when it renders
  useEffect(() => {
    const chunkRef: ChunkRef = {
      chunkId: id,
      page: page ? parseInt(page, 10) : undefined,
      lines: parsedLines,
      download,
    };

    addCitation({
      id,
      index,
      title: title || "Source",
      titleLoading: needsTitleFetch,
      chunks: [chunkRef],
    });
  }, [id, index, title, page, parsedLines, download, addCitation, needsTitleFetch]);

  // Fetch title from API if not provided in the cite tag
  useEffect(() => {
    if (!needsTitleFetch || !session?.access_token) return;

    const fetchTitle = async () => {
      try {
        const response = await fetch(`${getApiBaseUrl()}/citations/${id}`, {
          headers: {
            Authorization: `Bearer ${session.access_token}`,
          },
        });
        if (response.ok) {
          const data = await response.json();
          if (data.source_title) {
            updateCitationTitle(id, data.source_title);
          }
        }
      } catch {
        // Silently fail - "Source" fallback is acceptable
      }
    };

    fetchTitle();
  }, [id, needsTitleFetch, session?.access_token, updateCitationTitle]);

  // Get title from context (updates when fetched) or fall back to prop/default
  const citationEntry = id ? citations.get(id) : undefined;
  const displayTitle = citationEntry?.title || title || "Source";

  // Use a span-based tooltip to avoid hydration errors when citation is inside <p>
  // The tooltip content uses Portal so it won't cause nesting issues
  return (
    <TooltipProvider delayDuration={300}>
      <Tooltip>
        <TooltipTrigger asChild>
          <span
            role="button"
            tabIndex={0}
            onClick={() => onViewCitation(id, parsedLines ? [parsedLines] : undefined)}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                onViewCitation(id, parsedLines ? [parsedLines] : undefined);
              }
            }}
            className="inline-flex items-center justify-center min-w-[1.25rem] h-5 px-1 mr-0.5 text-xs font-medium text-ncl-blue bg-ncl-blue/10 rounded hover:bg-ncl-blue/20 transition-colors cursor-pointer align-super"
            aria-label={`View source: ${displayTitle}`}
          >
            {children}
          </span>
        </TooltipTrigger>
        <TooltipContent side="top" sideOffset={4}>
          <span className="text-sm">{displayTitle}</span>
          {page && <span className="text-xs text-muted-foreground ml-2">Page {page}</span>}
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

// Export the markdown tag renderers for CopilotChat
// Using 'as const' and explicit typing to satisfy CopilotKit's ComponentsMap
export const sourceTagRenderers: Record<string, React.FC<Record<string, unknown>>> = {
  cite: CiteRenderer as React.FC<Record<string, unknown>>,
  "img-cite": ImgCiteRenderer as React.FC<Record<string, unknown>>,
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

  // Convert map to sorted array (already consolidated by document)
  // Sort by original index to maintain order of first appearance
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
              {citationList.map((citation, idx) => {
                // Collect all line ranges from this document's chunks for highlighting
                const allLineRanges = citation.chunks
                  .filter((c) => c.lines)
                  .map((c) => c.lines as [number, number]);

                // Use sequential display index (1, 2, 3...) for clean numbering
                const displayIndex = idx + 1;

                return (
                  <SourceItem
                    key={citation.id}
                    citation={citation}
                    displayIndex={displayIndex}
                    onView={() => onViewCitation(citation.id, allLineRanges.length > 0 ? allLineRanges : undefined)}
                  />
                );
              })}
            </div>
          </ScrollArea>
        </div>
      )}
    </Card>
  );
}

interface SourceItemProps {
  citation: CitationEntry;
  displayIndex: number;
  onView: () => void;
}

function SourceItem({ citation, displayIndex, onView }: SourceItemProps) {
  // Build location info from chunks
  const pageNumbers = citation.chunks.map((c) => c.page).filter((p): p is number => p !== undefined);
  const pages = Array.from(new Set(pageNumbers));
  const locationParts: string[] = [];

  if (pages.length === 1) {
    locationParts.push(`p.${pages[0]}`);
  } else if (pages.length > 1) {
    locationParts.push(`p.${pages.sort((a, b) => a - b).join(", ")}`);
  }

  return (
    <div className="flex items-center justify-between px-3 py-2 rounded-md hover:bg-ncl-gray-light/30 group">
      <div className="flex items-center gap-3 min-w-0 flex-1">
        <span className="flex-shrink-0 w-6 h-6 rounded-full bg-ncl-blue/10 text-ncl-blue text-xs font-medium flex items-center justify-center">
          {displayIndex}
        </span>
        <div className="min-w-0 flex-1">
          {citation.titleLoading ? (
            <Skeleton className="h-4 w-48" />
          ) : (
            <p className="text-sm font-medium text-ncl-blue-dark truncate">
              {citation.title}
            </p>
          )}
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
        {citation.chunks[0]?.download && (
          <a
            href={`${getApiBaseUrl()}/archive/${citation.chunks[0].download}`}
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
  linesToHighlight?: [number, number][];
}

/**
 * Add highlight markers to markdown content for specified line ranges.
 * Wraps lines in <mark> tags that will be rendered with yellow background.
 */
function addLineHighlights(content: string, lineRanges: [number, number][]): string {
  if (!lineRanges || lineRanges.length === 0) return content;

  const lines = content.split("\n");
  const highlightedLineNums = new Set<number>();

  // Collect all line numbers to highlight
  for (const [start, end] of lineRanges) {
    for (let i = start; i <= end && i <= lines.length; i++) {
      highlightedLineNums.add(i);
    }
  }

  // Wrap highlighted lines
  const result = lines.map((line, idx) => {
    const lineNum = idx + 1; // Lines are 1-indexed
    if (highlightedLineNums.has(lineNum)) {
      // Use a special marker that we'll style with CSS
      return `<mark class="citation-highlight">${line}</mark>`;
    }
    return line;
  });

  return result.join("\n");
}

export function SourceViewDialog({ chunkId, open, onOpenChange, linesToHighlight }: SourceViewDialogProps) {
  const { session } = useAuth();
  const [citation, setCitation] = useState<Citation | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const contentRef = useRef<HTMLDivElement>(null);

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
        const response = await fetch(`${getApiBaseUrl()}/citations/${chunkId}`, {
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

  // Scroll to first highlight after content loads
  useEffect(() => {
    if (!loading && citation?.content && linesToHighlight && linesToHighlight.length > 0) {
      // Small delay to ensure DOM is ready
      const timer = setTimeout(() => {
        const firstHighlight = contentRef.current?.querySelector(".citation-highlight");
        if (firstHighlight) {
          firstHighlight.scrollIntoView({ behavior: "smooth", block: "center" });
        }
      }, 100);
      return () => clearTimeout(timer);
    }
  }, [loading, citation?.content, linesToHighlight]);

  const locationParts: string[] = [];
  if (citation?.page) locationParts.push(`Page ${citation.page}`);

  // Show highlighted line info
  if (linesToHighlight && linesToHighlight.length > 0) {
    if (linesToHighlight.length === 1) {
      const [start, end] = linesToHighlight[0];
      locationParts.push(`Lines ${start}-${end} highlighted`);
    } else {
      locationParts.push(`${linesToHighlight.length} sections highlighted`);
    }
  } else if (citation?.lines) {
    locationParts.push(`Lines ${citation.lines[0]}-${citation.lines[1]}`);
  }

  // Process content with highlights
  const processedContent = citation?.content && linesToHighlight
    ? addLineHighlights(citation.content, linesToHighlight)
    : citation?.content;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl h-[80vh] flex flex-col overflow-hidden" aria-describedby={undefined}>
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
                href={`${getApiBaseUrl()}/archive/${citation.archive_download_uri}`}
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
          <div ref={contentRef} className="pr-4 pb-4">
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

            {processedContent && !loading && (
              <div className="prose prose-sm max-w-none prose-headings:text-ncl-blue-dark prose-p:text-ncl-blue-dark prose-li:text-ncl-blue-dark prose-a:text-ncl-blue prose-a:underline prose-hr:my-4 [&_p]:mb-4 [&_br+br]:block [&_br+br]:h-2 [&_.citation-highlight]:bg-yellow-200 [&_.citation-highlight]:px-0.5 [&_.citation-highlight]:rounded">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm, remarkBreaks]}
                  // eslint-disable-next-line @typescript-eslint/no-explicit-any
                  rehypePlugins={[rehypeRaw as any]}
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
                          resolvedHref = `${getApiBaseUrl()}/archive/${resolvedHref}`;
                        } else {
                          // Relative path - resolve from current document's directory
                          const basePath = citation?.archive_browse_uri
                            ? citation.archive_browse_uri
                                .replace(/^\/archive/, "")  // Strip leading /archive
                                .replace(/\/[^/]+$/, "")     // Get directory path
                            : "";
                          resolvedHref = `${getApiBaseUrl()}/archive${basePath}/${resolvedHref}`;
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
                          resolvedSrc = `${getApiBaseUrl()}/archive/${resolvedSrc}`;
                        } else {
                          // Relative path - resolve from current document's directory
                          const basePath = citation?.archive_browse_uri
                            ? citation.archive_browse_uri
                                .replace(/^\/archive/, "")  // Strip leading /archive
                                .replace(/\/[^/]+$/, "")     // Get directory path
                            : "";
                          resolvedSrc = `${getApiBaseUrl()}/archive${basePath}/${resolvedSrc}`;
                        }
                      }

                      return <img src={resolvedSrc} alt={alt || ""} {...props} />;
                    },
                  }}
                >
                  {processedContent}
                </ReactMarkdown>
              </div>
            )}

            {!processedContent && !loading && !error && (
              <p className="text-ncl-gray text-sm">No content available for this source.</p>
            )}
          </div>
        </ScrollArea>
      </DialogContent>
    </Dialog>
  );
}
