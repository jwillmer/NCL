/**
 * Sources components for displaying citations in the chat UI.
 *
 * - CiteRenderer: Renders <cite> tags as interactive inline badges
 * - SourcesAccordion: Collapsible list of all sources in the current message
 * - SourceViewDialog: Full-screen dialog to view source content
 * - CitationProvider / MessageCitationProvider: Context to collect citations
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
import rehypeSanitize, { defaultSchema } from "rehype-sanitize";
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

// Allow <mark> tags through sanitizer (used for line highlighting)
const sourceSanitizeSchema = {
  ...defaultSchema,
  tagNames: [...(defaultSchema.tagNames || []), "mark"],
  attributes: { ...defaultSchema.attributes, mark: ["className", "class"] },
};

/** Strip leading /archive/ prefix to avoid doubling. */
function stripArchivePrefix(uri: string): string {
  return uri.replace(/^\/archive\//, "");
}

/** Resolve a relative or absolute archive URL to a full API path.
 *
 * Archive folder IDs are 32-char MD5 hashes (compute_folder_id). The old
 * regex only matched 16-char prefixes, so any href already in
 * ``<folder>/attachments/file`` form fell through to the "relative" branch
 * and got the folder prefix stapled on twice. Accept any leading hex run of
 * 16+ chars followed by ``/`` as "already absolute".
 */
function resolveArchiveUrl(href: string, browseUri?: string | null): string {
  const isAbsoluteArchivePath = /^[a-f0-9]{16,}\//.test(href);
  if (isAbsoluteArchivePath) {
    return `${getApiBaseUrl()}/archive/${href}`;
  }
  const basePath = browseUri
    ? browseUri.replace(/^\/archive/, "").replace(/\/[^/]+$/, "")
    : "";
  return `${getApiBaseUrl()}/archive${basePath}/${href}`;
}

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

/**
 * Shared citation provider implementation — used by both CitationProvider (global)
 * and MessageCitationProvider (per-message, isolates citations per response).
 */
function CitationProviderImpl({ children, onViewCitation }: CitationProviderProps) {
  const [citations, setCitations] = useState<Map<string, CitationEntry>>(new Map());

  const addCitation = useCallback((entry: CitationEntry) => {
    setCitations((prev) => {
      const next = new Map(prev);
      const existingEntry = Array.from(next.values()).find(
        (e) => e.title === entry.title && e.title !== "Source"
      );
      if (existingEntry) {
        const newChunk = entry.chunks[0];
        if (newChunk && !existingEntry.chunks.some((c) => c.chunkId === newChunk.chunkId)) {
          existingEntry.chunks.push(newChunk);
        }
      } else {
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

  const clearCitations = useCallback(() => setCitations(new Map()), []);

  return (
    <CitationContext.Provider value={{ citations, addCitation, updateCitationTitle, clearCitations, onViewCitation }}>
      {children}
    </CitationContext.Provider>
  );
}

/** Global citation provider (wraps the whole chat). */
export function CitationProvider(props: CitationProviderProps) {
  return <CitationProviderImpl {...props} />;
}

/** Per-message citation provider (isolates citations per assistant response). */
export function MessageCitationProvider(props: CitationProviderProps) {
  return <CitationProviderImpl {...props} />;
}

export function useCitationContext() {
  const context = useContext(CitationContext);
  if (!context) throw new Error("useCitationContext must be used within CitationProvider");
  return context;
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
      <span className="inline-block text-xs text-MTSS-gray bg-MTSS-gray-light/30 px-2 py-1 rounded">
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
      className="inline-block my-2 cursor-pointer rounded overflow-hidden border border-MTSS-gray-light hover:border-MTSS-blue transition-colors"
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
            className="inline-flex items-center justify-center min-w-[1.25rem] h-5 px-1 mr-0.5 text-xs font-medium text-MTSS-blue bg-MTSS-blue/10 rounded hover:bg-MTSS-blue/20 transition-colors cursor-pointer align-super"
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
        className="w-full flex items-center justify-between px-4 py-3 text-left hover:bg-MTSS-gray-light/20 transition-colors"
      >
        <div className="flex items-center gap-2">
          <FileText className="h-4 w-4 text-MTSS-blue" />
          <span className="font-medium text-sm text-MTSS-blue-dark">
            Sources ({citationList.length})
          </span>
        </div>
        {isOpen ? (
          <ChevronDown className="h-4 w-4 text-MTSS-gray" />
        ) : (
          <ChevronRight className="h-4 w-4 text-MTSS-gray" />
        )}
      </button>

      {isOpen && (
        <div className="border-t border-MTSS-gray-light">
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
    <div className="flex items-center justify-between px-3 py-2 rounded-md hover:bg-MTSS-gray-light/30 group">
      {/* Whole row (everything except the download button) is the dialog
          trigger — clicking the title is the natural move. `button` so
          keyboard users get focus + Enter/Space behavior for free. */}
      <button
        type="button"
        onClick={onView}
        className="flex items-center gap-3 min-w-0 flex-1 text-left cursor-pointer focus:outline-none focus-visible:ring-2 focus-visible:ring-MTSS-blue/40 rounded"
        title="View source"
      >
        <span className="flex-shrink-0 w-6 h-6 rounded-full bg-MTSS-blue/10 text-MTSS-blue text-xs font-medium flex items-center justify-center">
          {displayIndex}
        </span>
        <div className="min-w-0 flex-1">
          {citation.titleLoading ? (
            <Skeleton className="h-4 w-48" />
          ) : (
            <p className="text-sm font-medium text-MTSS-blue-dark truncate group-hover:underline">
              {citation.title}
            </p>
          )}
          {locationParts.length > 0 && (
            <p className="text-xs text-MTSS-gray">{locationParts.join(" | ")}</p>
          )}
        </div>
      </button>

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
            href={`${getApiBaseUrl()}/archive/${stripArchivePrefix(citation.chunks[0].download)}`}
            download
            className="inline-flex items-center justify-center h-8 px-2 rounded-md text-sm font-medium hover:bg-MTSS-gray-light/20 transition-colors"
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
                <p className="text-sm text-MTSS-gray mt-1">{locationParts.join(" | ")}</p>
              )}
            </div>
            {citation?.archive_download_uri && (
              <a
                href={
                  citation.archive_download_signed_url
                  ?? `${getApiBaseUrl()}/archive/${stripArchivePrefix(citation.archive_download_uri)}`
                }
                download
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center justify-center h-9 px-3 rounded-md border border-MTSS-gray-light bg-white text-sm font-medium hover:bg-MTSS-gray-light/20 transition-colors"
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
              <div className="prose prose-sm max-w-none prose-headings:text-MTSS-blue-dark prose-p:text-MTSS-blue-dark prose-li:text-MTSS-blue-dark prose-a:text-MTSS-blue prose-a:underline prose-hr:my-4 [&_p]:mb-4 [&_br+br]:block [&_br+br]:h-2 [&_.citation-highlight]:bg-yellow-200 [&_.citation-highlight]:px-0.5 [&_.citation-highlight]:rounded">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm, remarkBreaks]}
                  // eslint-disable-next-line @typescript-eslint/no-explicit-any
                  rehypePlugins={[rehypeRaw as any, [rehypeSanitize, sourceSanitizeSchema]]}
                  components={{
                    a: ({ href, children, ...props }) => {
                      let resolved = href || "";
                      // Fix malformed mailto links
                      if (resolved.includes("mailto:") && !resolved.startsWith("mailto:")) {
                        resolved = resolved.slice(resolved.indexOf("mailto:"));
                      }
                      if (!resolved.startsWith("mailto:") && !resolved.startsWith("http")) {
                        resolved = resolveArchiveUrl(resolved, citation?.archive_browse_uri);
                      }
                      // "Download Original" inside the rendered .md points at
                      // the auth-protected /api/archive path; a new-tab click
                      // can't send the Bearer header and would 401. If the
                      // resolved URL matches the citation's download_uri, swap
                      // in the pre-signed URL so the browser can follow it.
                      if (
                        citation?.archive_download_signed_url &&
                        citation.archive_download_uri &&
                        resolved.endsWith(stripArchivePrefix(citation.archive_download_uri))
                      ) {
                        resolved = citation.archive_download_signed_url;
                      }
                      return <a href={resolved} target="_blank" rel="noopener noreferrer" {...props}>{children}</a>;
                    },
                    img: ({ src, alt, ...props }) => {
                      let resolved = src || "";
                      if (!resolved.startsWith("http")) {
                        resolved = resolveArchiveUrl(resolved, citation?.archive_browse_uri);
                      }
                      return <img src={resolved} alt={alt || ""} {...props} />;
                    },
                  }}
                >
                  {processedContent}
                </ReactMarkdown>
              </div>
            )}

            {!processedContent && !loading && !error && (
              <p className="text-MTSS-gray text-sm">No content available for this source.</p>
            )}
          </div>
        </ScrollArea>
      </DialogContent>
    </Dialog>
  );
}
