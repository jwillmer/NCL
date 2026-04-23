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
  type MouseEvent as ReactMouseEvent,
  type ReactNode,
} from "react";
import ReactMarkdown from "react-markdown";
import rehypeRaw from "rehype-raw";
import rehypeSanitize, { defaultSchema } from "rehype-sanitize";
import remarkBreaks from "remark-breaks";
import remarkGfm from "remark-gfm";
import { visit } from "unist-util-visit";
import type { Root as MdastRoot, Text as MdastText } from "mdast";
import { ChevronDown, ChevronRight, FileText, Download, ArrowLeft, Mail } from "lucide-react";
import {
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

// Allow <mark> (line highlighting), <abbr title="..."> (MIME tooltip) and
// the injected image preview's <img class="source-image-preview"> through
// the sanitizer.
const sourceSanitizeSchema = {
  ...defaultSchema,
  tagNames: [...(defaultSchema.tagNames || []), "mark", "abbr"],
  attributes: {
    ...defaultSchema.attributes,
    mark: ["className", "class"],
    abbr: ["title"],
    img: [...(defaultSchema.attributes?.img || []), "className", "class"],
  },
};

const IMAGE_EXTENSIONS = /\.(png|jpe?g|gif|webp|svg|bmp|tiff?)(?:\?|$)/i;

/** Hover-preview image that shows a spinner until the fetch resolves.
 *  Gives the tooltip a stable 240×180 footprint so the popover doesn't
 *  appear empty on the first hover of large archive images.
 *
 *  Implementation note: the <img> is always rendered visibly (opacity
 *  transitions in once loaded) — the earlier ``hidden`` + ``loading=lazy``
 *  version could block the browser from ever fetching the file in Chrome,
 *  leaving the spinner stuck forever. */
function ImageHoverPreview({ src }: { src: string }) {
  const [status, setStatus] = useState<"loading" | "loaded" | "error">("loading");
  return (
    <div className="relative w-[240px] h-[180px] flex items-center justify-center">
      <img
        src={src}
        alt=""
        onLoad={() => setStatus("loaded")}
        onError={() => setStatus("error")}
        className={cn(
          "max-w-[240px] max-h-[180px] object-contain transition-opacity duration-150",
          status === "loaded" ? "opacity-100" : "opacity-0",
        )}
      />
      {status === "loading" && (
        <div
          className="absolute h-6 w-6 rounded-full border-2 border-MTSS-gray-light border-t-MTSS-blue animate-spin"
          aria-label="Loading preview"
        />
      )}
      {status === "error" && (
        <span className="absolute text-xs text-MTSS-gray">Preview unavailable</span>
      )}
    </div>
  );
}

/** Rewrite legacy attachment-listing lines inside archive email markdown —
 *  ``- [name](file) ([View](file.md))`` → ``- name — [Download](file) · [Details](file.md)``.
 *  Idempotent (skips lines already in the new format). The generator emits
 *  the new form for fresh ingests; this handles the corpus already on disk
 *  without rewriting the .md files. */
function rewriteAttachmentListing(content: string): string {
  let out = content.replace(
    /^(\s*-\s+)\[([^\]]+)\]\(([^)]+)\)\s*\(\[View\]\(([^)]+)\)\)\s*$/gm,
    (_m, dash, name, href, viewHref) =>
      `${dash}${name} — [Download](${href}) · [Details](${viewHref})`,
  );
  out = out.replace(
    /^(\s*-\s+)\[([^\]]+)\]\(([^)]+)\)\s*$/gm,
    (match, dash, name, href) => {
      // Only rewrite attachment paths (heuristic: contains "/attachments/").
      // Leaves body prose / unrelated bullet lists alone.
      if (!String(href).includes("/attachments/")) return match;
      return `${dash}${name} — [Download](${href})`;
    },
  );
  return out;
}

/** If the citation is an image attachment, inject the image inline right
 *  before the "## Content" heading so the dialog reads metadata → image →
 *  description. Users asked for the image up-front since it communicates
 *  faster than the LLM-generated caption below it. */
function injectImagePreview(
  content: string,
  imageUrl: string | null | undefined,
  isImage: boolean,
): string {
  if (!isImage || !imageUrl) return content;
  if (content.includes("source-image-preview")) return content;
  const replacement = `<img src="${imageUrl}" alt="" class="source-image-preview" />\n\n## Content`;
  if (/^##\s+Content\s*$/m.test(content)) {
    return content.replace(/^##\s+Content\s*$/m, replacement);
  }
  // No Content heading (edge case) — append the image after the metadata
  // block instead.
  return `${content}\n\n<img src="${imageUrl}" alt="" class="source-image-preview" />`;
}

// Friendly labels for common attachment MIME types rendered in archive markdown
// under "**Type:** <mime>". Users see the friendly label; raw MIME stays on
// hover via <abbr title>. Keep labels descriptive (not acronyms) — maritime
// users wanted "Spreadsheet" over "XLSX".
const MIME_FRIENDLY_LABELS: Record<string, string> = {
  "application/pdf": "PDF document",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "Word document",
  "application/msword": "Word document (legacy)",
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "Spreadsheet",
  "application/vnd.ms-excel": "Spreadsheet (legacy)",
  "application/vnd.openxmlformats-officedocument.presentationml.presentation": "Presentation",
  "application/vnd.ms-powerpoint": "Presentation (legacy)",
  "application/vnd.oasis.opendocument.text": "OpenDocument text",
  "application/vnd.oasis.opendocument.spreadsheet": "OpenDocument spreadsheet",
  "application/vnd.oasis.opendocument.presentation": "OpenDocument presentation",
  "text/csv": "CSV spreadsheet",
  "text/html": "HTML page",
  "text/plain": "Text file",
  "text/rtf": "Rich-text document",
  "application/rtf": "Rich-text document",
  "application/zip": "ZIP archive",
  "application/x-zip-compressed": "ZIP archive",
  "application/x-zip": "ZIP archive",
  "application/epub+zip": "EPUB book",
  "application/json": "JSON data",
  "application/octet-stream": "Binary file",
  "message/rfc822": "Email message",
  "image/png": "PNG image",
  "image/x-png": "PNG image",
  "image/jpeg": "JPEG image",
  "image/jpg": "JPEG image",
  "image/gif": "GIF image",
  "image/webp": "WebP image",
  "image/tiff": "TIFF image",
  "image/bmp": "Bitmap image",
  "image/svg+xml": "SVG image",
};

function friendlyMimeLabel(mime: string): string {
  const trimmed = mime.trim();
  const exact = MIME_FRIENDLY_LABELS[trimmed];
  if (exact) return exact;
  // Family fallbacks for unmapped MIMEs
  if (trimmed.startsWith("image/")) return "Image";
  if (trimmed.startsWith("audio/")) return "Audio";
  if (trimmed.startsWith("video/")) return "Video";
  if (trimmed.startsWith("text/")) return "Text file";
  return trimmed;
}

/** Rewrite "**Type:** <mime>" lines in archive markdown to a friendly label
 *  with the raw MIME on hover (<abbr title>). Idempotent — skips lines that
 *  already contain an <abbr>. */
function friendlifyTypeLine(content: string): string {
  return content.replace(
    /^(\*\*Type:\*\*\s+)(.+)$/gm,
    (match, prefix, value) => {
      if (value.includes("<abbr")) return match;
      const mime = value.trim();
      const label = friendlyMimeLabel(mime);
      if (label === mime) return match;
      const safeMime = mime.replace(/"/g, "&quot;");
      return `${prefix}<abbr title="${safeMime}">${label}</abbr>`;
    },
  );
}

/** Strip leading /archive/ prefix to avoid doubling. */
function stripArchivePrefix(uri: string): string {
  return uri.replace(/^\/archive\//, "");
}

/** Append ``?token=<jwt>`` to same-origin ``/api/archive/**`` URLs so <img>
 *  and anchor-new-tab navigations (which can't carry an Authorization header)
 *  still authenticate. The AuthMiddleware accepts the query param as a
 *  fallback to the Bearer header. No-op when a token isn't available or the
 *  URL already carries one. */
function withArchiveToken(url: string, token: string | undefined): string {
  if (!token || !url) return url;
  if (!url.includes("/api/archive/")) return url;
  if (/[?&]token=/.test(url)) return url;
  const sep = url.includes("?") ? "&" : "?";
  return `${url}${sep}token=${encodeURIComponent(token)}`;
}

// -------------------------------------------------------------------------
// Archive markdown cache
//
// Same document reopened in one session shouldn't re-fetch. Small LRU-ish
// Map (insertion-order iteration + manual eviction of oldest key) capped
// at a modest entry count. The values are the raw fetched markdown text
// — preamble stripping happens at render time so we don't need to re-run
// the fetch if the stripping logic changes mid-session (it won't, but it
// keeps the cache semantically simple).
// -------------------------------------------------------------------------
const ARCHIVE_MD_CACHE_MAX = 20;
const archiveMdCache = new Map<string, string>();

function getCachedArchiveMd(url: string): string | undefined {
  const hit = archiveMdCache.get(url);
  if (hit !== undefined) {
    // Refresh recency — delete then reinsert so the key moves to the tail.
    archiveMdCache.delete(url);
    archiveMdCache.set(url, hit);
  }
  return hit;
}

function setCachedArchiveMd(url: string, content: string): void {
  if (archiveMdCache.has(url)) archiveMdCache.delete(url);
  archiveMdCache.set(url, content);
  while (archiveMdCache.size > ARCHIVE_MD_CACHE_MAX) {
    const oldest = archiveMdCache.keys().next().value;
    if (oldest === undefined) break;
    archiveMdCache.delete(oldest);
  }
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

/** Strip the ``/api/archive/`` (or ``/archive/``) origin prefix from a URL
 *  so only the bucket-relative path remains — the shape the backend stores
 *  on ``chunks.archive_browse_uri``. */
function toBucketRelative(url: string): string {
  try {
    const u = new URL(url, typeof window !== "undefined" ? window.location.origin : "http://x");
    let p = u.pathname;
    if (p.startsWith("/api/archive/")) p = p.slice("/api/archive/".length);
    else if (p.startsWith("/archive/")) p = p.slice("/archive/".length);
    return p;
  } catch {
    return url.replace(/^.*\/api\/archive\//, "").replace(/^.*\/archive\//, "");
  }
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
  doc?: string;            // Stable source id from the backend (dedupe key)
  index: number;           // Document index (shared across all chunks from same doc)
  title: string;
  titleLoading?: boolean;
  // Originating email subject so the accordion can show "from <email>" under
  // each source. `undefined` = not fetched yet; `null` = fetched, no origin
  // (e.g., the cite IS the email itself); string = subject.
  originEmailSubject?: string | null;
  // Discriminator so the accordion can badge emails with a mail icon and
  // show an attachment-count subtitle. `undefined` = not fetched yet.
  documentType?: string | null;
  attachmentCount?: number | null;
  chunks: ChunkRef[];      // All chunks from this document
}

interface CitationContextType {
  citations: Map<string, CitationEntry>;
  addCitation: (entry: CitationEntry) => void;
  updateCitationTitle: (id: string, title: string) => void;
  updateCitationOriginEmail: (id: string, subject: string | null) => void;
  updateCitationDocMeta: (
    id: string,
    documentType: string | null,
    attachmentCount: number | null,
  ) => void;
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
      // Dedupe priority: stable doc id from the backend > title fallback.
      // The title path is a fallback for legacy cites that haven't been
      // re-rendered since the `doc` attribute shipped — it still guards
      // against "Source" placeholders collapsing into one entry before
      // their real titles land.
      const existingEntry = Array.from(next.values()).find((e) => {
        if (entry.doc && e.doc) return e.doc === entry.doc;
        return e.title === entry.title && e.title !== "Source";
      });
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

  const updateCitationOriginEmail = useCallback((id: string, subject: string | null) => {
    setCitations((prev) => {
      // Match by primary id OR by any chunk id — CiteRenderer fetches per
      // cite-tag chunk, but dedupe may have rolled that chunk into another
      // entry's chunks[] under a different primary id.
      const entry =
        prev.get(id) ||
        Array.from(prev.values()).find((e) => e.chunks.some((c) => c.chunkId === id));
      if (!entry) return prev;
      if (entry.originEmailSubject !== undefined) return prev;
      const next = new Map(prev);
      next.set(entry.id, { ...entry, originEmailSubject: subject });
      return next;
    });
  }, []);

  const updateCitationDocMeta = useCallback(
    (id: string, documentType: string | null, attachmentCount: number | null) => {
      setCitations((prev) => {
        const entry =
          prev.get(id) ||
          Array.from(prev.values()).find((e) => e.chunks.some((c) => c.chunkId === id));
        if (!entry) return prev;
        if (entry.documentType !== undefined) return prev;
        const next = new Map(prev);
        next.set(entry.id, { ...entry, documentType, attachmentCount });
        return next;
      });
    },
    [],
  );

  const clearCitations = useCallback(() => setCitations(new Map()), []);

  return (
    <CitationContext.Provider
      value={{
        citations,
        addCitation,
        updateCitationTitle,
        updateCitationOriginEmail,
        updateCitationDocMeta,
        clearCitations,
        onViewCitation,
      }}
    >
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
  const { session } = useAuth();

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
  const imageUrl = withArchiveToken(
    `${getApiBaseUrl()}/archive/${cleanSrc}`,
    session?.access_token,
  );

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
  const { id, doc, title, page, lines, download, children } = props;
  const {
    citations,
    addCitation,
    updateCitationTitle,
    updateCitationOriginEmail,
    updateCitationDocMeta,
    onViewCitation,
  } = useCitationContext();
  const { session } = useAuth();
  // Guard against duplicate fetches when CiteRenderer re-renders (e.g. highlight
  // state, parent rerender). One detail fetch per chunk per mount.
  const detailsFetchedRef = useRef(false);

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
      doc,
      index,
      title: title || "Source",
      titleLoading: needsTitleFetch,
      chunks: [chunkRef],
    });
  }, [id, doc, index, title, page, parsedLines, download, addCitation, needsTitleFetch]);

  // Fetch citation details once per chunk: backfill title if the cite tag
  // omitted it, and always populate the origin-email subject so the accordion
  // can show "from <email>" under each source. Guarded by a ref so re-renders
  // don't trigger duplicate requests.
  useEffect(() => {
    if (!session?.access_token || detailsFetchedRef.current) return;
    detailsFetchedRef.current = true;

    const fetchDetails = async () => {
      try {
        const response = await fetch(`${getApiBaseUrl()}/citations/${id}`, {
          headers: {
            Authorization: `Bearer ${session.access_token}`,
          },
        });
        if (response.ok) {
          const data = await response.json();
          if (needsTitleFetch && data.source_title) {
            updateCitationTitle(id, data.source_title);
          }
          updateCitationOriginEmail(id, data.origin_email?.subject ?? null);
          updateCitationDocMeta(
            id,
            typeof data.document_type === "string" ? data.document_type : null,
            typeof data.attachment_count === "number" ? data.attachment_count : null,
          );
        }
      } catch {
        // Silently fail — title/origin-email are progressive enhancement.
      }
    };

    fetchDetails();
  }, [
    id,
    needsTitleFetch,
    session?.access_token,
    updateCitationTitle,
    updateCitationOriginEmail,
    updateCitationDocMeta,
  ]);

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

  const isEmail = citation.documentType === "email";

  return (
    <div className="flex items-center px-3 py-2 rounded-md hover:bg-MTSS-gray-light/30 group">
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
            <div className="flex items-center gap-1.5 min-w-0">
              {isEmail && (
                <Mail
                  className="h-3.5 w-3.5 flex-shrink-0 text-MTSS-blue"
                  aria-hidden="true"
                />
              )}
              <p className="text-sm font-medium text-MTSS-blue-dark truncate min-w-0 flex-1 group-hover:underline">
                {citation.title}
              </p>
            </div>
          )}
          {isEmail && typeof citation.attachmentCount === "number" && (
            <p className="text-xs text-MTSS-gray truncate">
              {citation.attachmentCount === 0
                ? "No attachments"
                : `${citation.attachmentCount} attachment${citation.attachmentCount === 1 ? "" : "s"}`}
            </p>
          )}
          {!isEmail && citation.originEmailSubject && (
            <div
              className="flex items-center gap-1 min-w-0 text-xs text-MTSS-gray"
              title={`From email: ${citation.originEmailSubject}`}
            >
              <Mail className="h-3 w-3 flex-shrink-0" aria-hidden="true" />
              <span className="truncate min-w-0 flex-1">{citation.originEmailSubject}</span>
            </div>
          )}
          {locationParts.length > 0 && (
            <p className="text-xs text-MTSS-gray">{locationParts.join(" | ")}</p>
          )}
        </div>
      </button>

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
 * remark plugin that wraps every mdast ``text`` node whose source line
 * overlaps any of the supplied ``lineRanges`` in a ``<mark>`` HTML span.
 *
 * Operating on the AST (rather than the raw markdown source) means the
 * markdown has already been parsed by the time we add highlights — so we
 * never accidentally break a heading, list item, blockquote, or GFM table
 * row by prepending raw HTML to its source line. Block structure stays
 * intact; only the rendered text content inside each block gets wrapped.
 *
 * Line numbers come from ``node.position`` which remark fills in from the
 * original markdown source. Callers must pass line ranges that reference
 * the same source (``citation.archive_browse_uri`` content → ranges from
 * ``activeHighlights``).
 */
function remarkLineHighlights(lineRanges: [number, number][] | undefined) {
  return () => (tree: MdastRoot) => {
    if (!lineRanges || lineRanges.length === 0) return;
    const overlaps = (startLine: number, endLine: number) =>
      lineRanges.some(([s, e]) => !(endLine < s || startLine > e));

    // Collect first, mutate after — ``unist-util-visit`` doesn't guarantee
    // correct sibling iteration if you mutate ``parent.children`` during
    // the visit, and ``remark-breaks`` splits lines into multiple text
    // nodes that need each to be wrapped independently.
    const replacements: Array<{
      parent: { children: Array<MdastText | { type: "html"; value: string }> };
      index: number;
      value: string;
    }> = [];

    visit(tree, "text", (node: MdastText, index, parent) => {
      if (!parent || index === undefined) return;
      // Child text nodes inside setext headings and broken-paragraph
      // runs (``remark-breaks`` splits) get no ``position`` field of
      // their own — only the wrapping block carries it. Fall back to
      // the parent's position range so those text nodes still get
      // highlighted. Coarser than per-node but matches the granularity
      // users ask for anyway (whole chunk range).
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const parentPos = (parent as any)?.position;
      const startLine = node.position?.start.line ?? parentPos?.start?.line;
      const endLine = node.position?.end.line ?? parentPos?.end?.line;
      if (startLine === undefined || endLine === undefined) return;
      if (!overlaps(startLine, endLine)) return;
      replacements.push({
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        parent: parent as any,
        index,
        value: `<mark class="citation-highlight">${escapeMarkHtml(node.value)}</mark>`,
      });
    });

    // In-place replacement (no length change) — order doesn't matter.
    for (const { parent, index, value } of replacements) {
      parent.children[index] = { type: "html", value };
    }
  };
}

function escapeMarkHtml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

export function SourceViewDialog({ chunkId, open, onOpenChange, linesToHighlight }: SourceViewDialogProps) {
  const { session } = useAuth();
  const [citation, setCitation] = useState<Citation | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Full archived .md (preamble-stripped) fetched lazily from the archive
  // endpoint. Preferred over `citation.content` (which is the chunk slice,
  // and still has the old broken-GFM tables). Optimistic flow: chunk text
  // renders as soon as the /citations/ call resolves, then this swaps in
  // when the .md fetch lands. Fetch failure keeps chunk text visible.
  const [archiveMd, setArchiveMd] = useState<string | null>(null);
  const contentRef = useRef<HTMLDivElement>(null);

  // In-dialog navigation: user can click "From email: <subject>" to swap
  // content to the originating email. Stack holds previous chunk IDs so a
  // Back button can restore the attachment view.
  const [currentChunkId, setCurrentChunkId] = useState<string | null>(chunkId);
  const [navStack, setNavStack] = useState<string[]>([]);

  // Sync internal state with the externally-selected chunk: a fresh cite
  // click resets any in-dialog navigation.
  useEffect(() => {
    setCurrentChunkId(chunkId);
    setNavStack([]);
  }, [chunkId]);

  // highlights only apply to the originally-opened chunk, not to anything
  // the user navigates to via "From email".
  const isOriginalChunk = navStack.length === 0;
  const activeHighlights = isOriginalChunk ? linesToHighlight : undefined;

  useEffect(() => {
    if (!open || !currentChunkId || !session?.access_token) {
      setCitation(null);
      setError(null);
      setArchiveMd(null);
      return;
    }

    // Clear previous citation immediately so a rapid reopen with a new
    // chunkId doesn't flash the prior document. Same reason archiveMd is
    // reset here and on the cleanup path below.
    setCitation(null);
    setArchiveMd(null);
    setLoading(true);
    setError(null);

    const controller = new AbortController();

    (async () => {
      try {
        const response = await fetch(`${getApiBaseUrl()}/citations/${currentChunkId}`, {
          headers: {
            Authorization: `Bearer ${session.access_token}`,
          },
          signal: controller.signal,
        });

        if (!response.ok) {
          throw new Error(`Failed to fetch: ${response.statusText}`);
        }

        const data = await response.json();
        if (controller.signal.aborted) return;
        setCitation(data);
      } catch (err) {
        if (controller.signal.aborted) return;
        setError(err instanceof Error ? err.message : "Failed to load content");
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    })();

    return () => {
      controller.abort();
    };
  }, [open, currentChunkId, session]);

  const handleOpenOrigin = () => {
    if (!citation?.origin_email?.chunk_id || !currentChunkId) return;
    setNavStack((prev) => [...prev, currentChunkId]);
    setCurrentChunkId(citation.origin_email.chunk_id);
  };

  const handleBack = () => {
    setNavStack((prev) => {
      if (prev.length === 0) return prev;
      const next = prev.slice(0, -1);
      setCurrentChunkId(prev[prev.length - 1]);
      return next;
    });
  };

  // View-link interception: when a user clicks "[View](…md)" inside the
  // rendered archive markdown, keep them in the dialog by looking up the
  // matching chunk and swapping currentChunkId instead of opening a new tab.
  const handleOpenBrowseUri = useCallback(
    async (archiveUrl: string) => {
      if (!session?.access_token || !currentChunkId) return false;
      const bucketPath = toBucketRelative(archiveUrl);
      if (!bucketPath) return false;
      try {
        const resp = await fetch(
          `${getApiBaseUrl()}/citations/by-browse-uri?uri=${encodeURIComponent(bucketPath)}`,
          { headers: { Authorization: `Bearer ${session.access_token}` } },
        );
        if (!resp.ok) return false;
        const data = await resp.json();
        const nextId: string | undefined = data?.chunk_id;
        if (!nextId) return false;
        setNavStack((prev) => [...prev, currentChunkId]);
        setCurrentChunkId(nextId);
        return true;
      } catch {
        return false;
      }
    },
    [session?.access_token, currentChunkId],
  );

  const handleOpenChange = (nextOpen: boolean) => {
    if (!nextOpen) {
      setNavStack([]);
    }
    onOpenChange(nextOpen);
  };

  // Fetch the full archived .md — the per-document source of truth that
  // was rewritten to valid GFM after the initial ingest. The chunk text
  // in `citation.content` is rendered immediately for instant feedback;
  // this fetch swaps the body once it lands. Cached per-URL so reopening
  // the same doc during a session is free. No-op if `archive_browse_uri`
  // is missing — we simply keep the chunk text.
  useEffect(() => {
    if (!open || !citation?.archive_browse_uri || !session?.access_token) {
      return;
    }

    const url = `${getApiBaseUrl()}/archive/${stripArchivePrefix(citation.archive_browse_uri)}`;

    const cached = getCachedArchiveMd(url);
    if (cached !== undefined) {
      setArchiveMd(cached);
      return;
    }

    let cancelled = false;

    (async () => {
      try {
        const response = await fetch(url, {
          headers: { Authorization: `Bearer ${session.access_token}` },
        });
        if (!response.ok) {
          throw new Error(`Archive fetch failed: ${response.status} ${response.statusText}`);
        }
        const text = await response.text();
        setCachedArchiveMd(url, text);
        if (!cancelled) setArchiveMd(text);
      } catch (err) {
        // Fall back to citation.content (already rendering). Don't surface
        // this to the UI — the chunk text is a perfectly usable fallback.
        console.warn(`Failed to fetch archive markdown at ${url}:`, err);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [open, citation?.archive_browse_uri, session]);

  // Scroll to first highlight after content loads. Re-runs when the
  // rendered body swaps from chunk text to the full .md so highlights
  // align against the (larger) markdown file.
  useEffect(() => {
    if (!loading && citation?.content && activeHighlights && activeHighlights.length > 0) {
      // Small delay to ensure DOM is ready
      const timer = setTimeout(() => {
        const firstHighlight = contentRef.current?.querySelector(".citation-highlight");
        if (firstHighlight) {
          firstHighlight.scrollIntoView({ behavior: "smooth", block: "center" });
        }
      }, 100);
      return () => clearTimeout(timer);
    }
  }, [loading, archiveMd, citation?.content, activeHighlights]);

  const locationParts: string[] = [];
  if (citation?.page) locationParts.push(`Page ${citation.page}`);

  // Show highlighted line info
  if (activeHighlights && activeHighlights.length > 0) {
    if (activeHighlights.length === 1) {
      const [start, end] = activeHighlights[0];
      locationParts.push(`Lines ${start}-${end} highlighted`);
    } else {
      locationParts.push(`${activeHighlights.length} sections highlighted`);
    }
  } else if (citation?.lines) {
    locationParts.push(`Lines ${citation.lines[0]}-${citation.lines[1]}`);
  }

  // Process content: prefer the full archived .md (swapped in progressively
  // via the fetch above), falling back to the chunk text. Then rewrite legacy
  // attachment listings, friendlify the MIME "**Type:**" line, inject a
  // prominent image preview for image attachments, and apply line highlights
  // last (so it sees the final text). Render the archived .md verbatim —
  // stripping the preamble would shift line numbers out from under the
  // citation's activeHighlights (which reference original .md line numbers),
  // causing the highlight band to land on unrelated content. The preamble
  // duplicates info already shown in the dialog header, which is cosmetic.
  const renderedBody = archiveMd !== null
    ? archiveMd
    : (citation?.content ?? undefined);

  const isImageCitation =
    citation?.document_type === "attachment_image" ||
    IMAGE_EXTENSIONS.test(citation?.archive_download_uri || "");
  const imagePreviewUrl = isImageCitation && citation?.archive_download_uri
    ? withArchiveToken(
        `${getApiBaseUrl()}/archive/${stripArchivePrefix(citation.archive_download_uri)}`,
        session?.access_token,
      )
    : null;

  let processedContent = renderedBody;
  if (processedContent) {
    processedContent = rewriteAttachmentListing(processedContent);
    processedContent = friendlifyTypeLine(processedContent);
    processedContent = injectImagePreview(processedContent, imagePreviewUrl, isImageCitation);
  }
  // Highlighting is applied inside ReactMarkdown via ``remarkLineHighlights``
  // (an AST-level plugin) so block-level constructs — headings, list items,
  // blockquotes, GFM table rows — stay intact. See the plugin's docstring.

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="max-w-4xl h-[80vh] flex flex-col overflow-hidden" aria-describedby={undefined}>
        <DialogHeader className="flex-shrink-0">
          <div className="flex items-start justify-between gap-4 pr-8">
            <div className="min-w-0 flex-1">
              {navStack.length > 0 && (
                <button
                  type="button"
                  onClick={handleBack}
                  className="inline-flex items-center gap-1 text-xs text-MTSS-blue hover:underline mb-1"
                >
                  <ArrowLeft className="h-3 w-3" />
                  Back
                </button>
              )}
              <DialogTitle className="text-lg">
                {citation?.source_title || "Source Document"}
              </DialogTitle>
              {citation?.origin_email && (
                <button
                  type="button"
                  onClick={handleOpenOrigin}
                  className="inline-flex items-center gap-1 text-sm text-MTSS-blue hover:underline mt-1 max-w-full"
                  title="Open originating email"
                >
                  <Mail className="h-3.5 w-3.5 flex-shrink-0" />
                  <span className="truncate">
                    From email: {citation.origin_email.subject || "(no subject)"}
                  </span>
                </button>
              )}
              {citation?.document_type === "email"
                && typeof citation.attachment_count === "number" && (
                <p className="text-sm text-MTSS-gray mt-1">
                  {citation.attachment_count === 0
                    ? "No attachments"
                    : `${citation.attachment_count} attachment${citation.attachment_count === 1 ? "" : "s"}`}
                </p>
              )}
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
              <TooltipProvider delayDuration={200}>
              <div className="prose prose-sm max-w-none prose-headings:text-MTSS-blue-dark prose-p:text-MTSS-blue-dark prose-li:text-MTSS-blue-dark prose-a:text-MTSS-blue prose-a:underline prose-hr:my-4 [&_p]:mb-4 [&_br+br]:block [&_br+br]:h-2 [&_.citation-highlight]:bg-yellow-200 [&_.citation-highlight]:px-0.5 [&_.citation-highlight]:rounded [&_.source-image-preview]:my-3 [&_.source-image-preview]:max-w-full [&_.source-image-preview]:max-h-[420px] [&_.source-image-preview]:object-contain [&_.source-image-preview]:mx-auto [&_.source-image-preview]:block [&_.source-image-preview]:rounded [&_.source-image-preview]:border [&_.source-image-preview]:border-MTSS-gray-light">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm, remarkBreaks, remarkLineHighlights(activeHighlights)]}
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
                      // For remaining archive links, attach the session token
                      // as a query param so anchor-new-tab navigations pass
                      // AuthMiddleware (which accepts ?token= as a fallback
                      // when the Authorization header can't be sent).
                      const isArchive = resolved.includes("/api/archive/");
                      resolved = withArchiveToken(resolved, session?.access_token);
                      // Special-case "[Details](…md)" inside archive markdown
                      // — keep the user in this dialog instead of opening a
                      // new tab. Non-.md archive links (Download) still open
                      // externally with the token appended.
                      const isArchiveMd = isArchive && /\.md(?:\?|$)/.test(resolved);
                      const onClick = isArchiveMd
                        ? (e: ReactMouseEvent<HTMLAnchorElement>) => {
                            if (e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return;
                            e.preventDefault();
                            void handleOpenBrowseUri(resolved);
                          }
                        : undefined;

                      const anchor = (
                        <a
                          href={resolved}
                          target="_blank"
                          rel="noopener noreferrer"
                          onClick={onClick}
                          {...props}
                        >
                          {children}
                        </a>
                      );

                      // Show a floating image preview when hovering a
                      // Download link that points at an image file. Keeps
                      // users from having to open a tab just to see what
                      // they're downloading.
                      const isImageDownload =
                        isArchive && !isArchiveMd && IMAGE_EXTENSIONS.test(resolved);
                      if (!isImageDownload) return anchor;
                      return (
                        <Tooltip>
                          <TooltipTrigger asChild>{anchor}</TooltipTrigger>
                          <TooltipContent side="top" className="p-1 bg-white border shadow-md">
                            <ImageHoverPreview src={resolved} />
                          </TooltipContent>
                        </Tooltip>
                      );
                    },
                    img: ({ src, alt, ...props }) => {
                      let resolved = src || "";
                      if (!resolved.startsWith("http")) {
                        resolved = resolveArchiveUrl(resolved, citation?.archive_browse_uri);
                      }
                      resolved = withArchiveToken(resolved, session?.access_token);
                      return <img src={resolved} alt={alt || ""} {...props} />;
                    },
                  }}
                >
                  {processedContent}
                </ReactMarkdown>
              </div>
              </TooltipProvider>
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
