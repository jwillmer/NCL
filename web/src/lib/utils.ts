import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

/**
 * Merge Tailwind CSS classes with clsx.
 */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/**
 * Format a date string for display.
 */
export function formatDate(dateString: string | null): string {
  if (!dateString) return "";
  return new Date(dateString).toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

/**
 * Format a date as relative time (e.g., "2 hours ago", "3 days ago").
 */
export function formatRelativeTime(dateString: string | null): string {
  if (!dateString) return "";

  const date = new Date(dateString);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffSeconds = Math.floor(diffMs / 1000);
  const diffMinutes = Math.floor(diffSeconds / 60);
  const diffHours = Math.floor(diffMinutes / 60);
  const diffDays = Math.floor(diffHours / 24);

  if (diffSeconds < 60) return "just now";
  if (diffMinutes < 60) return `${diffMinutes}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays < 7) return `${diffDays}d ago`;

  return formatDate(dateString);
}

/**
 * Group conversations by date category (Today, Yesterday, Last 7 days, Older).
 */
export function groupByDate<T extends { last_message_at: string | null; created_at: string }>(
  items: T[]
): { label: string; items: T[] }[] {
  const now = new Date();
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const yesterday = new Date(today.getTime() - 24 * 60 * 60 * 1000);
  const lastWeek = new Date(today.getTime() - 7 * 24 * 60 * 60 * 1000);

  const groups: { label: string; items: T[] }[] = [
    { label: "Today", items: [] },
    { label: "Yesterday", items: [] },
    { label: "Last 7 days", items: [] },
    { label: "Older", items: [] },
  ];

  for (const item of items) {
    const dateStr = item.last_message_at || item.created_at;
    const date = new Date(dateStr);

    if (date >= today) {
      groups[0].items.push(item);
    } else if (date >= yesterday) {
      groups[1].items.push(item);
    } else if (date >= lastWeek) {
      groups[2].items.push(item);
    } else {
      groups[3].items.push(item);
    }
  }

  // Filter out empty groups
  return groups.filter((g) => g.items.length > 0);
}

/**
 * Truncate text to a maximum length.
 */
export function truncate(text: string, maxLength: number): string {
  if (text.length <= maxLength) return text;
  return text.slice(0, maxLength - 3) + "...";
}

/**
 * Get a human-readable label for document types.
 */
export function getDocumentTypeLabel(type: string): string {
  const labels: Record<string, string> = {
    email: "Email",
    attachment_pdf: "PDF",
    attachment_image: "Image",
    attachment_docx: "Word Document",
    attachment_pptx: "PowerPoint",
    attachment_xlsx: "Excel Spreadsheet",
    attachment_other: "Attachment",
  };
  return labels[type] || type;
}

/**
 * Get confidence level description from score.
 */
export function getConfidenceLevel(score: number): {
  label: string;
  color: string;
} {
  if (score >= 0.8) return { label: "High confidence", color: "text-green-600 bg-green-100" };
  if (score >= 0.6) return { label: "Good match", color: "text-ncl-blue bg-ncl-blue/10" };
  if (score >= 0.4) return { label: "Moderate match", color: "text-yellow-600 bg-yellow-100" };
  return { label: "Possible match", color: "text-ncl-gray bg-ncl-gray-light" };
}
