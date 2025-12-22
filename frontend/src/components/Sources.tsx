/**
 * Source display components - SourceList, SourceCard, ConfidenceIndicator.
 * Consolidated into a single file for simplicity (KISS principle).
 */

import { useState } from "react";
import { FileText, Mail, Image, FileSpreadsheet, Presentation, File, ChevronRight } from "lucide-react";
import { cn, formatDate, getConfidenceLevel, truncate, getDocumentTypeLabel } from "@/lib/utils";
import { ScrollArea, Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "./ui";
import type { Source } from "@/types/sources";

// =============================================================================
// Confidence Indicator
// =============================================================================

interface ConfidenceIndicatorProps {
  score: number;
  isReranked: boolean;
}

function ConfidenceIndicator({ score, isReranked }: ConfidenceIndicatorProps) {
  const percentage = Math.round(score * 100);
  const { label, color } = getConfidenceLevel(score);

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <div className={cn("flex-shrink-0 px-2 py-1 rounded text-xs font-medium", color)}>
            {percentage}%
            {isReranked && (
              <svg className="w-3 h-3 inline ml-1" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
              </svg>
            )}
          </div>
        </TooltipTrigger>
        <TooltipContent>
          <p>{label}</p>
          <p className="text-xs text-muted-foreground">
            {isReranked ? "Reranked for accuracy" : "Vector similarity"}
          </p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

// =============================================================================
// Document Icon
// =============================================================================

function getDocumentIcon(type: string) {
  switch (type) {
    case "email":
      return Mail;
    case "attachment_pdf":
      return FileText;
    case "attachment_image":
      return Image;
    case "attachment_xlsx":
      return FileSpreadsheet;
    case "attachment_pptx":
      return Presentation;
    default:
      return File;
  }
}

// =============================================================================
// Document Hierarchy
// =============================================================================

interface DocumentHierarchyProps {
  source: Source;
}

function DocumentHierarchy({ source }: DocumentHierarchyProps) {
  const isAttachment = source.document_type !== "email";
  const rootPath = source.root_file_path || source.file_path;
  const fileName = rootPath.split(/[/\\]/).pop() || rootPath;

  return (
    <div className="flex items-center gap-1 text-xs text-ncl-gray flex-wrap">
      {/* Root email */}
      <div className="flex items-center gap-1">
        <Mail className="w-3 h-3 flex-shrink-0" />
        <span className="truncate max-w-[120px]" title={rootPath}>
          {fileName}
        </span>
      </div>

      {isAttachment && (
        <>
          <ChevronRight className="w-3 h-3 text-ncl-gray-light flex-shrink-0" />
          <span className={cn("px-1.5 py-0.5 rounded text-xs", "bg-ncl-blue/10 text-ncl-blue")}>
            {getDocumentTypeLabel(source.document_type)}
          </span>
        </>
      )}

      {source.heading_path.length > 0 && (
        <>
          <ChevronRight className="w-3 h-3 text-ncl-gray-light flex-shrink-0" />
          <span className="text-ncl-gray-dark truncate max-w-[120px]" title={source.heading_path.join(" > ")}>
            {source.heading_path.slice(-1)[0]}
          </span>
        </>
      )}
    </div>
  );
}

// =============================================================================
// Source Card
// =============================================================================

interface SourceCardProps {
  source: Source;
  index: number;
  isSelected: boolean;
  onClick: () => void;
}

export function SourceCard({ source, index, isSelected, onClick }: SourceCardProps) {
  const relevanceScore = source.rerank_score ?? source.similarity_score;
  const isReranked = source.rerank_score !== null;
  const Icon = getDocumentIcon(source.document_type);

  return (
    <button
      onClick={onClick}
      className={cn(
        "w-full text-left p-3 rounded-lg border transition-all",
        "hover:border-ncl-blue hover:shadow-sm",
        isSelected
          ? "border-ncl-blue bg-ncl-blue/5"
          : "border-ncl-gray-light bg-white"
      )}
    >
      <div className="flex items-start gap-3">
        <span className="flex-shrink-0 w-6 h-6 rounded-full bg-ncl-blue text-white text-xs flex items-center justify-center font-medium">
          {index}
        </span>

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <Icon className="w-4 h-4 text-ncl-gray flex-shrink-0" />
            <span className="text-sm font-medium text-ncl-blue-dark truncate">
              {source.email_subject || source.file_path.split(/[/\\]/).pop()}
            </span>
          </div>

          {source.email_initiator && (
            <p className="text-xs text-ncl-gray truncate">
              From: {source.email_initiator}
            </p>
          )}

          {source.email_date && (
            <p className="text-xs text-ncl-gray">
              {formatDate(source.email_date)}
            </p>
          )}

          <p className="text-xs text-ncl-gray-dark mt-1 line-clamp-2">
            {truncate(source.chunk_content, 150)}
          </p>

          <div className="mt-2">
            <DocumentHierarchy source={source} />
          </div>
        </div>

        <ConfidenceIndicator score={relevanceScore} isReranked={isReranked} />
      </div>
    </button>
  );
}

// =============================================================================
// Source List
// =============================================================================

interface SourceListProps {
  sources: Source[];
  onSourceClick?: (source: Source) => void;
}

export function SourceList({ sources, onSourceClick }: SourceListProps) {
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);

  if (sources.length === 0) {
    return null;
  }

  const handleSourceClick = (source: Source, index: number) => {
    setSelectedIndex(index);
    onSourceClick?.(source);
  };

  return (
    <div className="border-t border-ncl-gray-light pt-4 mt-4">
      <h3 className="text-sm font-semibold text-ncl-blue mb-3 flex items-center">
        <FileText className="w-4 h-4 mr-2" />
        Sources ({sources.length})
      </h3>

      <ScrollArea className="max-h-64">
        <div className="space-y-2 pr-2">
          {sources.map((source, index) => (
            <SourceCard
              key={`${source.file_path}-${index}`}
              source={source}
              index={index + 1}
              isSelected={selectedIndex === index}
              onClick={() => handleSourceClick(source, index)}
            />
          ))}
        </div>
      </ScrollArea>
    </div>
  );
}
