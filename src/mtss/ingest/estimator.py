"""Ingest cost estimator — extracts attachments and counts pages for cost estimation.

Three-phase extraction + scan:
  Phase 1 — Discover: Scan for folders with summary.json (= complete).
  Phase 2 — Extract & Count: For new/failed EMLs, extract and count pages.
  Phase 3 — Aggregate: Sum all summary.json files, compute costs.
"""

from __future__ import annotations

import base64
import email
import io
import json
import logging
import re
import shutil
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email import policy
from pathlib import Path
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree

from ..config import get_settings
from ..processing.lane_classifier import (
    DOCUMENT_EXTENSIONS,
    IMAGE_EXTENSIONS,
    IMAGE_MIMETYPES,
    LLAMAPARSE_MIMETYPES,
    ZIP_MIMETYPES,
)
from ..utils import sanitize_filename

logger = logging.getLogger(__name__)

# UTF-8 BOM bytes — some email clients add this at the start of EML files
UTF8_BOM = b"\xef\xbb\xbf"

# Categories for classification
_PDF_EXTS = {".pdf"}
_DOCX_EXTS = {".docx"}
_PPTX_EXTS = {".pptx"}
_XLSX_EXTS = {".xlsx"}
_DOC_EXTS = {".doc"}
_PPT_EXTS = {".ppt"}
_XLS_EXTS = {".xls"}
_TEXT_EXTS = {".txt", ".md", ".csv", ".rtf", ".html", ".htm"}
_OTHER_DOC_EXTS = {".epub", ".odt", ".ods", ".odp"}


def _classify_file(filename: str) -> str:
    """Classify a file into a category based on its extension."""
    ext = Path(filename).suffix.lower()
    if ext in _PDF_EXTS:
        return "PDF"
    if ext in _DOCX_EXTS:
        return "DOCX"
    if ext in _PPTX_EXTS:
        return "PPTX"
    if ext in _XLSX_EXTS:
        return "XLSX"
    if ext in _DOC_EXTS:
        return "DOC"
    if ext in _PPT_EXTS:
        return "PPT"
    if ext in _XLS_EXTS:
        return "XLS"
    if ext in _TEXT_EXTS:
        return "Text/Markdown"
    if ext in IMAGE_EXTENSIONS:
        return "Images"
    if ext in _OTHER_DOC_EXTS:
        return "Other Docs"
    return "Other"


# Categories that have meaningful page counts
PAGE_COUNT_CATEGORIES = {"PDF", "DOCX", "PPTX", "XLSX", "DOC", "PPT", "XLS", "Other Docs"}

# Categories processed by LlamaParse
LLAMAPARSE_CATEGORIES = {"PDF", "DOCX", "PPTX", "XLSX", "DOC", "PPT", "XLS", "Other Docs"}

# Categories processed by vision API
VISION_CATEGORIES = {"Images"}

# Categories processed as text
TEXT_CATEGORIES = {"Text/Markdown"}


@dataclass
class CategoryStats:
    """Statistics for a single file category."""

    file_count: int = 0
    page_count: int = 0
    pages_unknown: int = 0
    errors: int = 0
    images_meaningful: int = 0
    images_skipped: int = 0


@dataclass
class FileIssue:
    """A file that couldn't be reliably processed."""

    file: str  # relative path within email folder
    issue: str  # "page_count_unknown" | "parse_error" | "zip_error"
    detail: str  # human-readable explanation


@dataclass
class ScanResult:
    """Aggregated results from scanning all EML files."""

    eml_count: int = 0
    extracted_count: int = 0
    cached_count: int = 0
    categories: Dict[str, CategoryStats] = field(default_factory=dict)
    scan_errors: List[str] = field(default_factory=list)
    all_issues: List[FileIssue] = field(default_factory=list)
    elapsed_seconds: float = 0.0


class IngestEstimator:
    """Estimates ingest cost by extracting attachments and counting pages.

    Persists extraction results so subsequent runs are instant.
    """

    def __init__(self, source_dir: Optional[Path] = None, estimate_dir: Optional[Path] = None):
        settings = get_settings()
        self.source_dir = source_dir or settings.eml_source_dir
        self.estimate_dir = estimate_dir or (settings.data_processed_dir / "estimate")
        self.zip_max_files = settings.zip_max_files
        self.zip_max_depth = settings.zip_max_depth
        self.zip_max_total_size_mb = settings.zip_max_total_size_mb

    def scan(self) -> ScanResult:
        """Run all three phases and return aggregated results."""
        start = time.monotonic()
        result = ScanResult()

        # Find all EML files
        eml_files = sorted(self.source_dir.rglob("*.eml"))
        result.eml_count = len(eml_files)

        if not eml_files:
            result.elapsed_seconds = time.monotonic() - start
            return result

        self.estimate_dir.mkdir(parents=True, exist_ok=True)

        # Phase 1 — Discover cached extractions
        cached_summaries, uncached_emls = self._discover(eml_files)
        result.cached_count = len(cached_summaries)

        # Phase 2 — Extract & count new EMLs
        new_summaries = []
        for eml_path in uncached_emls:
            try:
                summary = self._extract_and_count(eml_path)
                new_summaries.append(summary)
                result.extracted_count += 1
            except Exception as e:
                result.scan_errors.append(f"{eml_path.name}: {e}")
                logger.warning(f"Failed to process {eml_path}: {e}")

        # Phase 3 — Aggregate
        all_summaries = cached_summaries + new_summaries
        self._aggregate(all_summaries, result)

        result.elapsed_seconds = time.monotonic() - start
        return result

    # ── Phase 1: Discover ──────────────────────────────────────────────

    def _discover(
        self, eml_files: List[Path]
    ) -> tuple[List[Dict[str, Any]], List[Path]]:
        """Scan estimate folder for completed extractions.

        Returns:
            Tuple of (cached_summaries, uncached_eml_paths).
        """
        cached: List[Dict[str, Any]] = []
        uncached: List[Path] = []

        for eml_path in eml_files:
            folder_name = self._eml_folder_name(eml_path)
            folder = self.estimate_dir / folder_name
            summary_path = folder / "summary.json"

            if summary_path.exists():
                try:
                    with open(summary_path, "r", encoding="utf-8") as f:
                        cached.append(json.load(f))
                    continue
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning(f"Corrupt summary.json in {folder}, re-extracting: {e}")
                    # Fall through to uncached

            # Folder exists without summary.json → delete and re-extract
            if folder.exists():
                shutil.rmtree(folder, ignore_errors=True)

            uncached.append(eml_path)

        return cached, uncached

    # ── Phase 2: Extract & Count ───────────────────────────────────────

    def _extract_and_count(self, eml_path: Path) -> Dict[str, Any]:
        """Extract attachments from an EML and count pages.

        Creates folder structure and writes summary.json on success.
        """
        folder_name = self._eml_folder_name(eml_path)
        folder = self.estimate_dir / folder_name
        folder.mkdir(parents=True, exist_ok=True)
        attachments_dir = folder / "attachments"
        attachments_dir.mkdir(exist_ok=True)

        # Copy original EML
        shutil.copy2(eml_path, folder / "email.eml")

        # Parse EML (BOM-aware)
        with open(eml_path, "rb") as f:
            content = f.read()
        if content.startswith(UTF8_BOM):
            content = content[3:]
        msg = email.message_from_bytes(content, policy=policy.default)

        # Extract attachments
        saved_files: List[Path] = []
        issues: List[FileIssue] = []

        if msg.is_multipart():
            for part in msg.walk():
                content_disposition = str(part.get("Content-Disposition", ""))
                filename = part.get_filename()
                if "attachment" not in content_disposition and not filename:
                    continue

                filename = sanitize_filename(filename or "unnamed_attachment")
                payload = part.get_payload(decode=True)
                if not payload:
                    continue

                saved_path = attachments_dir / filename
                # Handle duplicate filenames
                counter = 1
                while saved_path.exists():
                    stem = Path(filename).stem
                    suffix = Path(filename).suffix
                    saved_path = attachments_dir / f"{stem}_{counter}{suffix}"
                    counter += 1

                saved_path.write_bytes(payload)

                # Check if it's a ZIP — extract contents
                ext = saved_path.suffix.lower()
                if ext == ".zip":
                    zip_issues = self._extract_zip(
                        saved_path, attachments_dir, folder, depth=0
                    )
                    issues.extend(zip_issues)
                    # Don't count the ZIP file itself
                else:
                    saved_files.append(saved_path)

        # Also pick up any files extracted from ZIPs
        for path in attachments_dir.rglob("*"):
            if path.is_file() and path not in saved_files and path.suffix.lower() != ".zip":
                saved_files.append(path)

        # Count pages per category
        categories: Dict[str, Dict[str, int]] = {}
        for file_path in saved_files:
            category = _classify_file(file_path.name)
            if category not in categories:
                categories[category] = {
                    "file_count": 0, "page_count": 0, "pages_unknown": 0,
                    "images_meaningful": 0, "images_skipped": 0,
                }

            categories[category]["file_count"] += 1

            if category in PAGE_COUNT_CATEGORIES:
                pages, issue = self._count_pages(file_path, folder)
                if issue:
                    issues.append(issue)
                    categories[category]["pages_unknown"] += 1
                categories[category]["page_count"] += pages
            elif category in VISION_CATEGORIES:
                if self._is_meaningful_image(file_path):
                    categories[category]["images_meaningful"] += 1
                else:
                    categories[category]["images_skipped"] += 1

        # Build summary
        total_files = sum(c["file_count"] for c in categories.values())
        total_pages = sum(
            c["page_count"] for c in categories.values()
            if any(
                cat in PAGE_COUNT_CATEGORIES
                for cat in categories
                if categories[cat] is c
            )
        )

        summary = {
            "source_eml": str(eml_path),
            "extracted_at": datetime.now(timezone.utc).isoformat(),
            "categories": categories,
            "total_files": total_files,
            "total_pages": total_pages,
            "issues": [
                {"file": i.file, "issue": i.issue, "detail": i.detail}
                for i in issues
            ],
        }

        # Write summary.json (completion marker)
        with open(folder / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary

    def _extract_zip(
        self,
        zip_path: Path,
        attachments_dir: Path,
        eml_folder: Path,
        depth: int,
    ) -> List[FileIssue]:
        """Recursively extract ZIP contents."""
        issues: List[FileIssue] = []
        rel = zip_path.relative_to(eml_folder)

        if depth >= self.zip_max_depth:
            issues.append(FileIssue(
                file=str(rel),
                issue="zip_error",
                detail=f"Max ZIP nesting depth ({self.zip_max_depth}) exceeded",
            ))
            return issues

        extract_dir = zip_path.parent / f"{zip_path.stem}_extracted"
        extract_dir.mkdir(exist_ok=True)

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                file_count = 0
                total_size = 0
                max_total_bytes = self.zip_max_total_size_mb * 1024 * 1024

                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    if file_count >= self.zip_max_files:
                        issues.append(FileIssue(
                            file=str(rel),
                            issue="zip_error",
                            detail=f"ZIP exceeds max file count ({self.zip_max_files})",
                        ))
                        break

                    total_size += info.file_size
                    if total_size > max_total_bytes:
                        issues.append(FileIssue(
                            file=str(rel),
                            issue="zip_error",
                            detail=f"ZIP exceeds max total size ({self.zip_max_total_size_mb}MB)",
                        ))
                        break

                    # Sanitize the member filename
                    member_name = sanitize_filename(Path(info.filename).name)
                    if not member_name:
                        continue

                    target = extract_dir / member_name
                    counter = 1
                    while target.exists():
                        stem = Path(member_name).stem
                        suffix = Path(member_name).suffix
                        target = extract_dir / f"{stem}_{counter}{suffix}"
                        counter += 1

                    try:
                        with zf.open(info) as src, open(target, "wb") as dst:
                            dst.write(src.read())
                        file_count += 1

                        # Recurse into nested ZIPs
                        if target.suffix.lower() == ".zip":
                            nested_issues = self._extract_zip(
                                target, attachments_dir, eml_folder, depth + 1
                            )
                            issues.extend(nested_issues)
                    except Exception as e:
                        issues.append(FileIssue(
                            file=f"{rel}/{member_name}",
                            issue="zip_error",
                            detail=str(e),
                        ))

        except zipfile.BadZipFile as e:
            issues.append(FileIssue(
                file=str(rel),
                issue="zip_error",
                detail=f"Corrupt ZIP: {e}",
            ))
        except Exception as e:
            issues.append(FileIssue(
                file=str(rel),
                issue="zip_error",
                detail=str(e),
            ))

        return issues

    def _count_pages(
        self, file_path: Path, eml_folder: Path
    ) -> tuple[int, Optional[FileIssue]]:
        """Count pages for a file. Returns (page_count, issue_or_none).

        If page count can't be determined, returns (1, FileIssue).
        """
        rel = str(file_path.relative_to(eml_folder))
        ext = file_path.suffix.lower()

        try:
            if ext == ".pdf":
                return self._count_pdf_pages(file_path, rel)
            if ext == ".docx":
                return self._count_docx_pages(file_path, rel)
            if ext == ".pptx":
                return self._count_pptx_pages(file_path, rel)
            if ext == ".xlsx":
                return self._count_xlsx_pages(file_path, rel)
            if ext == ".doc":
                return self._count_ole_pages(file_path, rel, "doc")
            if ext == ".ppt":
                return self._count_ole_pages(file_path, rel, "ppt")
            if ext == ".xls":
                return self._count_ole_pages(file_path, rel, "xls")
            # CSV, RTF, HTML, other docs → 1 page each
            return 1, None
        except Exception as e:
            return 1, FileIssue(
                file=rel, issue="parse_error", detail=str(e)
            )

    def _count_pdf_pages(self, path: Path, rel: str) -> tuple[int, Optional[FileIssue]]:
        """Count PDF pages using pypdf with fallbacks for broken PDFs.

        Fallback chain:
        1. pypdf on raw file
        2. If file is base64-encoded, decode and retry pypdf
        3. Regex: linearized /N value (page count in linearized PDF header)
        4. Regex: count /Type /Page objects in the raw/decoded data
        """
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError

        raw = path.read_bytes()
        pdf_data = raw

        # Detect base64-encoded PDFs (email extraction sometimes leaves these)
        if raw[:5] == b"JVBER":
            valid_len = (len(raw) // 4) * 4
            try:
                pdf_data = base64.b64decode(raw[:valid_len])
            except Exception:
                pass  # not actually valid base64, use raw

        # Strategy 1: pypdf
        try:
            reader = PdfReader(io.BytesIO(pdf_data))
            count = len(reader.pages)
            if count > 0:
                return count, None
        except (PdfReadError, Exception):
            pass

        # Strategy 2: linearized PDF /N value (fast, reliable when present)
        lin = re.search(rb"/Linearized\b[^>]*/N\s+(\d+)", pdf_data[:4096])
        if lin:
            count = int(lin.group(1))
            if count > 0:
                return count, None

        # Strategy 3: count /Type /Page objects (not /Pages)
        page_count = len(re.findall(rb"/Type\s*/Page\b(?!s)", pdf_data))
        if page_count > 0:
            return page_count, None

        return 1, FileIssue(
            file=rel,
            issue="parse_error",
            detail="could not determine page count (pypdf + regex fallbacks failed)",
        )

    def _count_docx_pages(self, path: Path, rel: str) -> tuple[int, Optional[FileIssue]]:
        """Count DOCX pages from metadata or structural analysis."""
        try:
            with zipfile.ZipFile(path, "r") as zf:
                # Strategy 1: docProps/app.xml → <Pages> element
                if "docProps/app.xml" in zf.namelist():
                    with zf.open("docProps/app.xml") as f:
                        tree = ElementTree.parse(f)
                    root = tree.getroot()
                    # Namespace varies; search for Pages element
                    for elem in root.iter():
                        if elem.tag.endswith("}Pages") or elem.tag == "Pages":
                            if elem.text and elem.text.strip().isdigit():
                                count = int(elem.text.strip())
                                if count > 0:
                                    return count, None

                # Strategy 2: Count page breaks + section breaks in document.xml
                if "word/document.xml" in zf.namelist():
                    with zf.open("word/document.xml") as f:
                        doc_xml = f.read().decode("utf-8", errors="replace")

                    page_breaks = doc_xml.count('w:type="page"')
                    # Section breaks also start new pages
                    sect_count = doc_xml.count("</w:sectPr>")
                    # Subtract 1 for the final section (always present)
                    extra_sections = max(0, sect_count - 1)
                    estimated = page_breaks + extra_sections + 1

                    if estimated > 1:
                        return estimated, None

                # Can't determine page count
                return 1, FileIssue(
                    file=rel,
                    issue="page_count_unknown",
                    detail="no metadata in docProps/app.xml, no manual page breaks found",
                )
        except zipfile.BadZipFile as e:
            return 1, FileIssue(
                file=rel, issue="parse_error", detail=f"Bad DOCX zip: {e}"
            )

    def _count_pptx_pages(self, path: Path, rel: str) -> tuple[int, Optional[FileIssue]]:
        """Count PPTX slides from presentation.xml."""
        try:
            with zipfile.ZipFile(path, "r") as zf:
                if "ppt/presentation.xml" in zf.namelist():
                    with zf.open("ppt/presentation.xml") as f:
                        content = f.read().decode("utf-8", errors="replace")
                    # Count <p:sldId> elements
                    count = content.count("<p:sldId ")
                    if count > 0:
                        return count, None

                return 1, FileIssue(
                    file=rel,
                    issue="page_count_unknown",
                    detail="could not find slide IDs in ppt/presentation.xml",
                )
        except zipfile.BadZipFile as e:
            return 1, FileIssue(
                file=rel, issue="parse_error", detail=f"Bad PPTX zip: {e}"
            )

    def _count_xlsx_pages(self, path: Path, rel: str) -> tuple[int, Optional[FileIssue]]:
        """Count XLSX worksheets from workbook.xml."""
        try:
            with zipfile.ZipFile(path, "r") as zf:
                if "xl/workbook.xml" in zf.namelist():
                    with zf.open("xl/workbook.xml") as f:
                        content = f.read().decode("utf-8", errors="replace")
                    # Count <sheet> elements
                    count = content.count("<sheet ")
                    if count > 0:
                        return count, None

                return 1, FileIssue(
                    file=rel,
                    issue="page_count_unknown",
                    detail="could not find sheets in xl/workbook.xml",
                )
        except zipfile.BadZipFile as e:
            return 1, FileIssue(
                file=rel, issue="parse_error", detail=f"Bad XLSX zip: {e}"
            )

    def _count_ole_pages(
        self, path: Path, rel: str, fmt: str
    ) -> tuple[int, Optional[FileIssue]]:
        """Count pages/slides/sheets in legacy Office formats using olefile."""
        try:
            import olefile
        except ImportError:
            return 1, FileIssue(
                file=rel,
                issue="parse_error",
                detail="olefile not installed",
            )

        try:
            ole = olefile.OleFileIO(str(path))
            meta = ole.get_metadata()

            if fmt == "doc":
                ole.close()
                if meta.num_pages and meta.num_pages > 0:
                    return meta.num_pages, None
            elif fmt == "ppt":
                ole.close()
                # OLE metadata stores slide count in num_pages for PPT
                if meta.num_pages and meta.num_pages > 0:
                    return meta.num_pages, None
            elif fmt == "xls":
                # OLE summary metadata doesn't have sheet count.
                # Parse the Workbook stream for BoundSheet8 records (type 0x0085).
                for stream_name in ["Workbook", "Book"]:
                    if ole.exists(stream_name):
                        data = ole.openstream(stream_name).read()
                        sheet_count = 0
                        pos = 0
                        while pos + 4 <= len(data):
                            rec_type = int.from_bytes(data[pos:pos+2], "little")
                            rec_len = int.from_bytes(data[pos+2:pos+4], "little")
                            if rec_type == 0x0085:  # BoundSheet8
                                sheet_count += 1
                            pos += 4 + rec_len
                            if rec_type == 0x000A:  # EOF record
                                break
                        if sheet_count > 0:
                            ole.close()
                            return sheet_count, None
                        break
                ole.close()
            else:
                ole.close()

            return 1, FileIssue(
                file=rel,
                issue="page_count_unknown",
                detail=f"OLE metadata has no page count for .{fmt}",
            )
        except Exception as e:
            return 1, FileIssue(
                file=rel, issue="parse_error", detail=f"olefile error: {e}"
            )

    # ── Phase 3: Aggregate ─────────────────────────────────────────────

    def _aggregate(
        self, summaries: List[Dict[str, Any]], result: ScanResult
    ) -> None:
        """Sum all summary.json data into the ScanResult."""
        for summary in summaries:
            cats = summary.get("categories", {})
            for cat_name, cat_data in cats.items():
                if cat_name not in result.categories:
                    result.categories[cat_name] = CategoryStats()
                stats = result.categories[cat_name]
                stats.file_count += cat_data.get("file_count", 0)
                stats.page_count += cat_data.get("page_count", 0)
                stats.pages_unknown += cat_data.get("pages_unknown", 0)
                stats.images_meaningful += cat_data.get("images_meaningful", 0)
                stats.images_skipped += cat_data.get("images_skipped", 0)

            # Collect issues
            for issue_data in summary.get("issues", []):
                result.all_issues.append(FileIssue(
                    file=issue_data.get("file", ""),
                    issue=issue_data.get("issue", ""),
                    detail=issue_data.get("detail", ""),
                ))

    # ── Image heuristic ──────────────────────────────────────────────

    @staticmethod
    def _is_meaningful_image(path: Path) -> bool:
        """Heuristic: return True if the image is likely meaningful content.

        Filters out tracking pixels, icons, logos, banners, and email
        signature images based on file size and dimensions.
        """
        try:
            file_size = path.stat().st_size
        except OSError:
            return True  # can't check, assume meaningful

        # Very small files are almost always tracking pixels or tiny icons
        if file_size < 15_000:
            return False

        # Check dimensions with PIL for more accurate filtering
        try:
            from PIL import Image

            with Image.open(path) as im:
                w, h = im.size
            # Tiny images: icons, tracking pixels, small logos
            if max(w, h) < 100:
                return False
            # Banner-shaped: wide and short (email separators, signature strips)
            if h < 50 and w > 3 * h:
                return False
        except Exception:
            pass  # can't read dimensions, file size check is enough

        return True

    # ── Helpers ─────────────────────────────────────────────────────────

    def _eml_folder_name(self, eml_path: Path) -> str:
        """Generate a unique folder name for an EML file.

        Uses the stem + first 8 chars of a hash for uniqueness.
        """
        import hashlib

        name = sanitize_filename(eml_path.stem)
        # Truncate long names
        if len(name) > 60:
            name = name[:60]
        hash_suffix = hashlib.sha256(
            str(eml_path.resolve()).encode()
        ).hexdigest()[:8]
        return f"{name}_{hash_suffix}"
