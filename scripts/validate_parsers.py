"""Validate new local parsers against real EML attachments.

This script:
1. Extracts attachments from real EML files using the existing EML parser
2. Tests each local parser against the extracted files
3. Reports results for parser validation
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import tempfile
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ── Data classes for results ────────────────────────────────────────────────

@dataclass
class ParserResult:
    filename: str
    file_type: str
    parser_used: str
    chars_extracted: int = 0
    content_preview: str = ""
    success: bool = False
    error: str = ""
    extra: dict = field(default_factory=dict)


@dataclass
class ImageFilterResult:
    filename: str
    size_bytes: int = 0
    dimensions: str = ""
    is_meaningful: bool = False
    reason: str = ""


@dataclass
class PDFClassifierResult:
    filename: str
    classification: str = ""
    success: bool = False
    error: str = ""


# ── Target EML files ────────────────────────────────────────────────────────

EML_FILES = [
    "100000922_vp4qcvdw.abx.eml",  # MARAN CANOPUS, equipment photos
    "100000366_qhlztvyi.alr.eml",  # DANAE, starter panel repair
    "100000376_lgqfor05.2fi.eml",  # CAPRICORN, chain compensator
    "100284018_5vcg2bgp.jwm.eml",  # MARS logistics
    "100297210_2ef23hqq.cni.eml",  # THETIS safety equipment
    "100297780_avnjkyr4.40y.eml",  # HERMIONE worklogs
]

EML_DIR = PROJECT_ROOT / "data" / "emails"


# ── Step 1: Extract attachments ─────────────────────────────────────────────

def extract_attachments(tmp_dir: Path) -> dict[str, list[dict]]:
    """Extract attachments from target EML files.

    Returns dict mapping EML filename -> list of attachment info dicts.
    """
    from mtss.parsers.eml_parser import EMLParser

    parser = EMLParser(attachments_dir=tmp_dir)
    results: dict[str, list[dict]] = {}

    for eml_name in EML_FILES:
        eml_path = EML_DIR / eml_name
        if not eml_path.exists():
            logger.warning(f"EML file not found: {eml_path}")
            continue

        logger.info(f"\n{'='*70}")
        logger.info(f"Parsing EML: {eml_name}")
        try:
            parsed = parser.parse_file(eml_path)
            attachments = []
            for att in parsed.attachments:
                att_info = {
                    "filename": att.filename,
                    "content_type": att.content_type,
                    "size_bytes": att.size_bytes,
                    "saved_path": att.saved_path,
                }
                attachments.append(att_info)
                logger.info(
                    f"  Attachment: {att.filename} "
                    f"({att.content_type}, {att.size_bytes:,} bytes)"
                )
            results[eml_name] = attachments
            if not attachments:
                logger.info("  (no attachments found)")
        except Exception as e:
            logger.error(f"  Failed to parse {eml_name}: {e}")
            results[eml_name] = []

    return results


# ── Step 2: Group attachments by type ────────────────────────────────────────

def group_by_type(all_attachments: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Group all attachments by file extension type."""
    grouped: dict[str, list[dict]] = defaultdict(list)

    for eml_name, attachments in all_attachments.items():
        for att in attachments:
            path = Path(att["saved_path"])
            ext = path.suffix.lower()
            att["source_eml"] = eml_name
            grouped[ext].append(att)

    return dict(grouped)


# ── Step 3: Test parsers ────────────────────────────────────────────────────

def test_pdf_classifier(pdf_files: list[dict]) -> tuple[list[PDFClassifierResult], list[ParserResult]]:
    """Test PDF classifier and local PDF parser."""
    from mtss.parsers.pdf_classifier import classify_pdf, PDFComplexity

    classifier_results: list[PDFClassifierResult] = []
    parser_results: list[ParserResult] = []

    # Check if local PDF parser is available
    from mtss.parsers.local_pdf_parser import LocalPDFParser
    pdf_parser = LocalPDFParser()
    pdf_parser_available = pdf_parser.is_available

    logger.info(f"\n{'='*70}")
    logger.info(f"PDF CLASSIFIER + PARSER TEST ({len(pdf_files)} files)")
    logger.info(f"  LocalPDFParser available: {pdf_parser_available}")

    for pdf_info in pdf_files:
        path = Path(pdf_info["saved_path"])
        fname = pdf_info["filename"]
        logger.info(f"\n  Testing: {fname} ({pdf_info['size_bytes']:,} bytes)")

        # Classify
        cr = PDFClassifierResult(filename=fname)
        try:
            complexity = classify_pdf(path)
            cr.classification = complexity.value
            cr.success = True
            logger.info(f"    Classification: {complexity.value}")
        except Exception as e:
            cr.error = str(e)
            logger.error(f"    Classification error: {e}")
        classifier_results.append(cr)

        # Parse if simple and parser available
        pr = ParserResult(
            filename=fname,
            file_type="pdf",
            parser_used="local_pdf" if cr.classification == "simple" else "would_use_llamaparse",
        )
        if cr.classification == "simple" and pdf_parser_available:
            try:
                content = asyncio.run(pdf_parser.parse(path))
                pr.chars_extracted = len(content)
                pr.content_preview = content[:300].replace("\n", " ")
                pr.success = True
                logger.info(f"    Parsed OK: {len(content):,} chars")
                logger.info(f"    Preview: {content[:150].replace(chr(10), ' ')}")
            except Exception as e:
                pr.error = str(e)
                logger.error(f"    Parse error: {e}")
        elif cr.classification == "complex":
            pr.extra["note"] = "Complex PDF - would go to LlamaParse"
            pr.success = True  # classifier worked correctly
            logger.info("    -> Complex: would be sent to LlamaParse")
        elif not pdf_parser_available:
            pr.error = "pymupdf4llm not installed"
            logger.warning("    -> pymupdf4llm not available")
        parser_results.append(pr)

    return classifier_results, parser_results


def test_docx_parser(docx_files: list[dict]) -> list[ParserResult]:
    """Test local DOCX parser."""
    from mtss.parsers.local_office_parser import LocalDocxParser

    parser = LocalDocxParser()
    results: list[ParserResult] = []

    logger.info(f"\n{'='*70}")
    logger.info(f"DOCX PARSER TEST ({len(docx_files)} files)")
    logger.info(f"  LocalDocxParser available: {parser.is_available}")

    for doc_info in docx_files:
        path = Path(doc_info["saved_path"])
        fname = doc_info["filename"]
        logger.info(f"\n  Testing: {fname} ({doc_info['size_bytes']:,} bytes)")

        pr = ParserResult(filename=fname, file_type="docx", parser_used="local_docx")
        if not parser.is_available:
            pr.error = "python-docx not installed"
            results.append(pr)
            continue

        try:
            content = asyncio.run(parser.parse(path))
            pr.chars_extracted = len(content)
            pr.content_preview = content[:300].replace("\n", " ")
            pr.success = True
            logger.info(f"    Parsed OK: {len(content):,} chars")
            logger.info(f"    Preview: {content[:150].replace(chr(10), ' ')}")
        except Exception as e:
            pr.error = str(e)
            logger.error(f"    Parse error: {e}")
        results.append(pr)

    return results


def test_xlsx_parser(xlsx_files: list[dict]) -> list[ParserResult]:
    """Test local XLSX parser."""
    from mtss.parsers.local_office_parser import LocalXlsxParser

    parser = LocalXlsxParser()
    results: list[ParserResult] = []

    logger.info(f"\n{'='*70}")
    logger.info(f"XLSX PARSER TEST ({len(xlsx_files)} files)")
    logger.info(f"  LocalXlsxParser available: {parser.is_available}")

    for xls_info in xlsx_files:
        path = Path(xls_info["saved_path"])
        fname = xls_info["filename"]
        logger.info(f"\n  Testing: {fname} ({xls_info['size_bytes']:,} bytes)")

        pr = ParserResult(filename=fname, file_type="xlsx", parser_used="local_xlsx")
        if not parser.is_available:
            pr.error = "openpyxl not installed"
            results.append(pr)
            continue

        try:
            content = asyncio.run(parser.parse(path))
            pr.chars_extracted = len(content)
            pr.content_preview = content[:300].replace("\n", " ")
            pr.success = True

            # Count sheets
            sheet_count = content.count("## ")
            pr.extra["sheet_count"] = sheet_count

            logger.info(f"    Parsed OK: {len(content):,} chars, {sheet_count} sheets")
            logger.info(f"    Preview: {content[:150].replace(chr(10), ' ')}")
        except Exception as e:
            pr.error = str(e)
            logger.error(f"    Parse error: {e}")
        results.append(pr)

    return results


def test_csv_parser(csv_files: list[dict]) -> list[ParserResult]:
    """Test local CSV parser."""
    from mtss.parsers.local_office_parser import LocalCsvParser

    parser = LocalCsvParser()
    results: list[ParserResult] = []

    logger.info(f"\n{'='*70}")
    logger.info(f"CSV PARSER TEST ({len(csv_files)} files)")

    for csv_info in csv_files:
        path = Path(csv_info["saved_path"])
        fname = csv_info["filename"]
        logger.info(f"\n  Testing: {fname} ({csv_info['size_bytes']:,} bytes)")

        pr = ParserResult(filename=fname, file_type="csv", parser_used="local_csv")
        try:
            content = asyncio.run(parser.parse(path))
            pr.chars_extracted = len(content)
            pr.content_preview = content[:300].replace("\n", " ")
            pr.success = True
            logger.info(f"    Parsed OK: {len(content):,} chars")
            logger.info(f"    Preview: {content[:150].replace(chr(10), ' ')}")
        except Exception as e:
            pr.error = str(e)
            logger.error(f"    Parse error: {e}")
        results.append(pr)

    return results


def test_html_parser(html_files: list[dict]) -> list[ParserResult]:
    """Test local HTML parser."""
    from mtss.parsers.local_office_parser import LocalHtmlParser

    parser = LocalHtmlParser()
    results: list[ParserResult] = []

    logger.info(f"\n{'='*70}")
    logger.info(f"HTML PARSER TEST ({len(html_files)} files)")

    for html_info in html_files:
        path = Path(html_info["saved_path"])
        fname = html_info["filename"]
        logger.info(f"\n  Testing: {fname} ({html_info['size_bytes']:,} bytes)")

        pr = ParserResult(filename=fname, file_type="html", parser_used="local_html")
        try:
            content = asyncio.run(parser.parse(path))
            pr.chars_extracted = len(content)
            pr.content_preview = content[:300].replace("\n", " ")
            pr.success = True
            logger.info(f"    Parsed OK: {len(content):,} chars")
            logger.info(f"    Preview: {content[:150].replace(chr(10), ' ')}")
        except Exception as e:
            pr.error = str(e)
            logger.error(f"    Parse error: {e}")
        results.append(pr)

    return results


def test_image_filter(image_files: list[dict]) -> list[ImageFilterResult]:
    """Test image filter on extracted images."""
    from mtss.image_filter import is_meaningful_image

    results: list[ImageFilterResult] = []

    logger.info(f"\n{'='*70}")
    logger.info(f"IMAGE FILTER TEST ({len(image_files)} files)")

    # Check PIL availability
    pil_available = False
    try:
        from PIL import Image
        pil_available = True
    except ImportError:
        pass
    logger.info(f"  PIL available: {pil_available}")

    for img_info in image_files:
        path = Path(img_info["saved_path"])
        fname = img_info["filename"]

        ir = ImageFilterResult(filename=fname, size_bytes=img_info["size_bytes"])

        # Get dimensions if PIL available
        if pil_available:
            try:
                from PIL import Image
                with Image.open(path) as im:
                    w, h = im.size
                    ir.dimensions = f"{w}x{h}"
            except Exception as e:
                ir.dimensions = f"error: {e}"

        # Run filter
        try:
            ir.is_meaningful = is_meaningful_image(path)
            # Determine why it was filtered
            if not ir.is_meaningful:
                from mtss.image_filter import _SKIP_FILENAME_PATTERNS
                stem = path.stem
                for pat in _SKIP_FILENAME_PATTERNS:
                    if pat.search(stem):
                        ir.reason = f"filename matches pattern: {pat.pattern}"
                        break
                if not ir.reason and img_info["size_bytes"] < 15_000:
                    ir.reason = f"file too small ({img_info['size_bytes']} < 15000 bytes)"
                if not ir.reason:
                    ir.reason = "dimensions too small or banner-shaped"
            else:
                ir.reason = "passed all checks"
        except Exception as e:
            ir.reason = f"error: {e}"

        logger.info(
            f"  {fname}: {ir.dimensions}, {ir.size_bytes:,} bytes, "
            f"meaningful={ir.is_meaningful}, reason={ir.reason}"
        )
        results.append(ir)

    return results


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 70)
    logger.info("PARSER VALIDATION - Testing local parsers against real EML attachments")
    logger.info("=" * 70)

    with tempfile.TemporaryDirectory(prefix="mtss_parser_validation_") as tmp:
        tmp_dir = Path(tmp)

        # Step 1: Extract attachments
        logger.info("\n\nSTEP 1: EXTRACTING ATTACHMENTS FROM EML FILES")
        all_attachments = extract_attachments(tmp_dir)

        total_attachments = sum(len(v) for v in all_attachments.values())
        logger.info(f"\nTotal attachments extracted: {total_attachments}")

        if total_attachments == 0:
            logger.error("No attachments found! Cannot validate parsers.")
            return

        # Step 2: Group by type
        grouped = group_by_type(all_attachments)
        logger.info("\nAttachments by type:")
        for ext, files in sorted(grouped.items()):
            logger.info(f"  {ext}: {len(files)} files")

        # Step 3: Test each parser
        all_parser_results: list[ParserResult] = []
        classifier_results: list[PDFClassifierResult] = []
        image_results: list[ImageFilterResult] = []

        # PDFs
        pdf_exts = [".pdf"]
        pdf_files = []
        for ext in pdf_exts:
            pdf_files.extend(grouped.get(ext, []))
        if pdf_files:
            cr, pr = test_pdf_classifier(pdf_files)
            classifier_results.extend(cr)
            all_parser_results.extend(pr)
        else:
            logger.info("\nNo PDF files found in attachments.")

        # DOCX
        docx_files = grouped.get(".docx", [])
        if docx_files:
            all_parser_results.extend(test_docx_parser(docx_files))
        else:
            logger.info("\nNo DOCX files found in attachments.")

        # XLSX
        xlsx_files = grouped.get(".xlsx", [])
        if xlsx_files:
            all_parser_results.extend(test_xlsx_parser(xlsx_files))
        else:
            logger.info("\nNo XLSX files found in attachments.")

        # XLS (legacy - note that local parser doesn't handle .xls)
        xls_files = grouped.get(".xls", [])
        if xls_files:
            logger.info(f"\n{'='*70}")
            logger.info(f"XLS FILES ({len(xls_files)} files) - Legacy format, no local parser")
            for xls_info in xls_files:
                pr = ParserResult(
                    filename=xls_info["filename"],
                    file_type="xls",
                    parser_used="none (legacy format - needs LlamaParse)",
                    extra={"note": "Legacy .xls format not handled by local parsers"},
                )
                all_parser_results.append(pr)
                logger.info(f"  {xls_info['filename']} ({xls_info['size_bytes']:,} bytes) - skipped")

        # CSV
        csv_files = grouped.get(".csv", [])
        if csv_files:
            all_parser_results.extend(test_csv_parser(csv_files))
        else:
            logger.info("\nNo CSV files found in attachments.")

        # HTML / HTM
        html_files = grouped.get(".html", []) + grouped.get(".htm", [])
        if html_files:
            all_parser_results.extend(test_html_parser(html_files))
        else:
            logger.info("\nNo HTML files found in attachments.")

        # Images
        img_exts = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp"]
        image_files = []
        for ext in img_exts:
            image_files.extend(grouped.get(ext, []))
        if image_files:
            image_results = test_image_filter(image_files)
        else:
            logger.info("\nNo image files found in attachments.")

        # DOC (legacy)
        doc_files = grouped.get(".doc", [])
        if doc_files:
            logger.info(f"\n{'='*70}")
            logger.info(f"DOC FILES ({len(doc_files)} files) - Legacy format, no local parser")
            for doc_info in doc_files:
                pr = ParserResult(
                    filename=doc_info["filename"],
                    file_type="doc",
                    parser_used="none (legacy format - needs LlamaParse)",
                    extra={"note": "Legacy .doc format not handled by local parsers"},
                )
                all_parser_results.append(pr)
                logger.info(f"  {doc_info['filename']} ({doc_info['size_bytes']:,} bytes) - skipped")

        # Other types we didn't test
        tested_exts = set(pdf_exts + [".docx", ".xlsx", ".xls", ".csv", ".html", ".htm", ".doc"] + img_exts)
        other_exts = set(grouped.keys()) - tested_exts
        if other_exts:
            logger.info(f"\nOther attachment types found (not tested): {sorted(other_exts)}")
            for ext in sorted(other_exts):
                for f in grouped[ext]:
                    pr = ParserResult(
                        filename=f["filename"],
                        file_type=ext,
                        parser_used="none (no local parser for this type)",
                    )
                    all_parser_results.append(pr)

        # Step 4: Generate report
        logger.info(f"\n\n{'='*70}")
        logger.info("GENERATING REPORT")
        generate_report(all_attachments, all_parser_results, classifier_results, image_results, grouped)


def generate_report(
    all_attachments: dict,
    parser_results: list[ParserResult],
    classifier_results: list[PDFClassifierResult],
    image_results: list[ImageFilterResult],
    grouped: dict,
):
    """Generate markdown report file."""
    lines: list[str] = []
    lines.append("# Parser Validation Results")
    lines.append("")
    lines.append("Validation of local parsers against real attachments extracted from EML files.")
    lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")
    total_files = sum(len(v) for v in all_attachments.values())
    total_attachments = sum(len(v) for v in all_attachments.values())
    lines.append(f"- **EML files tested:** {len(all_attachments)}")
    lines.append(f"- **Total attachments extracted:** {total_attachments}")
    lines.append(f"- **Attachment types found:** {', '.join(sorted(grouped.keys()))}")
    lines.append("")

    # Attachments per EML
    lines.append("### Attachments per EML File")
    lines.append("")
    lines.append("| EML File | Attachments | Types |")
    lines.append("|----------|------------|-------|")
    for eml_name, atts in all_attachments.items():
        types = set(Path(a["saved_path"]).suffix.lower() for a in atts)
        lines.append(f"| {eml_name} | {len(atts)} | {', '.join(sorted(types)) if types else 'none'} |")
    lines.append("")

    # Type distribution
    lines.append("### Attachment Type Distribution")
    lines.append("")
    lines.append("| Extension | Count | Total Size |")
    lines.append("|-----------|-------|------------|")
    for ext in sorted(grouped.keys()):
        files = grouped[ext]
        total_size = sum(f["size_bytes"] for f in files)
        lines.append(f"| {ext} | {len(files)} | {total_size:,} bytes |")
    lines.append("")

    # PDF Classifier Results
    if classifier_results:
        lines.append("## PDF Classifier Results")
        lines.append("")
        simple_count = sum(1 for c in classifier_results if c.classification == "simple")
        complex_count = sum(1 for c in classifier_results if c.classification == "complex")
        error_count = sum(1 for c in classifier_results if not c.success)
        lines.append(f"- **Simple PDFs:** {simple_count}")
        lines.append(f"- **Complex PDFs:** {complex_count}")
        lines.append(f"- **Errors:** {error_count}")
        lines.append("")
        lines.append("| Filename | Classification | Error |")
        lines.append("|----------|---------------|-------|")
        for cr in classifier_results:
            err = cr.error if cr.error else "-"
            lines.append(f"| {cr.filename} | {cr.classification or 'error'} | {err} |")
        lines.append("")

    # Parser Results Table
    lines.append("## Parser Results")
    lines.append("")
    success_count = sum(1 for p in parser_results if p.success)
    fail_count = sum(1 for p in parser_results if not p.success)
    lines.append(f"- **Successful parses:** {success_count}")
    lines.append(f"- **Failed parses:** {fail_count}")
    lines.append("")
    lines.append("| Filename | Type | Parser | Chars | Success | Error/Notes |")
    lines.append("|----------|------|--------|-------|---------|-------------|")
    for pr in parser_results:
        notes = pr.error if pr.error else (pr.extra.get("note", "-"))
        if pr.extra.get("sheet_count"):
            notes = f"{pr.extra['sheet_count']} sheets"
        lines.append(
            f"| {pr.filename} | {pr.file_type} | {pr.parser_used} | "
            f"{pr.chars_extracted:,} | {'YES' if pr.success else 'NO'} | {notes} |"
        )
    lines.append("")

    # Content previews for successful parses
    successful = [p for p in parser_results if p.success and p.chars_extracted > 0]
    if successful:
        lines.append("### Content Quality Samples")
        lines.append("")
        for pr in successful:
            lines.append(f"**{pr.filename}** ({pr.file_type}, {pr.chars_extracted:,} chars):")
            lines.append("```")
            lines.append(pr.content_preview[:200] if pr.content_preview else "(empty)")
            lines.append("```")
            lines.append("")

    # Image Filter Results
    if image_results:
        lines.append("## Image Filter Results")
        lines.append("")
        meaningful_count = sum(1 for i in image_results if i.is_meaningful)
        filtered_count = sum(1 for i in image_results if not i.is_meaningful)
        lines.append(f"- **Meaningful images (kept):** {meaningful_count}")
        lines.append(f"- **Filtered out:** {filtered_count}")
        lines.append("")
        lines.append("| Filename | Size | Dimensions | Meaningful? | Reason |")
        lines.append("|----------|------|------------|-------------|--------|")
        for ir in image_results:
            lines.append(
                f"| {ir.filename} | {ir.size_bytes:,} | {ir.dimensions} | "
                f"{'YES' if ir.is_meaningful else 'NO'} | {ir.reason} |"
            )
        lines.append("")

    # Quality Assessment
    lines.append("## Quality Assessment")
    lines.append("")

    # Check for empty outputs
    empty_results = [p for p in parser_results if p.success and p.chars_extracted == 0 and p.parser_used not in ("would_use_llamaparse", "none (legacy format - needs LlamaParse)", "none (no local parser for this type)")]
    if empty_results:
        lines.append("### Empty Outputs (potential issues)")
        lines.append("")
        for pr in empty_results:
            lines.append(f"- **{pr.filename}**: Parser returned 0 chars")
        lines.append("")

    # Check for very short outputs
    short_results = [p for p in parser_results if p.success and 0 < p.chars_extracted < 100]
    if short_results:
        lines.append("### Very Short Outputs (<100 chars)")
        lines.append("")
        for pr in short_results:
            lines.append(f"- **{pr.filename}**: Only {pr.chars_extracted} chars extracted")
        lines.append("")

    # Overall assessment
    lines.append("### Overall RAG Usability")
    lines.append("")
    if successful:
        avg_chars = sum(p.chars_extracted for p in successful) / len(successful)
        lines.append(f"- Average chars per successful parse: {avg_chars:,.0f}")
        lines.append(f"- Total successful parses: {success_count}/{len(parser_results)}")

        if all(p.chars_extracted > 50 for p in successful):
            lines.append("- All successful parses produced substantive content (>50 chars)")
        else:
            very_short = [p for p in successful if p.chars_extracted <= 50]
            lines.append(f"- {len(very_short)} successful parses produced very little content (<=50 chars)")
    lines.append("")

    # Recommendations
    lines.append("## Recommendations")
    lines.append("")

    recommendations = []
    if any(p.error and "not installed" in p.error for p in parser_results):
        missing = set()
        for p in parser_results:
            if p.error and "not installed" in p.error:
                missing.add(p.error)
        for m in sorted(missing):
            recommendations.append(f"- Install missing dependency: {m}")

    if fail_count > 0:
        error_types = defaultdict(int)
        for p in parser_results:
            if p.error and "not installed" not in p.error:
                error_types[p.error] += 1
        for err, count in error_types.items():
            recommendations.append(f"- Fix parser error ({count} files): {err[:100]}")

    if not recommendations:
        recommendations.append("- All tested parsers are working correctly.")

    for r in recommendations:
        lines.append(r)
    lines.append("")

    # Write report
    report_path = PROJECT_ROOT / "docs" / "investigation" / "plans" / "parser-validation-results.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"\nReport written to: {report_path}")


if __name__ == "__main__":
    main()
