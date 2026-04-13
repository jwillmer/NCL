"""Extended parser validation - test DOCX/XLSX parsers and force-test PDF parser.

Supplements the main validation by targeting EML files with office attachments
and testing the PDF parser regardless of classifier result.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# EML files with DOCX/XLSX attachments (found via peek)
EXTRA_EML_FILES = [
    "100312568_f14vs02x.kpf.eml",  # DOCX - low insulation explanation
    "100315058_juuqiiay.aa3.eml",  # DOCX - maritime service request form
    "100297754_xz31rifr.qvh.eml",  # XLSX - CT engine
    "100298134_g2wuhyot.jpk.eml",  # XLSX - SIMOPS plan
    "100299464_gejziqm1.mqc.eml",  # XLSX - liquid weight
    "100297240_nemelpl4.gm1.eml",  # XLS (legacy) - boiler logs
]

EML_DIR = PROJECT_ROOT / "data" / "emails"


def main():
    logger.info("=" * 70)
    logger.info("EXTENDED PARSER VALIDATION - DOCX/XLSX + Force PDF parse")
    logger.info("=" * 70)

    results_lines: list[str] = []

    with tempfile.TemporaryDirectory(prefix="mtss_ext_val_") as tmp:
        tmp_dir = Path(tmp)

        # ── Extract attachments from extra EML files ──────────────────────
        from mtss.parsers.eml_parser import EMLParser

        eml_parser = EMLParser(attachments_dir=tmp_dir)
        all_attachments: dict[str, list[dict]] = {}

        for eml_name in EXTRA_EML_FILES:
            eml_path = EML_DIR / eml_name
            if not eml_path.exists():
                logger.warning(f"Not found: {eml_path}")
                continue
            try:
                parsed = eml_parser.parse_file(eml_path)
                atts = []
                for att in parsed.attachments:
                    atts.append({
                        "filename": att.filename,
                        "content_type": att.content_type,
                        "size_bytes": att.size_bytes,
                        "saved_path": att.saved_path,
                    })
                    logger.info(f"  {eml_name} -> {att.filename} ({att.content_type}, {att.size_bytes:,} bytes)")
                all_attachments[eml_name] = atts
            except Exception as e:
                logger.error(f"  Failed: {eml_name}: {e}")

        # Group by extension
        grouped: dict[str, list[dict]] = defaultdict(list)
        for eml_name, atts in all_attachments.items():
            for att in atts:
                ext = Path(att["saved_path"]).suffix.lower()
                att["source_eml"] = eml_name
                grouped[ext].append(att)

        logger.info(f"\nExtracted types: {dict((k, len(v)) for k, v in sorted(grouped.items()))}")

        # ── Test DOCX parser ──────────────────────────────────────────────
        docx_files = grouped.get(".docx", [])
        if docx_files:
            logger.info(f"\n{'='*70}")
            logger.info(f"DOCX PARSER TEST ({len(docx_files)} files)")
            from mtss.parsers.local_office_parser import LocalDocxParser
            docx_parser = LocalDocxParser()
            logger.info(f"  Available: {docx_parser.is_available}")

            for f in docx_files:
                path = Path(f["saved_path"])
                fname = f["filename"]
                logger.info(f"\n  {fname} ({f['size_bytes']:,} bytes)")
                try:
                    content = asyncio.run(docx_parser.parse(path))
                    logger.info(f"    OK: {len(content):,} chars")
                    logger.info(f"    Preview: {content[:200].replace(chr(10), ' ')}")

                    # Check for tables
                    has_tables = " | " in content
                    para_count = content.count("\n\n")
                    logger.info(f"    Has tables: {has_tables}, Paragraphs: ~{para_count}")

                    results_lines.append(
                        f"| {fname} | docx | local_docx | {len(content):,} | YES | "
                        f"tables={has_tables}, paras~{para_count} |"
                    )
                except Exception as e:
                    logger.error(f"    FAILED: {e}")
                    results_lines.append(f"| {fname} | docx | local_docx | 0 | NO | {e} |")

        # ── Test XLSX parser ──────────────────────────────────────────────
        xlsx_files = grouped.get(".xlsx", [])
        if xlsx_files:
            logger.info(f"\n{'='*70}")
            logger.info(f"XLSX PARSER TEST ({len(xlsx_files)} files)")
            from mtss.parsers.local_office_parser import LocalXlsxParser
            xlsx_parser = LocalXlsxParser()
            logger.info(f"  Available: {xlsx_parser.is_available}")

            for f in xlsx_files:
                path = Path(f["saved_path"])
                fname = f["filename"]
                logger.info(f"\n  {fname} ({f['size_bytes']:,} bytes)")
                try:
                    content = asyncio.run(xlsx_parser.parse(path))
                    sheet_count = content.count("## ")
                    row_count = content.count("\n")
                    logger.info(f"    OK: {len(content):,} chars, {sheet_count} sheets, ~{row_count} rows")
                    logger.info(f"    Preview: {content[:200].replace(chr(10), ' ')}")
                    results_lines.append(
                        f"| {fname} | xlsx | local_xlsx | {len(content):,} | YES | "
                        f"{sheet_count} sheets, ~{row_count} rows |"
                    )
                except Exception as e:
                    logger.error(f"    FAILED: {e}")
                    results_lines.append(f"| {fname} | xlsx | local_xlsx | 0 | NO | {e} |")

        # ── Test XLS (legacy) with XLSX parser (should fail gracefully) ───
        xls_files = grouped.get(".xls", [])
        if xls_files:
            logger.info(f"\n{'='*70}")
            logger.info(f"XLS (LEGACY) TEST ({len(xls_files)} files)")
            from mtss.parsers.local_office_parser import LocalXlsxParser
            xlsx_parser = LocalXlsxParser()

            for f in xls_files:
                path = Path(f["saved_path"])
                fname = f["filename"]
                logger.info(f"\n  {fname} ({f['size_bytes']:,} bytes)")
                try:
                    content = asyncio.run(xlsx_parser.parse(path))
                    logger.info(f"    OK: {len(content):,} chars (unexpectedly worked!)")
                    results_lines.append(
                        f"| {fname} | xls | local_xlsx | {len(content):,} | YES | legacy format worked |"
                    )
                except Exception as e:
                    logger.info(f"    Expected failure: {type(e).__name__}: {str(e)[:100]}")
                    results_lines.append(
                        f"| {fname} | xls | local_xlsx | 0 | NO (expected) | needs LlamaParse |"
                    )

        # ── Force-test PDF local parser on all PDFs from first script ─────
        # Re-extract PDFs from the original 6 EML files
        logger.info(f"\n{'='*70}")
        logger.info("FORCE PDF PARSE TEST (testing LocalPDFParser on complex PDFs)")

        original_emls = [
            "100000922_vp4qcvdw.abx.eml",
            "100000366_qhlztvyi.alr.eml",
            "100000376_lgqfor05.2fi.eml",
        ]

        from mtss.parsers.local_pdf_parser import LocalPDFParser
        from mtss.parsers.pdf_classifier import classify_pdf

        pdf_parser = LocalPDFParser()
        logger.info(f"  LocalPDFParser available: {pdf_parser.is_available}")

        for eml_name in original_emls:
            eml_path = EML_DIR / eml_name
            if not eml_path.exists():
                continue
            parsed = eml_parser.parse_file(eml_path)
            for att in parsed.attachments:
                if not att.filename.lower().endswith(".pdf"):
                    continue
                path = Path(att.saved_path)
                fname = att.filename
                logger.info(f"\n  {fname} ({att.size_bytes:,} bytes)")

                # Classify
                complexity = classify_pdf(path)
                logger.info(f"    Classified as: {complexity.value}")

                # Force parse regardless
                if pdf_parser.is_available:
                    try:
                        content = asyncio.run(pdf_parser.parse(path))
                        logger.info(f"    Force-parsed: {len(content):,} chars")
                        logger.info(f"    Preview: {content[:200].replace(chr(10), ' ')}")

                        # Quality check
                        lines = content.strip().split("\n")
                        non_empty_lines = [l for l in lines if l.strip()]
                        logger.info(f"    Lines: {len(lines)}, non-empty: {len(non_empty_lines)}")

                        results_lines.append(
                            f"| {fname} | pdf | local_pdf (force) | {len(content):,} | YES | "
                            f"classified={complexity.value}, {len(non_empty_lines)} non-empty lines |"
                        )
                    except Exception as e:
                        logger.error(f"    Force-parse FAILED: {e}")
                        results_lines.append(
                            f"| {fname} | pdf | local_pdf (force) | 0 | NO | {e} |"
                        )

        # ── Also check image002.png (large image filtered by name pattern) ─
        logger.info(f"\n{'='*70}")
        logger.info("IMAGE FILTER EDGE CASE: image002.png from HERMIONE")

        hermione_eml = EML_DIR / "100297780_avnjkyr4.40y.eml"
        parsed = eml_parser.parse_file(hermione_eml)
        for att in parsed.attachments:
            if att.filename == "image002.png":
                from PIL import Image
                from mtss.image_filter import is_meaningful_image

                path = Path(att.saved_path)
                result = is_meaningful_image(path)
                with Image.open(path) as im:
                    w, h = im.size
                logger.info(
                    f"  image002.png: {w}x{h}, {att.size_bytes:,} bytes, "
                    f"meaningful={result}"
                )
                logger.info(
                    f"  NOTE: This 243KB image is filtered ONLY because filename "
                    f"matches image\\d{{3}} pattern. It might be a meaningful diagram."
                )

        # ── Print summary table ──────────────────────────────────────────
        logger.info(f"\n\n{'='*70}")
        logger.info("EXTENDED RESULTS TABLE")
        logger.info("| Filename | Type | Parser | Chars | Success | Notes |")
        logger.info("|----------|------|--------|-------|---------|-------|")
        for line in results_lines:
            logger.info(line)


if __name__ == "__main__":
    main()
