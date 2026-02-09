import re
import pdfplumber
from typing import Dict, List


def detect_toc_page(pages_text: List[Dict]) -> int:
    """Find the page that likely contains the Table of Contents. Returns 0-based index or -1."""
    scan_limit = min(20, max(10, len(pages_text) // 7))

    for idx, page_data in enumerate(pages_text[:scan_limit]):
        text = page_data.get("text", "")
        if not text:
            continue

        first_300 = text.lower()[:300]
        toc_indicators = ["table of contents", "contents", "table of content"]
        has_toc_marker = any(ind in first_300 for ind in toc_indicators)

        lines = text.split("\n")
        numbered_lines = sum(
            1 for line in lines
            if re.match(r"^\s*(?:Chapter\s+)?\d+[\.\:)]", line.strip(), re.IGNORECASE)
        )

        if numbered_lines >= 5 or (has_toc_marker and numbered_lines >= 3):
            return idx

    return -1


def extract_toc_from_page(page_text: str) -> List[Dict]:
    """Extract TOC entries from a page. Returns list of {section, page_num?}."""
    toc = []
    seen_numbers = set()

    for line in page_text.split("\n"):
        line = line.strip()
        if len(line) < 10 or line.isdigit() or line.lower() in ["contents", "table of contents"]:
            continue

        # Pattern: "Chapter N: Title ..... page"
        chapter_match = re.match(
            r"^Chapter\s+(\d+)[\:\s]+(.+?)[\s\.]+(\d+)\s*$", line, re.IGNORECASE
        )
        if chapter_match:
            num, title, page_num = chapter_match.group(1), chapter_match.group(2).strip(), int(chapter_match.group(3))
            if num in seen_numbers and len(toc) >= 5:
                return toc
            seen_numbers.add(num)
            title = re.sub(r"\.+$", "", title).strip()
            if len(title) > 3:
                toc.append({"section": f"Chapter {num}: {title}", "page_num": page_num})
                continue

        # Pattern: "N. Title ..... page" or "N.M Title ..... page"
        numbered_match = re.match(
            r"^(\d+(?:\.\d+)?)[\.\s]+([A-Z].+?)[\s\.]+(\d+)\s*$", line, re.IGNORECASE
        )
        if numbered_match:
            num = numbered_match.group(1).split(".")[0]
            title = numbered_match.group(2).strip()
            page_num = int(numbered_match.group(3))
            if num in seen_numbers and len(toc) >= 5:
                return toc
            seen_numbers.add(num)
            title = re.sub(r"\.+$", "", title).strip()
            if len(title) > 5 and title[0].isupper():
                toc.append({"section": f"{numbered_match.group(1)} {title}", "page_num": page_num})
                continue

    return toc


def scan_for_headings(pages_text: List[Dict]) -> List[Dict]:
    """Fallback: scan document for chapter/section headings."""
    headings = []
    seen_numbers = set()

    for page_data in pages_text:
        text = page_data.get("text", "")
        lines = text.split("\n")

        for i, line in enumerate(lines):
            line = line.strip()
            if len(line) < 8 or len(line) > 100:
                continue

            # "Chapter N" pattern
            chapter_match = re.match(r"^Chapter\s+(\d+)[\:\s]*(.*)$", line, re.IGNORECASE)
            if chapter_match:
                num = chapter_match.group(1)
                title = chapter_match.group(2).strip()
                if num in seen_numbers and len(headings) >= 5:
                    return headings
                seen_numbers.add(num)
                heading = f"Chapter {num}: {title}" if title else f"Chapter {num}"
                if heading not in [h["section"] for h in headings]:
                    headings.append({"section": heading})
                if len(headings) >= 20:
                    return headings
                continue

            # Numbered sections: "N. Title" or "N.M Title"
            numbered_match = re.match(r"^(\d+(?:\.\d+)?)[\.\s]+([A-Z][A-Za-z\s]{3,60})$", line)
            if numbered_match:
                num = numbered_match.group(1).split(".")[0]
                title = numbered_match.group(2).strip()
                if "," in title or ";" in title or title.endswith("."):
                    continue
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line and next_line[0].islower():
                        continue
                if num in seen_numbers and len(headings) >= 5:
                    return headings
                seen_numbers.add(num)
                heading = f"{numbered_match.group(1)} {title}"
                if heading not in [h["section"] for h in headings]:
                    headings.append({"section": heading})
                if len(headings) >= 20:
                    return headings

    return headings


def detect_headings_from_text(pages_text: List[Dict]) -> List[Dict]:
    """Smart TOC extraction with multiple strategies."""
    # Strategy 1: Find explicit TOC page
    toc_page_idx = detect_toc_page(pages_text)
    if toc_page_idx != -1:
        toc = extract_toc_from_page(pages_text[toc_page_idx].get("text", ""))
        if len(toc) >= 3:
            return toc

    # Strategy 2: Scan for chapter headings
    headings = scan_for_headings(pages_text)
    if headings:
        return headings

    # Strategy 3: Fallback
    return [{"section": "Content"}]


def map_sections_to_pages(toc: List[Dict], pages_text: List[Dict]) -> List[Dict]:
    """Map each TOC section to page ranges.

    If TOC entries have explicit page numbers, use them directly.
    Otherwise, evenly distribute pages across sections.
    """
    enhanced_toc = []
    total_pages = len(pages_text)
    has_page_nums = all("page_num" in entry for entry in toc)

    if has_page_nums:
        for i, entry in enumerate(toc):
            page_start = entry["page_num"]
            page_end = toc[i + 1]["page_num"] - 1 if i < len(toc) - 1 else total_pages
            enhanced_toc.append({
                "section": entry["section"],
                "page_start": page_start,
                "page_end": max(page_end, page_start),
            })
    else:
        # No page numbers: distribute pages evenly across sections
        pages_per_section = max(1, total_pages // len(toc))
        for i, entry in enumerate(toc):
            page_start = i * pages_per_section + 1
            if i < len(toc) - 1:
                page_end = (i + 1) * pages_per_section
            else:
                page_end = total_pages
            enhanced_toc.append({
                "section": entry["section"],
                "page_start": page_start,
                "page_end": min(page_end, total_pages),
            })

    return enhanced_toc


def table_to_text(table_data: List[List]) -> str:
    """Convert table to markdown-like text for embedding."""
    if not table_data:
        return ""
    rows = []
    for row in table_data:
        if row:
            cleaned = [str(cell) if cell is not None else "" for cell in row]
            rows.append(" | ".join(cleaned))
    return "\n".join(rows)


def extract_pdf_content(pdf_path: str) -> Dict:
    """Extract text, tables, and TOC from PDF with section-to-page mapping."""
    pages_text = []
    tables = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages_text.append({"page_num": i + 1, "text": text})
            try:
                page_tables = page.extract_tables()
                if page_tables:
                    tables.extend([{"page": i + 1, "data": t} for t in page_tables if t])
            except Exception:
                pass

    toc = detect_headings_from_text(pages_text)
    toc_with_pages = map_sections_to_pages(toc, pages_text)

    return {
        "pages_text": pages_text,
        "tables": tables,
        "toc": toc,
        "toc_with_pages": toc_with_pages,
    }


def extract_section_text(pages_text: List[Dict], page_start: int, page_end: int) -> str:
    """Extract text from a specific page range."""
    section_texts = []
    for page_data in pages_text:
        if page_start <= page_data["page_num"] <= page_end:
            section_texts.append(page_data["text"])
    return "\n\n".join(section_texts)
