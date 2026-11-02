"""Turn the MDCG guidance PDFs into searchable passages ("chunks").

Each chunk keeps its document and page numbers, so every answer can cite exactly where it came from. Text is extracted page by page with pdftotext (poppler); repeated page headers and page
numbers are removed; a new chunk starts at each numbered section heading, and chunks are cut at about
CHUNK_WORDS words on paragraph boundaries, with one paragraph of overlap.
"""
import csv
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "data" / "pdf"
MANIFEST = ROOT / "data" / "manifest.csv"
CHUNKS = ROOT / "data" / "chunks.jsonl"
CHUNK_WORDS = 220
# Footnotes also start with a number; these openings mark them as footnotes, not section headings
FOOTNOTE_START = re.compile(r"(Article|Articles|See|Regulation|Directive|WMA|Definition of|List of|To |In this "
                            r"context|Commission|OJ|https?:|As |For |The term|Annex [IVX]+, section|Table of)")
HEADING = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,3})\.?\s+([A-Z][^\n]{2,90})$")


def load_manifest(path=MANIFEST):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def page_texts(pdf):
    n = int(re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", str(pdf)], capture_output=True,
                                                      text=True).stdout).group(1))
    pages = []
    for i in range(1, n + 1):
        out = subprocess.run(["pdftotext", "-f", str(i), "-l", str(i), str(pdf), "-"],
                             capture_output=True, text=True).stdout
        pages.append(out)
    return pages


def clean_pages(pages):
    """Drop lines that repeat on many pages (running headers) and bare page numbers."""
    counts = Counter(line.strip() for p in pages for line in set(p.splitlines()) if line.strip())
    repeated = {line for line, c in counts.items() if c >= max(3, len(pages) // 3) and len(line) < 120}
    cleaned = []
    for p in pages:
        keep = [ln.rstrip() for ln in p.replace("\f", "").splitlines()
                if ln.strip() and ln.strip() not in repeated
                and not re.fullmatch(r"\d{1,3}", ln.strip())
                and not re.fullmatch(r"Page \d+ of \d+", ln.strip())]
        cleaned.append(keep)
    return cleaned


def is_heading_line(s):
    """A short numbered line that is not a sentence or a table-of-contents entry (those end in a page number)."""
    m = HEADING.match(s)
    return (bool(m) and len(s) < 100 and not s.endswith((".", ":", ","))
            and not re.search(r"(\.{3,}|\s\d{1,3})$", s)
            and not FOOTNOTE_START.match(m.group(2)))


def paragraphs(lines):
    """Join wrapped lines into paragraphs; headings and bullets start new ones."""
    paras, cur = [], []
    for ln in lines:
        s = ln.strip()
        starts_new = (is_heading_line(s) or s in ("•", "-", "▪", "o")
                      or re.match(r"^(•|▪|–|-|\(?[a-z]\)|\d+\.)\s", s)
                      or (cur and cur[-1].endswith((".", ":", ";")) and s[:1].isupper()))
        if starts_new and cur:
            paras.append(" ".join(cur))
            cur = []
        if s in ("•", "▪"):
            cur = ["•"]
            continue
        if is_heading_line(s):
            paras.append(s)
            continue
        cur.append(s)
    if cur:
        paras.append(" ".join(cur))
    return [re.sub(r"\s+", " ", p).replace("• ", "• ").strip() for p in paras if p.strip()]


def is_next_heading(prev, num):
    """Accept a numbered heading only if it plausibly follows the previous one (1 -> 1.1, 1.2 or 2).
    This rejects footnotes and numbered list items, which also start with a number."""
    if not prev:
        return num in ((1,), (2,), (1, 1))
    if num == prev + (1,):
        return True
    for depth in range(len(prev), 0, -1):
        if num == prev[:depth - 1] + (prev[depth - 1] + 1,):
            return True
    return False


def chunk_document(meta, pages):
    """Yield chunks for one document. page numbers are 1-based PDF pages."""
    lines_by_page = clean_pages(pages)
    section = "Introduction"
    last_num = ()
    buf, buf_pages = [], []
    chunks = []

    def flush(keep_last=True):
        nonlocal buf, buf_pages
        if not buf:
            return
        text = " ".join(buf)
        is_contents_page = text.count("....") >= 3            # table of contents: dot leaders
        if len(text.split()) >= 25 and not is_contents_page:
            chunks.append({"chunk_id": f"{meta['doc_id']}#{len(chunks) + 1:03d}", "doc_id": meta["doc_id"],
                           "title": meta["title"],
                           "page_start": min(buf_pages), "page_end": max(buf_pages), "text": text})
        if keep_last and len(buf) > 1 and len(buf[-1].split()) < CHUNK_WORDS // 2:
            buf, buf_pages = [buf[-1]], [buf_pages[-1]]
        else:
            buf, buf_pages = [], []

    for pno, lines in enumerate(lines_by_page, start=1):
        for para in paragraphs(lines):
            m = HEADING.match(para) if is_heading_line(para) else None
            num = tuple(int(x) for x in m.group(1).split(".")) if m else None
            if m and is_next_heading(last_num, num):
                last_num = num
                flush(keep_last=False)
                section = f"{m.group(1)} {m.group(2).strip()}"
                continue
            buf.append(para)
            buf_pages.append(pno)
            if sum(len(p.split()) for p in buf) >= CHUNK_WORDS:
                flush()
    flush(keep_last=False)
    return chunks


def build(pdf_dir=PDF_DIR, out=CHUNKS):
    all_chunks = []
    for meta in load_manifest():
        pdf = Path(pdf_dir) / meta["file"]
        if not pdf.exists():
            raise FileNotFoundError(f"{pdf} missing: download it from {meta['url']}")
        all_chunks += chunk_document(meta, page_texts(pdf))
    with open(out, "w", encoding="utf-8") as fh:
        for c in all_chunks:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    return all_chunks


def load_chunks(path=CHUNKS):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


if __name__ == "__main__":
    chunks = build()
    print(f"{len(chunks)} chunks written to {CHUNKS.relative_to(ROOT)}")
