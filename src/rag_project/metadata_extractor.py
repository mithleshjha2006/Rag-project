import re
from typing import Optional

from .schemas import Document

YEAR_PATTERN = re.compile(r"\b(?:199\d|20[0-2]\d)\b")
NUMBERED_START = re.compile(r"^\d+(\.\d+)*\.?\s+[A-Z]")

ABSTRACT_STOPS = ("introduction", "keywords", "key words", "index terms")
JUNK_WORDS = (
    "university", "department", "institute", "research", "laboratory",
    "school", "labs", "proceedings", "journal", "association",
    "college", "center", "centre", "publishers",
)
MAX_ABSTRACT_CHARS = 2000


class MetadataExtractor:
    def extract(self, doc: Document) -> Document:
        lines = self._get_lines(doc)
        title_lines = self._title_lines(lines)

        return doc.model_copy(update={
            "title": " ".join(title_lines),
            "authors": self._find_authors(lines, title_lines),
            "year": self._find_year(lines),
            "abstract": self._find_abstract(lines),
        })

    # ---------- helpers ----------

    def _get_lines(self, doc: Document) -> list[str]:
        text = doc.pages[0].text
        return [line.strip() for line in text.split("\n") if line.strip()]

    def _is_abstract_line(self, line: str) -> bool:
        return line.lower().strip(" .:") == "abstract"

    def _is_junk(self, line: str) -> bool:
        lower = line.lower()
        return (
            len(line) < 10
            or "@" in line
            or bool(YEAR_PATTERN.search(line))
            or any(word in lower for word in JUNK_WORDS)
        )

    # ---------- year ----------

    def _find_year(self, lines: list[str]) -> Optional[int]:
        text = " ".join(lines)
        years = [int(y) for y in YEAR_PATTERN.findall(text)]
        return years[0] if years else None

    # ---------- abstract ----------

    def _find_abstract(self, lines: list[str]) -> str:
        start = None
        for i, line in enumerate(lines):
            if self._is_abstract_line(line):
                start = i
                break
        if start is None:
            return ""

        collected = []
        for line in lines[start + 1:]:
            if line.lower().startswith(ABSTRACT_STOPS) or NUMBERED_START.match(line):
                break
            collected.append(line)

        text = ""
        for line in collected:
            if text.endswith("-"):          # word split across two lines
                text = text[:-1] + line
            else:
                text = (text + " " + line).strip()

        return text[:MAX_ABSTRACT_CHARS]

    # ---------- title ----------

    def _title_lines(self, lines: list[str]) -> list[str]:
        candidates = []
        for line in lines[:8]:
            if self._is_abstract_line(line):
                break
            if not self._is_junk(line):
                candidates.append(line)
            if len(candidates) == 2:
                break
        return candidates

    # ---------- authors ----------

    def _find_authors(self, lines: list[str], title_lines: list[str]) -> list[str]:
        end = len(lines[:15])
        for i, line in enumerate(lines[:15]):
            if self._is_abstract_line(line):
                end = i
                break

        author_lines = [
            line for line in lines[:end]
            if line not in title_lines and not self._is_junk(line)
        ]

        authors = []
        for line in author_lines:
            cleaned = re.sub(r"[^A-Za-z.\-' ,]", "", line)
            for name in re.split(r",|\band\b", cleaned):
                name = name.strip()
                if len(name.split()) >= 2:
                    authors.append(name)
        return authors


if __name__ == "__main__":
    from pathlib import Path
    from .pdf_parser import PDFParser

    parser = PDFParser()
    extractor = MetadataExtractor()

    for pdf in sorted(Path("data/raw_pdfs").glob("*.pdf")):
        doc = extractor.extract(parser.parse(pdf))
        print("=" * 60)
        print(pdf.name)
        print("title   :", doc.title)
        print("authors :", doc.authors)
        print("year    :", doc.year)
        print("abstract:", doc.abstract[:100])