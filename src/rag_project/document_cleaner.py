import re

from .schemas import Document, PageContent

LIGATURES = {
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬀ": "ff",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
}
HYPHEN_BREAK = re.compile(r"-\n([a-z])")
SINGLE_NEWLINE = re.compile(r"(?<!\n)\n(?!\n)")
REFERENCES_LINE = re.compile(r"^(\d+\.?\s+)?references$", re.IGNORECASE)


class DocumentCleaner:
    def _fix_ligatures(self, text: str) -> str:
        for old, new in LIGATURES.items():
            text = text.replace(old, new)
        return text

    def _fix_hyphens(self, text: str) -> str:
        return HYPHEN_BREAK.sub(r"\1", text)

    def _join_lines(self, text: str) -> str:
        return SINGLE_NEWLINE.sub(" ", text)

    def _normalize_spaces(self, text: str) -> str:
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _remove_page_numbers(self, text: str) -> str:
        lines = text.split("\n")
        kept = [line for line in lines if not line.strip().isdigit()]
        return "\n".join(kept)

    def _clean_text(self, text: str) -> str:
        text = self._fix_ligatures(text)
        text = self._remove_page_numbers(text)
        text = self._fix_hyphens(text)
        text = self._join_lines(text)
        text = self._normalize_spaces(text)
        return text

    def _cut_references(self, doc: Document) -> Document:
        kept = []
        for page in doc.pages:
            lines = page.text.split("\n")
            cut = None
            for i, line in enumerate(lines):
                if REFERENCES_LINE.match(line.strip(" .:")):
                    cut = i
                    break

            if cut is None:
                kept.append(page)
            else:
                above = "\n".join(lines[:cut]).strip()
                if above:
                    kept.append(PageContent(page_number=page.page_number, text=above))
                break

        if not kept:
            return doc
        return doc.model_copy(update={"pages": kept})

    def clean(self, doc: Document) -> Document:
        doc = self._cut_references(doc)

        new_pages = [
            PageContent(page_number=p.page_number, text=self._clean_text(p.text))
            for p in doc.pages
        ]
        new_pages = [p for p in new_pages if p.text]

        if not new_pages:
            return doc
        return doc.model_copy(update={"pages": new_pages})


if __name__ == "__main__":
    from pathlib import Path
    from .pdf_parser import PDFParser

    cleaner = DocumentCleaner()

    fake = "Text Classiﬁcation is use-\nful.\n42\nIt works  well\nin practice.\n\nNext paragraph."
    print(repr(cleaner._clean_text(fake)))

    parser = PDFParser()
    for pdf in sorted(Path("data/raw_pdfs").glob("*.pdf")):
        doc = parser.parse(pdf)
        cleaned = cleaner.clean(doc)
        print(pdf.name, "|", len(doc.pages), "->", len(cleaned.pages), "pages")