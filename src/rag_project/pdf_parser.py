import logging
import re
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader

from .schemas import Document, PageContent, Section

logger = logging.getLogger(__name__)

HEADING_NAMES = {
    "abstract", "introduction", "related work", "background",
    "methodology", "methods", "method", "experiments", "results",
    "discussion", "conclusion", "conclusions", "references",
    "acknowledgments", "acknowledgements",
}

NUMBERED_HEADING = re.compile(r"^\d+(\.\d+)*\.?\s+[A-Z].*$")

MAX_HEADING_LENGTH = 80


class PDFParser:
    def parse(self, path: Path) -> Document:
        path = Path(path)

        try:
            lc_pages = PyMuPDFLoader(str(path)).load()
        except Exception:
            logger.exception("Failed to load PDF: %s", path.name)
            raise

        pages = []
        for lc_page in lc_pages:
            text = lc_page.page_content
            if text.strip() == "":
                continue
            pages.append(
                PageContent(page_number=lc_page.metadata["page"] + 1, text=text)
            )

        if not pages:
            raise ValueError(f"No text extracted from {path.name}")

        sections = self.find_sections(pages)

        return Document(
            doc_id=path.stem,
            file_path=str(path),
            pages=pages,
            sections=sections,
        )

    def find_sections(self, pages: list[PageContent]) -> list[Section]:
        sections = []

        for page in pages:
            for line in page.text.split("\n"):
                line = line.strip()

                if not line or len(line) > MAX_HEADING_LENGTH:
                    continue

                if line.lower() in HEADING_NAMES or NUMBERED_HEADING.match(line):
                    sections.append(Section(title=line, start_page=page.page_number))

        return sections


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = PDFParser()
    doc = parser.parse(Path("data/raw_pdfs/10_RNTN.pdf"))
    print(len(doc.pages), "pages,", len(doc.sections), "sections")
    for s in doc.sections:
        print(s.start_page, "->", s.title)