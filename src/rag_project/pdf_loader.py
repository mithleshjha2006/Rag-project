import logging
from pathlib import Path

from .config import RAW_PDF_DIR

logger = logging.getLogger(__name__)


class PDFLoader:
    def __init__(self, pdf_dir: str | Path = RAW_PDF_DIR):
        self.pdf_dir = Path(pdf_dir)

    def find_pdfs(self) -> list[Path]:
        if not self.pdf_dir.exists():
            raise FileNotFoundError(f"PDF folder not found: {self.pdf_dir}")
        pdfs = [
            path for path in self.pdf_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".pdf"
        ]
        return sorted(pdfs)

    def is_valid(self, path: Path) -> bool:
        path = Path(path)

        if not path.is_file():
            return False

        if path.stat().st_size == 0:
            return False

        try:
            with open(path, "rb") as f:
                header = f.read(5)
        except OSError:
            return False

        return header == b"%PDF-"

    def load(self) -> list[Path]:
        all_pdfs = self.find_pdfs()
        valid_pdfs = []

        for path in all_pdfs:
            if self.is_valid(path):
                valid_pdfs.append(path)
            else:
                logger.warning("Skipping invalid PDF: %s", path.name)

        logger.info("Found %d valid PDFs out of %d files", len(valid_pdfs), len(all_pdfs))

        if not valid_pdfs:
            raise ValueError(f"No valid PDFs found in {self.pdf_dir}")

        return valid_pdfs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    loader = PDFLoader()
    pdfs = loader.load()
    print(len(pdfs), "valid PDFs")
    for p in pdfs:
        print(" -", p.name)