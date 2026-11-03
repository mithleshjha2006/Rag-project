import logging
from pathlib import Path

from .config import PROCESSED_DIR
from .document_cleaner import DocumentCleaner
from .metadata_extractor import MetadataExtractor
from .pdf_loader import PDFLoader
from .pdf_parser import PDFParser
from .schemas import Document

logger = logging.getLogger(__name__)


class DocumentProcessor:
    def __init__(self, processed_dir: str | Path = PROCESSED_DIR):
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self.loader = PDFLoader()
        self.parser = PDFParser()
        self.extractor = MetadataExtractor()
        self.cleaner = DocumentCleaner()

    def process_one(self, path: Path) -> Document:
        doc = self.parser.parse(path)
        doc = self.extractor.extract(doc)  
        doc = self.cleaner.clean(doc)
        return doc

    def _json_path(self, doc_id: str) -> Path:
        return self.processed_dir / f"{doc_id}.json"

    def save(self, doc: Document) -> Path:
        out_path = self._json_path(doc.doc_id)
        out_path.write_text(doc.model_dump_json(indent=2), encoding="utf-8")
        return out_path

    def load(self, doc_id: str) -> Document:
        text = self._json_path(doc_id).read_text(encoding="utf-8")
        return Document.model_validate_json(text)

    def process_all(self, force: bool = False) -> list[Document]:
        paths = self.loader.load()
        documents = []
        failed = []

        for path in paths:
            doc_id = path.stem

            if not force and self._json_path(doc_id).exists():
                documents.append(self.load(doc_id))
                logger.info("Loaded saved: %s", path.name)
                continue

            try:
                doc = self.process_one(path)
                self.save(doc)
                documents.append(doc)
                logger.info("Processed: %s", path.name)
            except Exception:
                logger.exception("Failed to process: %s", path.name)
                failed.append(path.name)

        logger.info("Done: %d of %d succeeded", len(documents), len(paths))
        if failed:
            logger.warning("Failed papers: %s", failed)

        return documents


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    processor = DocumentProcessor()
    docs = processor.process_all()

    for d in docs:
        print(d.doc_id, "|", d.title[:40], "|", d.year, "|",
              len(d.pages), "pages |", len(d.sections), "sections")

    # check that load() round-trips
    first = docs[0]
    loaded = processor.load(first.doc_id)
    print("Round trip OK:", loaded.title == first.title and len(loaded.pages) == len(first.pages))