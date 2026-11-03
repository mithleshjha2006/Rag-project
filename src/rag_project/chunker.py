from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import CHUNK_OVERLAP, CHUNK_SIZE
from .schemas import Chunk, Document


class DocumentChunker:
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk_document(self, doc: Document) -> list[Chunk]:
        chunks = []
        counter = 0
        title = doc.title or doc.doc_id

        for page in doc.pages:
            pieces = self.text_splitter.split_text(page.text)
            section = self._section_for_page(doc, page.page_number)

            for piece in pieces:
                if len(piece.strip()) < 50:
                    continue

                chunks.append(Chunk(
                    chunk_id=f"{doc.doc_id}_{counter:04d}",
                    doc_id=doc.doc_id,
                    paper_title=title,
                    text=piece,
                    page=page.page_number,
                    section=section,
                ))
                counter += 1

        return chunks

    def _section_for_page(self, doc: Document, page_number: int) -> str:
        current = ""
        for section in doc.sections:
            if section.start_page <= page_number:
                current = section.title
        return current

    def chunk_all(self, docs: list[Document]) -> list[Chunk]:
        all_chunks = []
        for doc in docs:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks


if __name__ == "__main__":
    import logging
    from collections import Counter

    from .document_processor import DocumentProcessor

    logging.basicConfig(level=logging.INFO)

    docs = DocumentProcessor().process_all()
    chunks = DocumentChunker().chunk_all(docs)

    print("Total chunks:", len(chunks))
    print(Counter(c.doc_id for c in chunks))

    lengths = [len(c.text) for c in chunks]
    print("avg:", sum(lengths) // len(lengths), "min:", min(lengths), "max:", max(lengths))

    for c in chunks[:3]:
        print(c.chunk_id, "| page", c.page, "|", c.section, "|", c.text[:200])

    print("IDs unique:", len({c.chunk_id for c in chunks}) == len(chunks))

    # overlap check: two neighbouring chunks from the same page
    for a, b in zip(chunks, chunks[1:]):
        if a.doc_id == b.doc_id and a.page == b.page:
            print("END of first  :", repr(a.text[-100:]))
            print("START of second:", repr(b.text[:100]))
            break