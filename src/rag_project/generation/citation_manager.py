import logging
import re

from ..schemas import Chunk, Citation, RAGResponse

logger = logging.getLogger(__name__)

CITATION_PATTERN = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


class CitationManager:
    def extract(
        self, answer: str, context_chunks: list[Chunk]
    ) -> tuple[list[Citation], list[int]]:
        numbers = set()
        for group in CITATION_PATTERN.findall(answer):
            numbers.update(int(n) for n in re.findall(r"\d+", group))

        citations = []
        invalid = []
        for n in sorted(numbers):
            if 1 <= n <= len(context_chunks):
                chunk = context_chunks[n - 1]
                citations.append(Citation(
                    number=n,
                    paper_title=chunk.paper_title,
                    page=chunk.page,
                    passage=chunk.text,
                ))
            else:
                invalid.append(n)

        return citations, invalid

    def build_response(
        self, question: str, answer: str, context_chunks: list[Chunk]
    ) -> RAGResponse:
        citations, invalid = self.extract(answer, context_chunks)
        if invalid:
            logger.warning("Answer cites numbers that do not exist: %s", invalid)
        return RAGResponse(question=question, answer=answer, citations=citations)