from ..schemas import Chunk, SearchResult

MAX_CONTEXT_CHARS = 6000


class ContextBuilder:
    def build(
        self, results: list[SearchResult], max_chars: int = MAX_CONTEXT_CHARS
    ) -> tuple[str, list[Chunk]]:
        blocks = []
        used: list[Chunk] = []
        total = 0

        for result in results:
            chunk = result.chunk
            number = len(used) + 1
            block = f"[{number}] {chunk.paper_title}, p. {chunk.page}\n{chunk.text}"

            if used and total + len(block) > max_chars:
                break

            blocks.append(block)
            used.append(chunk)
            total += len(block)

        return "\n\n".join(blocks), used