import re

STOPWORDS = {
    "the", "of", "and", "is", "in", "to", "a", "for", "that", "with",
    "an", "as", "are", "was", "were", "be", "been", "by", "on", "at",
    "or", "it", "its", "this", "these", "those", "from", "we", "our",
    "can", "which", "has", "have", "not", "but", "also", "such",
}

WORD_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    words = WORD_PATTERN.findall(text.lower())
    return [w for w in words if len(w) > 1 and w not in STOPWORDS]