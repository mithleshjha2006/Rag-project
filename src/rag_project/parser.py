from langchain_community.document_loaders import PyMuPDFLoader
from pathlib import Path

pdf_path = sorted(Path("data/raw_pdfs").glob("*pdf"))[0]

loader = PyMuPDFLoader(str(pdf_path))
pages = loader.load()

print(len(pages))

print(type(pages[0]))

print(pages[0].metadata,
    pages[1].metadata)

print(pages[0].page_content[:500])

print("---- headings ----")
for p in pages:
    for line in p.page_content.split("\n"):
        if line.strip().lower() in ("abstract", "introduction", "references"):
            print("page", p.metadata["page"], "->", repr(line))