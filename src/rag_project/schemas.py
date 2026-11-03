from pydantic import BaseModel,Field
from typing import Optional,Any

class PageContent(BaseModel):
    page_number:int
    text:str

class Section(BaseModel):
    title:str
    start_page:int

class Document(BaseModel):
    doc_id: str
    file_path: str
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    year: Optional[int] = None
    abstract: str = ""
    pages: list[PageContent] = Field(min_length=1)
    sections: list[Section] = Field(default_factory=list)
    @property
    def full_text(self)->str:
        return "/n/n".join(page.text for page in self.pages)

class Chunk(BaseModel):
    chunk_id:str
    doc_id:str
    paper_title:str
    text:str
    page:int
    section:str=""

class SearchResult(BaseModel):
    chunk:Chunk
    score:float
    source:str

class Citation(BaseModel):
    number:int
    page_tittle:str
    page:int
    passage:str

class RAGResponse(BaseModel):
    question:str
    answer:str
    citations:list[Citation]=Field(default_factory=list)