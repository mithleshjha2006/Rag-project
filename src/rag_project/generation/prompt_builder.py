from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """You are a research assistant answering questions about academic papers.

Rules:
- Answer using ONLY the numbered context passages provided.
- After every claim, cite the passage number in square brackets, like [1] or [2].
- If the context does not contain the answer, reply exactly: I could not find this in the provided papers.
- Do not use outside knowledge. Be concise and precise."""

HUMAN_PROMPT = """Context:
{context}

Question: {question}"""


class PromptBuilder:
    def __init__(self):
        self.template = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ])

    def build(self, context: str, question: str):
        return self.template.invoke({"context": context, "question": question})