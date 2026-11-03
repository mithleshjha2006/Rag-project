import logging
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from ..config import BASE_DIR, LLM_MODEL_NAME

load_dotenv(BASE_DIR / ".env")

logger = logging.getLogger(__name__)


class AnswerGenerator:
    def __init__(self, model_name: str = LLM_MODEL_NAME):
        if not os.getenv("GROQ_API_KEY"):
            raise ValueError("GROQ_API_KEY not found. Add it to the .env file.")
        self.llm = ChatGroq(model=model_name, temperature=0)

    def generate(self, prompt) -> str:
        try:
            response = self.llm.invoke(prompt)
        except Exception:
            logger.exception("LLM call failed")
            raise
        return response.content