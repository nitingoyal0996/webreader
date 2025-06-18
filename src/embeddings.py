import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

class OpenAIEmbeddings:
    def __init__(self):
        self.model = "text-embedding-3-large"
        self.model_dimension = 3072
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []
        response = await self.client.embeddings.create(
            model=self.model, input=texts, encoding_format="float"
        )
        return [data.embedding for data in response.data]

    async def embed_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add embeddings to documents with 'text' field."""
        texts = [doc.get("text", "") for doc in documents]
        embeddings = await self.embed_texts(texts)
        return [{**doc, "embedding": emb} for doc, emb in zip(documents, embeddings)]
