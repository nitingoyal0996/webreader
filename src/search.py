from typing import Any, Dict, List

import faiss
import nltk
import numpy as np
from dotenv import load_dotenv
from instructor import patch
from nltk.tokenize import sent_tokenize
from openai import AsyncOpenAI
from pydantic import BaseModel

from .embeddings import OpenAIEmbeddings
from .parser import WebContentParser

load_dotenv()


class QueryClassification(BaseModel):
    classification: str

    def is_generic(self) -> bool:
        return self.classification.strip().upper() == "GENERIC"


class KeywordExtraction(BaseModel):
    keywords: List[str]


class WebPageSearch:
    """Unified RAG pipeline for single web page processing with citations."""

    def __init__(self, dimension: int = 3072):
        self.title = ""
        self._content_keywords = ""
        self._original_content = ""

        self.parser = WebContentParser()
        self.embedding_service = OpenAIEmbeddings()
        self.openai_client = patch(AsyncOpenAI())

        self.dimension = dimension
        self.index = faiss.IndexFlatIP(self.dimension)
        self.documents: List[Dict[str, Any]] = []
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")

    async def process_url(self, url: str) -> Dict[str, Any]:
        """Fetch, parse, chunk, embed, and index a web page."""
        parsed_content = await self.parser.fetch_and_parse(url)
        self._original_content = parsed_content["content"]
        self.title = parsed_content["metadata"]["title"]
        await self._extract_content_keywords()
        self.documents.clear()
        self.index = faiss.IndexFlatIP(self.dimension)
        chunks = self._create_chunks(
            content=parsed_content["content"],
            metadata=parsed_content["metadata"],
        )
        embedded_chunks = await self.embedding_service.embed_documents(chunks)
        self._add_to_index(embedded_chunks)
        return {
            "title": parsed_content["metadata"]["title"],
            "chunks_created": len(chunks),
            "ready_for_search": True,
        }

    async def search(
        self, query: str, k: int = 10, score_threshold: float = 0.4
    ) -> List[Dict[str, Any]]:
        """Search with query classification and expansion."""
        if self.index.ntotal == 0:
            return []
        is_generic = await self.is_generic_query(query)
        if is_generic:
            query = await self._expand_query(query)
        results = await self._semantic_search(query, k, score_threshold)
        return results[:5]

    # =============================================================================
    # PROCESS QUERY
    # =============================================================================
    async def is_generic_query(self, query: str) -> bool:
        """Classify if the query is generic or specific using Pydantic and instructor."""
        classification_prompt = self._get_query_classification_prompt(self.title, query)

        try:
            response = await self.openai_client.chat.completions.create(  # pyright: ignore[reportCallIssue, arg-type]
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": classification_prompt}],
                max_tokens=10,
                temperature=0,
                response_model=QueryClassification,
            )
            return response.is_generic()
        except Exception as e:
            print(f"Query classification failed: {e}")
            return False

    async def _expand_query(self, query: str) -> str:
        """Expand generic queries using extracted content keywords."""
        if not self._content_keywords:
            await self._extract_content_keywords()
        keywords_str = ", ".join(self._content_keywords[:10])
        expansion_prompt = self._get_query_expansion_prompt(
            self.title, keywords_str, query
        )

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": expansion_prompt}],
                max_tokens=50,
                temperature=0,
            )
            content = response.choices[0].message.content
            expanded_query = content.strip() if content is not None else query
            return expanded_query
        except Exception as e:
            print(f"Query expansion failed: {e}")
            return query

    async def _extract_content_keywords(self):
        """Extract key concepts and keywords from the content."""
        content_preview = (
            self._original_content[:2000] if hasattr(self, "_original_content") else ""
        )
        keyword_prompt = self._get_keyword_extraction_prompt(
            self.title, content_preview
        )

        try:
            response = await self.openai_client.chat.completions.create(  # pyright: ignore[reportCallIssue, arg-type]
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": keyword_prompt}],
                max_tokens=200,
                temperature=0,
                response_model=KeywordExtraction,
            )
            self._content_keywords = response.keywords
            print(f"Extracted keywords: {self._content_keywords}")
        except Exception as e:
            print(f"Keyword extraction failed: {e}")
            self._content_keywords = []

    # =============================================================================
    # PROCESS CONTENT
    # =============================================================================
    def _create_chunks(
        self, content: str, metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Create sentence-based chunks with metadata and position tracking."""
        sentences = sent_tokenize(content)
        chunks = []
        SENTENCES_PER_CHUNK = 3
        OVERLAP = 1

        for i in range(0, len(sentences), SENTENCES_PER_CHUNK - OVERLAP):
            chunk_sentences = sentences[i : i + SENTENCES_PER_CHUNK]
            if not chunk_sentences:
                continue

            chunk_text = " ".join(chunk_sentences)
            if not chunk_text.strip():
                continue

            sentence_map = {}
            for j, sentence in enumerate(chunk_sentences):
                sentence_map[j + 1] = sentence

            chunk_data = {
                "text": chunk_text,
                "chunk_id": len(chunks),
                "sentence_count": len(chunk_sentences),
                "sentence_map": sentence_map,
                "metadata": {
                    **metadata,
                    "chunk_index": len(chunks),
                    "total_sentences": len(chunk_sentences),
                },
            }
            chunks.append(chunk_data)
        return chunks

    def _add_to_index(self, documents: List[Dict[str, Any]]) -> None:
        """Add embedded documents to the FAISS index."""
        if not documents:
            return

        embeddings = [np.array(doc["embedding"], dtype=np.float32) for doc in documents]
        arr = np.vstack(embeddings)
        faiss.normalize_L2(arr)
        self.index.add(arr)  # type: ignore

        for doc in documents:
            doc_copy = doc.copy()
            del doc_copy["embedding"]
            doc_copy["doc_id"] = len(self.documents)
            self.documents.append(doc_copy)

    async def _semantic_search(
        self, query: str, k: int, score_threshold: float
    ) -> List[Dict[str, Any]]:
        """Perform semantic search with given parameters."""
        query_emb = await self.embedding_service.embed_texts([query])
        q = np.array([query_emb[0]], dtype=np.float32)
        faiss.normalize_L2(q)
        distances, indices = self.index.search(q, min(k, self.index.ntotal))  # type: ignore

        results = []
        for i, (score, idx) in enumerate(zip(distances[0], indices[0])):
            if 0 <= idx < len(self.documents) and score >= score_threshold:
                doc = self.documents[idx].copy()
                doc.update(
                    {
                        "cosine_similarity": float(score),
                        "rank": i + 1,
                    }
                )
                results.append(doc)
        return results

    # =============================================================================
    # PROMPTS
    # =============================================================================
    @staticmethod
    def _get_query_classification_prompt(title: str, query: str) -> str:
        """Get the prompt for classifying if a query is generic or specific."""
        return f"""You are an expert at classifying user queries about web articles.
Instructions:
1. If the query asks for a summary, main topic, or general overview, classify as GENERIC.
2. If the query asks about specific details, methods, concepts, or facts, classify as SPECIFIC.

Article Title: {title or 'Unknown'}
User Query: "{query}"
"""

    @staticmethod
    def _get_keyword_extraction_prompt(title: str, content_preview: str) -> str:
        """Get the prompt for extracting keywords from article content."""
        return f"""Extract the 15 most important keywords and concepts from this article.
Article Title: {title or 'Unknown'}
Content: {content_preview}...

Return the keywords as a structured list."""

    @staticmethod
    def _get_query_expansion_prompt(title: str, keywords_str: str, query: str) -> str:
        """Get the prompt for expanding generic queries into specific search terms."""
        return f"""Transform this generic query into specific search terms using the article's key concepts.
Article Title: {title or 'Unknown'}
Key Concepts: {keywords_str}
Generic Query: "{query}"
Return a specific search query (max 15 words):"""
