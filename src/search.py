from typing import Any, Dict, List

import faiss
import nltk
import numpy as np
from dotenv import load_dotenv
from nltk.tokenize import sent_tokenize

from .embeddings import OpenAIEmbeddings
from .parser import WebContentParser

load_dotenv()


class WebPageSearch:
    """Unified RAG pipeline for single web page processing with citations."""

    def __init__(self, dimension: int = 3072):
        self.parser = WebContentParser()
        self.embedding_service = OpenAIEmbeddings()
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
        """Search and return results with formatted citations."""
        if self.index.ntotal == 0:
            return []
        query_emb = await self.embedding_service.embed_texts([query])
        q = np.array([query_emb[0]], dtype=np.float32)
        faiss.normalize_L2(q)
        distances, indices = self.index.search(q, min(k, self.index.ntotal))

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
            else:
                print(
                    f"Rejected: idx valid: {0 <= idx < len(self.documents)}, score above threshold: {score >= score_threshold}"
                )
        return results

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
        if not documents:
            return
        embeddings = [np.array(doc["embedding"], dtype=np.float32) for doc in documents]
        arr = np.vstack(embeddings)
        faiss.normalize_L2(arr)
        self.index.add(arr)

        for doc in documents:
            doc_copy = doc.copy()
            del doc_copy["embedding"]
            doc_copy["doc_id"] = len(self.documents)
            self.documents.append(doc_copy)
