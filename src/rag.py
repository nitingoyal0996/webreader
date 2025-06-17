from typing import Any, Dict, List

import faiss
import numpy as np
import tiktoken
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .embeddings.openai_embeddings import OpenAIEmbeddingService
from .parser import WebContentParser

load_dotenv()


class WebPageRAG:
    """Unified RAG pipeline for single web page processing with citations."""

    def __init__(self, dimension: int = 3072):
        self.parser = WebContentParser()
        self.embedding_service = OpenAIEmbeddingService()
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(self.dimension)
        self.documents: List[Dict[str, Any]] = []

        # Chunking setup
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.chunker = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,  # Reduced from 200 to 50 tokens
            separators=["\n## ", "\n### ", "\n", "\n\n", ". "],
            length_function=self._count_tokens,
            is_separator_regex=False,
        )


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


    async def search_with_citations(
        self, query: str, k: int = 5, score_threshold: float = 0.6
    ) -> List[Dict[str, Any]]:
        """Search and return results with formatted citations."""
        if self.index.ntotal == 0:
            return []
        query_emb = await self.embedding_service.embed_texts([query])
        q = np.array([query_emb[0]], dtype=np.float32)
        faiss.normalize_L2(q)
        distances, indices = self.index.search(q, min(k, self.index.ntotal))    # type: ignore
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


    def _create_chunks(
        self, content: str, metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Create chunks with metadata and original position tracking."""
        text_chunks = self.chunker.split_text(content)
        # Add debug output to see token vs character counts
        chunks_with_positions = []
        current_position = 0
        for i, chunk in enumerate(text_chunks):
            if chunk.strip():
                # Find the exact position of this chunk in the original content
                chunk_start = content.find(chunk.strip(), current_position)
                if chunk_start == -1:
                    # Fallback: try to find it from the beginning
                    chunk_start = content.find(chunk.strip())
                chunk_end = chunk_start + len(chunk.strip()) if chunk_start != -1 else current_position
                
                chunks_with_positions.append({
                    "chunk_id": i,
                    "text": chunk.strip(),
                    "title": metadata.get("title", "Untitled"),
                    "author": metadata.get("author"),
                    "publication_date": metadata.get("publication_date"),
                    "token_count": self._count_tokens(chunk),
                    "position": {
                        "start": chunk_start,
                        "end": chunk_end,
                    }
                })
                current_position = chunk_end
        return chunks_with_positions


    def _add_to_index(self, documents: List[Dict[str, Any]]) -> None:
        if not documents:
            return
        embeddings = [np.array(doc["embedding"], dtype=np.float32) for doc in documents]
        arr = np.vstack(embeddings)
        faiss.normalize_L2(arr)
        self.index.add(arr)     # type: ignore
        for doc in documents:
            doc_copy = doc.copy()
            del doc_copy["embedding"]
            doc_copy["doc_id"] = len(self.documents)
            self.documents.append(doc_copy)


    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

