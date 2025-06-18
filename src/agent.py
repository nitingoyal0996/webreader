import json
import os
from collections.abc import AsyncGenerator
from typing import Optional

from ichatbio.agent import IChatBioAgent
from ichatbio.types import (AgentCard, AgentEntrypoint, ArtifactMessage,
                            Message, ProcessMessage, TextMessage)
from openai import AsyncOpenAI
from pydantic import BaseModel
from typing_extensions import override

from .models.request import WebReaderRequest
from .rag import WebPageRAG


web_reader_agent_card = AgentCard(
    name="Web Reader Agent",
    description="""Agent that extract, reads and analyzes web content and returns related content 
    right from the source based on a user query.
    Please provide a search query and target URL.
    """,
    icon=None,
    entrypoints=[
        AgentEntrypoint(
            id="read_web",
            description="Parse and analyzes web content to search related content. Provide search query and target URL.",
            parameters=WebReaderRequest,
        ),
    ],
)


class WebReaderAgent(IChatBioAgent):
    def __init__(self):
        self.agent_card = web_reader_agent_card
        self.rag = WebPageRAG()
        self.openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


    @override
    def get_agent_card(self) -> AgentCard:
        return self.agent_card


    @override
    async def run(                                                                      # type: ignore
        self, request: str, entrypoint: str, params: Optional[BaseModel]
    ) -> AsyncGenerator[Message, None]: 
        if entrypoint == "read_web":
            if not isinstance(params, WebReaderRequest):
                yield TextMessage(
                    text="Invalid parameters for read_web. Expected WebReaderRequest."
                )
                return
            async for message in self.read_and_analyze(params):
                yield message
        else:
            yield TextMessage(text=f"Unknown entrypoint: {entrypoint}")


    async def read_and_analyze(
        self, params: WebReaderRequest
    ) -> AsyncGenerator[Message, None]:
        yield ProcessMessage(
            summary="Starting web content analysis",
            description=f"Processing URL '{params.url}' with query '{params.query}'",
        )
        K = 5  # Get more results for better analysis
        COSINE_THRESHOLD = 0.6    # Lower threshold to catch more relevant content

        try:
            # Use RAG to process the URL
            result = await self.rag.process_url(str(params.url))

            yield ProcessMessage(
                summary="Content processed and indexed",
                description=f"Successfully processed '{result['title']}' - Created {result['chunks_created']} chunks",
            )

            yield ProcessMessage(
                summary="Performing semantic search",
                description=f"Searching for content relevant to query '{params.query}'",
            )

            search_results = await self.rag.search(
                query=params.query, 
                k=K,
                score_threshold=COSINE_THRESHOLD
            )

            yield ProcessMessage(
                summary="Search completed",
                description=f"Found {len(search_results)} semantically relevant sections with over {COSINE_THRESHOLD} cosine similarity",
            )

            # Generate AI response based on search results
            yield ProcessMessage(
                summary="Generating AI response",
                description="Analyzing search results to provide comprehensive answer",
            )

            ai_response = await self._generate_ai_response(params.query, search_results, result["title"])

            yield TextMessage(text=ai_response)

            results = {
                "url": str(params.url),
                "query": params.query,
                "title": result["title"],
                "total_chunks": result["chunks_created"],
                "relevant_page_content": search_results,
            }

            yield ArtifactMessage(
                mimetype="application/json",
                description=f"Semantic search results for '{params.query}'",
                content=json.dumps(results, indent=2).encode("utf-8"),
                uris=[str(params.url)],
                metadata={
                    "total_matches": len(search_results),
                    "cosine_score_threshold": COSINE_THRESHOLD,
                },
            )

            yield ProcessMessage(
                summary="Web content analysis completed",
                description=f"Completed analysis with {len(search_results)} relevant matches with precise citations",
            )

        except Exception as e:
            yield TextMessage(text=f"Error analyzing web content: {str(e)}")
            import traceback
            yield TextMessage(text=f"Traceback: {traceback.format_exc()}")


    async def _generate_ai_response(self, query: str, search_results: list, page_title: str) -> str:
        """Generate a comprehensive AI response based on search results."""
        if not search_results:
            return f"I couldn't find any relevant content about '{query}' in the webpage '{page_title}'. The content may not cover this topic or you might want to try a different search query."

        # Prepare context from similarity search results
        context_parts = []
        for i, result in enumerate(search_results[:5], 1):  # Use top 5 most relevant results
            text = result["text"]
            score = result["cosine_similarity"]
            position = result["position"]
            context_parts.append(f"[Source {i}] (Score: {score:.3f}, Position: {position['start']}-{position['end']})\n{text}\n")
        context = "\n".join(context_parts)
        prompt = f"""You are a helpful assistant that answers questions based on web content. 
User Question: {query}
Web Page Title: {page_title}
Relevant Content (with similarity scores and positions):
{context}
Important Rules:
- Answer the user's question based ONLY on the provided relevant content
- Do not make assumptions or provide information not found in the content
- If the content does not answer the question, clearly state that
- Use the provided content to generate a clear, concise, and accurate response
- If the content is insufficient, acknowledge that and suggest the user try a different query
- Keep your response natural and conversational
- If there are multiple perspectives or details, include them
Answer:"""
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a helpful assistant that provides clear, accurate answers based on web content."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            return response.choices[0].message.content or "I couldn't generate a response."
            
        except Exception as e:
            return f"Error generating AI response: {str(e)}"
