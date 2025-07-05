import json
import os
from typing import Optional

from ichatbio.agent import IChatBioAgent
from ichatbio.agent_response import IChatBioAgentProcess, ResponseContext
from ichatbio.types import AgentCard, AgentEntrypoint
from openai import AsyncOpenAI
from pydantic import BaseModel
from typing_extensions import override

from .citations import CitationProcessor
from .models.request import WebReaderRequest
from .search import WebPageSearch

# Number of search results to retrieve
K = 10
# Minimum cosine similarity threshold
COSINE_THRESHOLD = 0.3


web_reader_agent_card = AgentCard(
    name="Web Reader Agent",
    description="""Agent that extracts, reads and analyzes web content and returns related content 
    right from the source based on a user query.
    Please provide a search query and target URL.
    """,
    icon=None,
    entrypoints=[
        AgentEntrypoint(
            id="read_web",
            description="Parse and analyze web content to search related content. Provide search query and target URL.",
            parameters=WebReaderRequest,
        ),
    ],
)


class WebReaderAgent(IChatBioAgent):
    def __init__(self):
        self.agent_card = web_reader_agent_card
        self.rag = WebPageSearch()
        self.openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.citation_processor = CitationProcessor()

    @override
    def get_agent_card(self) -> AgentCard:
        return self.agent_card

    @override
    async def run(
        self,
        context: ResponseContext,
        request: str,
        entrypoint: str,
        params: Optional[BaseModel],
    ):
        if entrypoint != "read_web":
            await context.reply(f"Unknown entrypoint: {entrypoint}")
            return

        if not isinstance(params, WebReaderRequest):
            await context.reply(
                "Invalid parameters for read_web. Expected WebReaderRequest."
            )
            return

        await self.read_and_analyze(context, request, params)

    async def read_and_analyze(
        self, context: ResponseContext, query: str, params: WebReaderRequest
    ):
        async with context.begin_process(summary="Web content analysis") as process:
            process: IChatBioAgentProcess

            await process.log(
                f"Starting web content analysis for URL '{params.url}' with query '{query}'",
                data={"url": str(params.url), "query": query},
            )

            try:
                result = await self.rag.process_url(str(params.url))

                await process.log(
                    "Content processed and indexed, performing semantic search",
                    data={
                        "title": result["title"],
                        "chunks_created": result["chunks_created"],
                        "query": query,
                    },
                )

                search_results = await self.rag.search(
                    query=query, k=K, score_threshold=COSINE_THRESHOLD
                )

                await process.log(
                    "Search completed, compiling response",
                    data={
                        "relevant_sections_found": len(search_results),
                        "cosine_threshold": COSINE_THRESHOLD,
                    },
                )

                raw_answer = await self._generate_ai_response(
                    query, search_results, result["title"], str(params.url)
                )

                clean_answer, markdown_answer, citations = (
                    self.citation_processor.process_citations(
                        raw_answer, search_results, str(params.url)
                    )
                )

                await context.reply(markdown_answer)

                await process.create_artifact(
                    mimetype="application/json",
                    description="Answer with inline citations and hyperlinks",
                    content=json.dumps(
                        {
                            "answer": clean_answer,
                            "citations": citations,
                            "markdown_answer": markdown_answer,
                        },
                        indent=2,
                    ).encode("utf-8"),
                    uris=[str(params.url)],
                    metadata={
                        "total_matches": len(search_results),
                        "cosine_score_threshold": COSINE_THRESHOLD,
                        "page_title": result["title"],
                        "chunks_created": result["chunks_created"],
                    },
                )

                await process.log(
                    f"Analysis completed with {len(citations)} citations",
                    data={"citations_count": len(citations)},
                )

            except Exception as e:
                await context.reply(f"Error analyzing web content: {str(e)}")
                import traceback

                await process.log(
                    f"Error details: {traceback.format_exc()}",
                    data={"error": str(e), "traceback": traceback.format_exc()},
                )

    async def _generate_ai_response(
        self, query: str, search_results: list, page_title: str, page_url: str
    ) -> str:
        """Generate a LLM response based on search results."""
        if not search_results:
            return f"I couldn't find any relevant content about '{query}' in the webpage '{page_title}'. The content may not cover this topic or you might want to try a different search query."

        # Prepare context from similarity search results
        context_parts = []
        for hit in search_results[:5]:
            sentence_info = ""
            if "sentence_map" in hit:
                sentence_count = len(hit["sentence_map"])
                sentence_info = f" (contains {sentence_count} sentences)"
            context_parts.append(f"[{hit['chunk_id']}]{sentence_info} {hit['text']}")

        system_prompt = self.citation_processor.get_citation_prompt()

        user_prompt = (
            f"User Question: {query}\n"
            f"Web Page Title: {page_title}\n"
            f"Relevant Content:\n"
            f"{"\n".join(context_parts)}\n"
            "Answer:"
        )

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=1500,
                temperature=0.2,  # lower temp for less creativity.
            )
            return (
                response.choices[0].message.content or "I couldn't generate a response."
            )
        except Exception as e:
            return f"Error generating AI response: {str(e)}"
