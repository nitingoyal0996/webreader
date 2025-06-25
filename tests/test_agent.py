import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from ichatbio.types import ArtifactMessage, ProcessMessage, TextMessage
from pydantic import HttpUrl
from src.agent import WebReaderAgent
from src.citations import CitationProcessor
from src.models.request import WebReaderRequest


@pytest.fixture
def mock_agent():
    with patch.object(WebReaderAgent, "__init__", lambda self: None):
        agent = WebReaderAgent()
        from openai import AsyncOpenAI
        from src.search import WebPageSearch

        agent.rag = WebPageSearch()
        agent.citation_processor = CitationProcessor()
        agent.openai_client = AsyncOpenAI()
        return agent


@pytest.fixture
def sample_search_results():
    return [
        {
            "chunk_id": 1,
            "text": "Bumblebees are found in the Northern Hemisphere and are important pollinators.",
            "cosine_similarity": 0.9,
            "sentence_map": {
                1: "Bumblebees are found in the Northern Hemisphere and are important pollinators."
            },
        },
        {
            "chunk_id": 2,
            "text": "Climate change threatens biodiversity worldwide.",
            "cosine_similarity": 0.8,
            "sentence_map": {1: "Climate change threatens biodiversity worldwide."},
        },
    ]


@pytest.fixture
def test_request():
    return {
        "query": "Where are bumblebees found?",
        "params": WebReaderRequest(url=HttpUrl("https://example.com")),
    }


@pytest.mark.asyncio
async def test_citation_processing_functionality():
    """Test that citation processing works correctly - unit test for CitationProcessor."""
    processor = CitationProcessor()
    ai_response = '<CIT chunk_id="1" sentences="1">Bumblebees are found in the Northern Hemisphere.</CIT>'
    search_hits = [
        {
            "chunk_id": 1,
            "text": "Bumblebees are found in the Northern Hemisphere and are important pollinators.",
        }
    ]
    page_url = "https://example.com"
    clean_text, markdown_text, citations = processor.process_citations(
        ai_response, search_hits, page_url
    )
    assert clean_text == "Bumblebees are found in the Northern Hemisphere."
    assert len(citations) == 1
    assert citations[0]["chunk_id"] == 1
    assert "#:~:text=" in citations[0]["link"]
    assert "[Bumblebees are found in the Northern Hemisphere.](" in markdown_text
    assert "https://example.com#:~:text=" in markdown_text


@pytest.mark.asyncio
async def test_successful_agent_run_with_llm(
    mock_agent, sample_search_results, test_request
):
    """Test full agent flow with real LLM call - integration test."""
    mock_agent.rag.process_url = AsyncMock(
        return_value={"title": "Test Page about Bees", "chunks_created": 5}
    )
    mock_agent.rag.search = AsyncMock(return_value=sample_search_results)
    messages = [
        message
        async for message in mock_agent.read_and_analyze(
            test_request["query"], test_request["params"]
        )
    ]
    assert len(messages) >= 4, "Should yield at least 4 messages"
    process_messages = [m for m in messages if isinstance(m, ProcessMessage)]
    text_messages = [m for m in messages if isinstance(m, TextMessage)]
    artifact_messages = [m for m in messages if isinstance(m, ArtifactMessage)]
    assert len(process_messages) >= 3, "Should have multiple process messages"
    assert len(text_messages) >= 1, "Should have at least one text message with answer"
    assert len(artifact_messages) == 1, "Should have one artifact message"
    text_message = text_messages[0]
    assert len(text_message.text) > 20, "LLM should generate substantial content"
    artifact_content = json.loads(artifact_messages[0].content.decode())
    assert "answer" in artifact_content
    assert "citations" in artifact_content
    assert "markdown_answer" in artifact_content
    mock_agent.rag.process_url.assert_awaited_once()
    mock_agent.rag.search.assert_awaited_once()


@pytest.mark.asyncio
async def test_no_search_results_scenario(mock_agent, test_request):
    """Test agent behavior when no search results are found."""
    mock_agent.rag.process_url = AsyncMock(
        return_value={"title": "Empty Test Page", "chunks_created": 0}
    )
    mock_agent.rag.search = AsyncMock(return_value=[])  # No results
    messages = [
        message
        async for message in mock_agent.read_and_analyze(
            test_request["query"], test_request["params"]
        )
    ]
    text_messages = [m for m in messages if isinstance(m, TextMessage)]
    assert len(text_messages) >= 1, "Should have at least one text message"
    text_content = text_messages[0].text
    assert "couldn't find any relevant content" in text_content
    assert test_request["query"] in text_content
    artifact_messages = [m for m in messages if isinstance(m, ArtifactMessage)]
    if artifact_messages:
        artifact_content = json.loads(artifact_messages[0].content.decode())
        assert len(artifact_content.get("citations", [])) == 0


if __name__ == "__main__":
    pytest.main([__file__])
