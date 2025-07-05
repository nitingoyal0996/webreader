import json
from unittest.mock import AsyncMock, patch

import pytest
from ichatbio.agent_response import (ArtifactResponse, DirectResponse,
                                     ProcessLogResponse, ResponseChannel,
                                     ResponseContext, ResponseMessage)
from pydantic import HttpUrl

from src.agent import WebReaderAgent
from src.citations import CitationProcessor
from src.models.request import WebReaderRequest

TEST_CONTEXT_ID = "617727d1-4ce8-4902-884c-db786854b51c"


class InMemoryResponseChannel(ResponseChannel):
    def __init__(self, message_buffer: list):
        self.message_buffer = message_buffer

    async def submit(self, message: ResponseMessage, context_id: str):
        self.message_buffer.append(message)


@pytest.fixture(scope="function")
def messages() -> list[ResponseMessage]:
    return []


@pytest.fixture(scope="function")
def context(messages) -> ResponseContext:
    return ResponseContext(InMemoryResponseChannel(messages), TEST_CONTEXT_ID)


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


async def get_all_agent_messages_with_agent(agent, params_dict, context, messages):
    """Helper function that uses a provided agent instance."""
    params_model = WebReaderRequest(**params_dict)
    await agent.run(
        context,
        request="Where are bumblebees found?",
        entrypoint="read_web",
        params=params_model,
    )
    return messages


async def get_all_agent_messages(params_dict, context, messages):
    """Helper function that creates a new agent instance."""
    agent = WebReaderAgent()
    params_model = WebReaderRequest(**params_dict)
    await agent.run(
        context,
        request="Where are bumblebees found?",
        entrypoint="read_web",
        params=params_model,
    )
    return messages


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
async def test_successful_agent_run_with_mocked_llm(context, messages):
    """Test full agent flow with mocked components."""
    with patch.object(WebReaderAgent, "__init__", lambda self: None):
        agent = WebReaderAgent()
        from openai import AsyncOpenAI

        from src.citations import CitationProcessor
        from src.search import WebPageSearch

        agent.rag = WebPageSearch()
        agent.citation_processor = CitationProcessor()
        agent.openai_client = AsyncOpenAI()

        # Mock the RAG components
        agent.rag.process_url = AsyncMock(
            return_value={"title": "Test Page about Bees", "chunks_created": 5}
        )
        agent.rag.search = AsyncMock(
            return_value=[
                {
                    "chunk_id": 1,
                    "text": "Bumblebees are found in the Northern Hemisphere and are important pollinators.",
                    "cosine_similarity": 0.9,
                    "sentence_map": {
                        1: "Bumblebees are found in the Northern Hemisphere and are important pollinators."
                    },
                }
            ]
        )

        # Mock the OpenAI client
        mock_response = AsyncMock()
        mock_response.choices = [AsyncMock()]
        mock_response.choices[0].message.content = (
            '<CIT chunk_id="1" sentences="1">Bumblebees are found in the Northern Hemisphere and are important pollinators.</CIT>'
        )
        agent.openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        params = {"url": "https://example.com"}
        test_messages = await get_all_agent_messages_with_agent(
            agent, params, context, messages
        )

        assert test_messages, "Agent should yield messages"

        # Check for process log messages
        process_messages = [
            m for m in test_messages if isinstance(m, ProcessLogResponse)
        ]
        assert len(process_messages) >= 2, "Should have at least 2 process log messages"

        # Check for direct response (the actual answer)
        direct_messages = [m for m in test_messages if isinstance(m, DirectResponse)]
        assert (
            len(direct_messages) >= 1
        ), "Should have at least one direct response message"

        # Check for artifact message
        artifact_messages = [
            m for m in test_messages if isinstance(m, ArtifactResponse)
        ]
        assert len(artifact_messages) == 1, "Should have one artifact message"

        # Verify artifact content
        artifact = artifact_messages[0]
        assert artifact.mimetype == "application/json"
        assert "Answer with inline citations" in artifact.description
        assert artifact.metadata is not None
        assert "total_matches" in artifact.metadata
        assert "page_title" in artifact.metadata

        # Verify the mocks were called
        agent.rag.process_url.assert_awaited_once()
        agent.rag.search.assert_awaited_once()


@pytest.mark.asyncio
async def test_no_search_results_scenario(context, messages):
    """Test agent behavior when no search results are found."""
    with patch.object(WebReaderAgent, "__init__", lambda self: None):
        agent = WebReaderAgent()
        from openai import AsyncOpenAI

        from src.citations import CitationProcessor
        from src.search import WebPageSearch

        agent.rag = WebPageSearch()
        agent.citation_processor = CitationProcessor()
        agent.openai_client = AsyncOpenAI()

        # Mock empty search results
        agent.rag.process_url = AsyncMock(
            return_value={"title": "Empty Test Page", "chunks_created": 0}
        )
        agent.rag.search = AsyncMock(return_value=[])

        # Mock OpenAI response for no results
        mock_response = AsyncMock()
        mock_response.choices = [AsyncMock()]
        mock_response.choices[0].message.content = (
            "I couldn't find any relevant content about 'test query' in the webpage 'Empty Test Page'."
        )
        agent.openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        params = {"url": "https://example.com"}
        test_messages = await get_all_agent_messages_with_agent(
            agent, params, context, messages
        )

        assert test_messages, "Agent should yield messages"

        # Check for direct response messages
        direct_messages = [m for m in test_messages if isinstance(m, DirectResponse)]
        assert (
            len(direct_messages) >= 1
        ), "Should have at least one direct response message"

        # Verify the response indicates no content found
        response_text = direct_messages[0].text
        assert "couldn't find any relevant content" in response_text

        # Check artifact still exists but with empty citations
        artifact_messages = [
            m for m in test_messages if isinstance(m, ArtifactResponse)
        ]
        if artifact_messages:
            artifact = artifact_messages[0]
            assert artifact.metadata is not None
            assert artifact.metadata.get("total_matches", 0) == 0


@pytest.mark.asyncio
async def test_invalid_entrypoint(context, messages):
    """Test agent behavior with invalid entrypoint."""
    params = {"url": "https://example.com"}
    agent = WebReaderAgent()
    params_model = WebReaderRequest(**params)

    await agent.run(
        context,
        request="test query",
        entrypoint="invalid_entrypoint",
        params=params_model,
    )

    assert messages, "Agent should yield messages"
    direct_messages = [m for m in messages if isinstance(m, DirectResponse)]
    assert any(
        "Unknown entrypoint: invalid_entrypoint" in m.text for m in direct_messages
    ), "Expected error message for invalid entrypoint"


@pytest.mark.asyncio
async def test_invalid_parameters(context, messages):
    """Test agent behavior with invalid parameters."""
    agent = WebReaderAgent()

    await agent.run(context, request="test query", entrypoint="read_web", params=None)

    assert messages, "Agent should yield messages"
    direct_messages = [m for m in messages if isinstance(m, DirectResponse)]
    assert any(
        "Invalid parameters for read_web" in m.text for m in direct_messages
    ), "Expected error message for invalid parameters"


@pytest.mark.asyncio
async def test_url_processing_error(context, messages):
    """Test agent behavior when URL processing fails."""
    with patch.object(WebReaderAgent, "__init__", lambda self: None):
        agent = WebReaderAgent()
        from openai import AsyncOpenAI

        from src.citations import CitationProcessor
        from src.search import WebPageSearch

        agent.rag = WebPageSearch()
        agent.citation_processor = CitationProcessor()
        agent.openai_client = AsyncOpenAI()

        # Mock URL processing to raise an exception
        agent.rag.process_url = AsyncMock(
            side_effect=Exception("Failed to process URL")
        )

        params = {"url": "https://invalid-url.com"}
        test_messages = await get_all_agent_messages_with_agent(
            agent, params, context, messages
        )

        assert test_messages, "Agent should yield messages"

        # Check for error response
        direct_messages = [m for m in test_messages if isinstance(m, DirectResponse)]
        assert any(
            "Error analyzing web content" in m.text for m in direct_messages
        ), "Expected error message for URL processing failure"

        # Check for error log
        process_messages = [
            m for m in test_messages if isinstance(m, ProcessLogResponse)
        ]
        assert any(
            "Error details:" in m.text for m in process_messages
        ), "Expected error details in process log"


def test_webreader_request_validation():
    """Test WebReaderRequest model validation."""
    # Valid URL - Pydantic HttpUrl adds trailing slash
    request = WebReaderRequest(url=HttpUrl("https://example.com"))
    assert str(request.url) == "https://example.com/"

    # Invalid URL should raise validation error
    with pytest.raises(Exception):
        WebReaderRequest(url="invalid-url")


if __name__ == "__main__":
    pytest.main([__file__])
