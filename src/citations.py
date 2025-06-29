from __future__ import annotations

import re
from typing import Dict, List, Tuple
from urllib.parse import quote


class CitationProcessor:
    """Handles extraction and processing of citations from LLM responses."""

    def __init__(self):
        self.citation_regex = re.compile(
            r"<CIT\s+chunk_id=['\"](?P<cid>\d+)['\"]\s+sentences=['\"](?P<rng>[\d\-]+)['\"]\s*>(?P<src>.*?)</CIT>",
            re.DOTALL,
        )

    def extract_citations(self, text: str) -> Tuple[str, List[Dict]]:
        """
        Extract citations from LLM response text and return cleaned text + citation data.


        Returns:
            Tuple(clean_text_without_citations, list_of_citation_dicts)
        """
        clean_parts, citations, last_pos = [], [], 0
        for match in self.citation_regex.finditer(text):
            clean_parts.append(text[last_pos : match.start()])
            citations.append(
                {
                    "chunk_id": int(match.group("cid")),
                    "sentences": match.group("rng"),
                    "snippet": match.group("src").strip(),
                }
            )
            if clean_parts and not clean_parts[-1].endswith(" "):
                clean_parts.append(" ")
            # Add the citation content to clean text (without the tags)
            clean_parts.append(match.group("src").strip())
            last_pos = match.end()

        clean_parts.append(text[last_pos:])
        clean_text = "".join(clean_parts).strip()
        return clean_text, citations

    def resolve_citations(
        self, citations: List[Dict], search_hits: List[Dict], page_url: str
    ) -> List[Dict]:
        """
        Resolve citations to create text fragment URLs.
        """
        resolved = []
        for citation in citations:
            hit = next(
                (h for h in search_hits if h["chunk_id"] == citation["chunk_id"]), None
            )
            if hit is None:
                continue
            try:
                citation_text = citation.get("snippet", "").strip()
                if not citation_text:
                    continue
                start_text, end_text = self._prepare_fragment_text(citation_text)
                encoded_start = quote(start_text, safe="")
                encoded_end = quote(end_text, safe="")
                link = f"{page_url}#:~:text={encoded_start},{encoded_end}"
                fragment_text = f"{start_text}...{end_text}"
                resolved.append(
                    {
                        **citation,
                        "source_text": citation_text,
                        "link": link,
                        "fragment_text": fragment_text,
                    }
                )
            except Exception as e:
                print(f"Error creating citation link: {e}")
                continue
        return resolved

    def convert_to_markdown_links(self, text: str, citations: List[Dict]) -> str:
        """
        Convert citation tags in text to markdown hyperlinks.

        Args:
            text: Text containing citation tags
            citations: List of resolved citations with links
        """
        result = text
        for match in self.citation_regex.finditer(text):
            full_match = match.group(0)
            snippet = match.group("src").strip()
            matching_cite = next(
                (c for c in citations if c.get("snippet") == snippet), None
            )
            if matching_cite and matching_cite.get("link"):
                markdown_link = f"[{snippet}]({matching_cite['link']})"
                result = result.replace(full_match, markdown_link)
            else:
                result = result.replace(full_match, snippet)
        return result

    def process_citations(
        self, ai_response: str, search_hits: List[Dict], page_url: str
    ) -> Tuple[str, str, List[Dict]]:
        """
        Complete citation processing pipeline.

        Args:
            ai_response: Raw AI response with citation tags
            search_hits: Search result chunks
            page_url: Source page URL

        Returns:
            Tuple of (clean_text, markdown_text_with_links, resolved_citations)
        """
        clean_text, raw_citations = self.extract_citations(ai_response)
        resolved_citations = self.resolve_citations(
            raw_citations, search_hits, page_url
        )
        markdown_text = self.convert_to_markdown_links(ai_response, resolved_citations)
        return clean_text, markdown_text, resolved_citations

    def _prepare_fragment_text(self, text: str) -> tuple[str, str]:
        text = " ".join(text.split())
        sentences = text.split(". ")
        start_text = " ".join(sentences[0].split()[:2])
        end_text = " ".join(sentences[-1].split()[-2:])
        return start_text, end_text

    @staticmethod
    def get_citation_prompt():
        return """
You answer using ONLY the evidence chunks provided.
When you reference content from a chunk, wrap it like:
<CIT chunk_id="{cid}" sentences="{srange}">exact quoted text</CIT>

CRITICAL RULES:
- Each <CIT> tag must wrap only contiguous sentences from a single chunk, and the text inside must match exactly the corresponding sentences from the chunk's sentence_map.
- Do NOT combine sentences from different parts of the chunk or from different chunks in a single <CIT> tag.
- If you need to cite multiple sentences from different places, use multiple <CIT> tags.
- chunk_id must exactly match the number in brackets [chunk_id] from the provided chunks
- sentences should be "1" for first sentence, "1-2" for first two sentences, etc.
- DO NOT repeat text outside and inside citation tags
- ALL your answer content should be wrapped in citation tags
- Copy the exact words, punctuation, and spacing from the source chunks
- Do NOT add extra text outside the citation tags

Return your complete answer with all content properly cited."""
