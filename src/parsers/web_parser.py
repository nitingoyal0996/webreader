import re
from typing import Dict, List

import aiohttp
import trafilatura
from trafilatura.settings import use_config


class WebContentParser:
    """
    WebContentParser - Web Content Extraction and Parsing;
    Supports high-quality content extraction in markdown format.
    """

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36"
        }

        self.config = use_config()
        self.config.set("DEFAULT", "EXTRACTION_TIMEOUT", "30")
        self.config.set("DEFAULT", "MIN_EXTRACTED_SIZE", "200")
        self.config.set("DEFAULT", "MIN_OUTPUT_SIZE", "100")

    async def fetch_and_parse(self, url: str) -> Dict[str, any]:
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as session:
                async with session.get(url, headers=self.headers) as response:
                    if response.status == 200:
                        html_content = await response.text()
                        return self._parse_html_content(html_content, url)
                    else:
                        raise Exception(
                            f"Failed to fetch content. Status code: {response.status}"
                        )
        except aiohttp.ClientError as e:
            raise Exception(f"Network error: {str(e)}")
        except Exception as e:
            raise Exception(f"Error fetching content: {str(e)}")

    def _parse_html_content(self, html_content: str, url: str) -> Dict[str, any]:
        extracted_markdown = trafilatura.extract(
            html_content,
            config=self.config,
            output_format="markdown",
            include_comments=False,
            include_tables=True,
            include_formatting=True,
            include_links=True,
            target_language="en",
        )

        metadata = trafilatura.extract_metadata(html_content)

        if not extracted_markdown:
            raise Exception("Failed to extract meaningful content from the webpage")

        clean_content = self._clean_markdown(extracted_markdown)

        return {
            "url": url,
            "metadata": {
                "title": metadata.title if metadata and metadata.title else "Untitled",
                "description": (
                    metadata.description if metadata and metadata.description else ""
                ),
                "author": metadata.author if metadata and metadata.author else None,
                "publication_date": (
                    metadata.date if metadata and metadata.date else None
                ),
                "language": (
                    metadata.language if metadata and metadata.language else None
                ),
            },
            "content": clean_content,
        }

    def _clean_markdown(self, markdown: str) -> str:
        if not markdown:
            return ""

        lines = []
        prev_line_empty = False

        for line in markdown.split("\n"):
            line = line.rstrip()

            if not line.strip():
                if not prev_line_empty:
                    lines.append("")
                prev_line_empty = True
                continue

            lines.append(line)
            prev_line_empty = False

        cleaned = "\n".join(lines)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = re.sub(r"\*\s+\*", "", cleaned)
        cleaned = re.sub(r"_\s+_", "", cleaned)
        cleaned = re.sub(r"\[\s*\]\(\s*\)", "", cleaned)
        cleaned = re.sub(r"^(#{1,6})\s*$", "", cleaned, flags=re.MULTILINE)

        return cleaned.strip()
