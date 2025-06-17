from typing import Any

from pydantic import (BaseModel, Field, HttpUrl, StringConstraints,
                      field_validator, model_validator)


class WebReaderRequest(BaseModel):
    """
    Request model for the web reader agent.

    Args:
        query (str): Non-empty search query (1-200 characters)
        url (HttpUrl): Target webpage url (http or https)
    """
    query: str = Field(
        min_length=1,
        max_length=200,
        description="Non-empty search query (1-200 characters)",
        examples=[
            "species richness",
            "habitat loss",
        ],
    )
    url: HttpUrl = Field(
        description="Target webpage URL (http or https)",
        examples=[
            "https://www.gbif.org",
            "https://www.worldwildlife.org",
        ],
    )

    @field_validator("query")
    @classmethod
    def validate_query_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be blank or whitespace only")
        return v.strip()

    @field_validator("url")
    @classmethod
    def validate_url_scheme(cls, v: HttpUrl) -> HttpUrl:
        if v.scheme not in ("http", "https"):
            raise ValueError("URL scheme must be 'http' or 'https'")
        return v

    @model_validator(mode="before")
    @classmethod
    def validate_required_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "query" in data and isinstance(data["query"], str):
                data["query"] = data["query"].strip()
            if not data.get("query"):
                raise ValueError("query must be provided and non-empty")
            if not data.get("url"):
                raise ValueError("url must be provided")
        return data


    model_config = {
        "str_strip_whitespace": True,
        "frozen": True,
        "json_schema_extra": {
            "examples": [
                {
                    "query": "Impacts of climate change on biodiversity",
                    "url": "https://www.un.org/en/climatechange/science/climate-issues/biodiversity",
                }
            ]
        },
    }
