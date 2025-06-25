from typing import Any

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator


class WebReaderRequest(BaseModel):
    """
    Request model for the web reader agent.

    Args:
        url (HttpUrl): Target webpage URL (http or https)
    """
    url: HttpUrl = Field(
        description="Target webpage URL (http or https)",
        examples=[
            "https://www.gbif.org",
            "https://www.worldwildlife.org",
            "https://www.un.org/en/climatechange/science/climate-issues/biodiversity",
        ],
    )

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
            if not data.get("url"):
                raise ValueError("url must be provided")
        return data

    model_config = {
        "str_strip_whitespace": True,
        "frozen": True,
        "json_schema_extra": {
            "examples": [
                {
                    "url": "https://www.un.org/en/climatechange/science/climate-issues/biodiversity",
                }
            ]
        },
    }
