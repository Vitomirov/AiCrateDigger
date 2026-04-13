from typing import Literal

from pydantic import BaseModel, Field


class ParsedQuery(BaseModel):
    artist: str = Field(..., description="Resolved artist name")
    album: str = Field(..., description="Resolved album/release title")
    format: Literal["Vinyl", "CD", "Cassette"] = Field(
        ...,
        description="Requested physical format",
    )
    country: str = Field(..., description="Primary target country")
    language: str = Field(..., description="Detected query language")
    original_query: str = Field(..., description="Original user input")


class ParseRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Raw user query text")
