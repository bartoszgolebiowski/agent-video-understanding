from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Type, TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from .config import LLMConfig
from .exceptions import LLMCallError, LLMConfigurationError

TModel = TypeVar("TModel", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class VideoLLMConfig:
    """Immutable configuration for video-enabled LLM client."""

    api_key: str
    model: str = "google/gemini-2.5-flash-lite"
    base_url: str = "https://openrouter.ai/api/v1"
    temperature: float = 0.2
    max_tokens: int = 1000000

    @classmethod
    def from_env(cls) -> "VideoLLMConfig":
        """Build a config object using standard environment variables."""
        import os

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise LLMConfigurationError(
                "OPENROUTER_API_KEY environment variable must be set"
            )

        return cls(
            api_key=api_key,
            model=os.getenv("VIDEO_MODEL", "google/gemini-2.5-flash-lite"),
            base_url=os.getenv("VIDEO_BASE_URL", "https://openrouter.ai/api/v1"),
            temperature=float(os.getenv("VIDEO_TEMPERATURE", "0.2")),
            max_tokens=int(os.getenv("VIDEO_MAX_TOKENS", "1000000")),
        )


@dataclass(slots=True)
class VideoLLMClient:
    """LLM client with support for video and image understanding."""

    config: VideoLLMConfig
    _client: Optional[OpenAI] = None

    def __post_init__(self) -> None:
        if not self.config.api_key:
            raise LLMConfigurationError("VideoLLMConfig.api_key must be provided")
        if self._client is None:
            self._client = OpenAI(
                api_key=self.config.api_key, base_url=self.config.base_url
            )

    @classmethod
    def from_env(cls) -> "VideoLLMClient":
        """Convenience constructor that loads configuration from env variables."""
        return cls(config=VideoLLMConfig.from_env())

    def invoke_with_media(
        self,
        *,
        text: str,
        image_blobs: Optional[list[str]] = None,
        output_model: Type[TModel],
    ) -> TModel:
        """
        Execute a prompt with optional image inputs using OpenRouter API.

        Args:
            text: The text prompt/question
            image_blobs: List of base64-encoded images (with or without data URI prefix)

        Returns:
            Validated instance of output_model
        """
        # Build content array with text and images
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": text,
            }
        ]

        # Add images to content array
        if image_blobs:
            for blob in image_blobs:
                content.append({"type": "image_url", "image_url": {"url": blob}})

        try:
            response = self._client.chat.completions.create(  # type: ignore[union-attr]
                model=self.config.model,
                messages=[
                    {
                        "role": "user",
                        "content": content,
                    }  # type: ignore
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        except Exception as exc:
            print(f"Video LLM invocation error: {exc}")
            raise LLMCallError("Failed to execute video LLM call") from exc

        # Extract message content from OpenAI response
        if not response.choices or not response.choices[0].message.content:
            raise LLMCallError("Video LLM returned empty response")

        content_text = response.choices[0].message.content

        # Wrap plain text response in expected format for output_model
        try:
            return output_model.model_validate({"content": content_text})
        except ValidationError as exc:
            raise LLMCallError("LLM response failed schema validation") from exc
