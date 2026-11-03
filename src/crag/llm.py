"""Groq LLM wrapper for text, JSON, retries, and vision inputs."""

import base64
import json
import logging
import mimetypes
import re
import time

from pathlib import Path

from groq import Groq

from crag.config import Settings

logger = logging.getLogger("crag")


def parse_json(text: str) -> dict:
    """Parse a JSON object from model output."""

    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    )

    try:
        result = json.loads(cleaned)

    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start == -1 or end <= start:
            raise ValueError(
                f"No JSON object found: {text[:200]!r}"
            ) from None

        result = json.loads(cleaned[start:end + 1])

    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object.")

    return result


def image_part(path: str | Path) -> dict:
    """Convert a local image into a Base64 data URL."""

    path = Path(path)

    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"

    encoded_image = base64.b64encode(
        path.read_bytes()
    ).decode("utf-8")

    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{mime_type};base64,{encoded_image}"
        },
    }


class LLM:
    """Centralized client for Groq chat-completion requests."""

    def __init__(
        self,
        api_key: str,
        model: str,
        vision_model: str,
        max_retries: int = 3,
        client=None,
        sleep=time.sleep,
    ):
        self.model = model
        self.vision_model = vision_model
        self.max_retries = max_retries
        self._sleep = sleep

        # This wrapper manages retries itself.
        self.client = client or Groq(
            api_key=api_key,
            max_retries=0,
        )

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        model: str | None = None,
    ) -> "LLM":
        """Initialize the LLM using application settings."""

        return cls(
            api_key=settings.llm_api_key,
            model=model or settings.llm_model,
            vision_model=settings.vision_model,
        )

    def _attach_images(
        self,
        messages: list[dict],
        images: list[str | Path],
    ) -> list[dict]:
        """Attach images to the final message."""

        if not messages:
            raise ValueError("At least one message is required.")

        copied_messages = [
            dict(message) for message in messages
        ]

        last_message = copied_messages[-1]

        if not isinstance(last_message.get("content"), str):
            raise ValueError(
                "Final message content must be a text string."
            )

        text_content = last_message["content"]

        last_message["content"] = [
            {
                "type": "text",
                "text": text_content,
            },
            *[
                image_part(path)
                for path in images
            ],
        ]

        return copied_messages

    def chat(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        json_mode: bool = False,
        images: list[str | Path] | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send a chat-completion request to Groq."""

        if not messages:
            raise ValueError("At least one message is required.")

        has_images = bool(images)

        if has_images:
            messages = self._attach_images(
                messages,
                images,
            )

        selected_model = (
            model
            or (
                self.vision_model
                if has_images
                else self.model
            )
        )

        use_json_mode = json_mode
        last_error = None

        for attempt in range(self.max_retries):

            request = {
                "model": selected_model,
                "messages": messages,
                "temperature": temperature,
            }

            if use_json_mode:
                request["response_format"] = {
                    "type": "json_object"
                }

            if max_tokens is not None:
                request["max_completion_tokens"] = max_tokens

            try:
                response = (
                    self.client.chat.completions.create(
                        **request
                    )
                )

                return (
                    response.choices[0].message.content
                    or ""
                )

            except Exception as exc:
                last_error = exc

                if (
                    use_json_mode
                    and "response_format" in str(exc).lower()
                ):
                    logger.warning(
                        "Groq rejected JSON mode; "
                        "retrying without it."
                    )

                    use_json_mode = False
                    continue

                logger.warning(
                    "Groq request failed (attempt %d/%d): %s",
                    attempt + 1,
                    self.max_retries,
                    exc,
                )

                if attempt < self.max_retries - 1:
                    self._sleep(
                        0.5 * (2 ** attempt)
                    )

        raise RuntimeError(
            f"Groq request failed after "
            f"{self.max_retries} attempts: {last_error}"
        ) from last_error

    def json(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        images: list[str | Path] | None = None,
    ) -> dict:
        """Generate and parse a JSON response."""

        current_messages = list(messages)
        last_error = None

        for attempt in range(2):

            response_text = self.chat(
                current_messages,
                model=model,
                json_mode=True,
                images=images,
            )

            try:
                return parse_json(response_text)

            except ValueError as exc:
                last_error = exc

                if attempt == 0:
                    current_messages = current_messages + [
                        {
                            "role": "user",
                            "content": (
                                "Return ONLY one valid JSON object. "
                                "Do not include Markdown fences "
                                "or extra text."
                            ),
                        }
                    ]

        raise ValueError(
            f"Groq did not return valid JSON: {last_error}"
        ) from last_error