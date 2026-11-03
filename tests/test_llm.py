from types import SimpleNamespace

import pytest

from crag.llm import LLM, parse_json


def make_client(outputs):
    """Create a fake Groq client for offline testing."""

    calls = []

    def create(**kwargs):
        calls.append(kwargs)

        item = outputs.pop(0)

        if isinstance(item, Exception):
            raise item

        message = SimpleNamespace(content=item)

        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=message)
            ]
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=create
            )
        )
    )

    return client, calls


def make_llm(outputs, max_retries=3):
    client, calls = make_client(outputs)

    llm = LLM(
        api_key="test-key",
        model="test-text-model",
        vision_model="test-vision-model",
        max_retries=max_retries,
        client=client,
        sleep=lambda seconds: None,
    )

    return llm, calls


def test_parse_json_variants():

    assert parse_json(
        '{"a": 1}'
    ) == {"a": 1}

    assert parse_json(
        '```json\n{"a": 1}\n```'
    ) == {"a": 1}

    assert parse_json(
        'Result: {"a": 1} Done.'
    ) == {"a": 1}

    with pytest.raises(ValueError):
        parse_json("no json here")

    with pytest.raises(ValueError):
        parse_json("[1, 2, 3]")


def test_chat_retries_then_succeeds():

    llm, calls = make_llm([
        RuntimeError("rate limit"),
        "hello",
    ])

    result = llm.chat([
        {
            "role": "user",
            "content": "hi",
        }
    ])

    assert result == "hello"
    assert len(calls) == 2
    assert calls[0]["model"] == "test-text-model"


def test_chat_raises_after_max_retries():

    llm, calls = make_llm(
        [
            RuntimeError("temporary error"),
            RuntimeError("temporary error"),
            RuntimeError("temporary error"),
        ],
        max_retries=3,
    )

    with pytest.raises(RuntimeError):
        llm.chat([
            {
                "role": "user",
                "content": "hi",
            }
        ])

    assert len(calls) == 3


def test_json_mode_falls_back_when_unsupported():

    llm, calls = make_llm([
        RuntimeError("response_format not supported"),
        '{"ok": true}',
    ])

    result = llm.json([
        {
            "role": "user",
            "content": "Return JSON",
        }
    ])

    assert result == {"ok": True}

    assert "response_format" in calls[0]
    assert "response_format" not in calls[1]


def test_images_are_attached_and_vision_model_is_selected(
    tmp_path,
):

    image_path = tmp_path / "chart.png"
    image_path.write_bytes(b"\x89PNG fake")

    llm, calls = make_llm([
        "Image description"
    ])

    result = llm.chat(
        [
            {
                "role": "user",
                "content": "Describe this chart",
            }
        ],
        images=[image_path],
    )

    assert result == "Image description"

    assert calls[0]["model"] == "test-vision-model"

    content = calls[0]["messages"][-1]["content"]

    assert content[0] == {
        "type": "text",
        "text": "Describe this chart",
    }

    assert content[1]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )