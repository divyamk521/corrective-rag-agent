
import numpy as np
import pytest

from crag.embeddings import Embeddings


class FakeEmbeddingModel:
    """Fake model for testing without downloading a real model."""

    def __init__(self):
        self.received_texts = []
        self.received_options = {}

    def encode(self, texts, **kwargs):
        self.received_texts = texts
        self.received_options = kwargs

        # Return predictable two-dimensional vectors.
        return np.array(
            [
                [3.0, 4.0]
                for _ in texts
            ],
            dtype=np.float32,
        )


def test_embed_documents_returns_normalized_vectors():

    fake_model = FakeEmbeddingModel()

    embeddings = Embeddings(
        model_name="test-model",
        model=fake_model,
    )

    result = embeddings.embed_documents([
        "First document",
        "Second document",
    ])

    assert len(result) == 2
    assert len(result[0]) == 2

    # [3, 4] normalized becomes [0.6, 0.8].
    assert result[0] == pytest.approx([0.6, 0.8])
    assert result[1] == pytest.approx([0.6, 0.8])

    assert fake_model.received_texts == [
        "First document",
        "Second document",
    ]

    assert fake_model.received_options["normalize_embeddings"] is True


def test_embed_query_applies_query_prefix():

    fake_model = FakeEmbeddingModel()

    embeddings = Embeddings(
        model_name="test-model",
        query_prefix="Represent this sentence for searching relevant passages: ",
        model=fake_model,
    )

    result = embeddings.embed_query("What is corrective RAG?")

    assert fake_model.received_texts == [
        "Represent this sentence for searching relevant passages: "
        "What is corrective RAG?"
    ]

    assert result == pytest.approx([0.6, 0.8])


def test_empty_inputs_are_handled():

    fake_model = FakeEmbeddingModel()

    embeddings = Embeddings(
        model_name="test-model",
        model=fake_model,
    )

    assert embeddings.embed_documents([]) == []

    with pytest.raises(ValueError):
        embeddings.embed_documents(["valid text", "  "])

    with pytest.raises(ValueError):
        embeddings.embed_query("  ")
