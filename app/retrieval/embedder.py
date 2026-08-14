"""
Embedding wrapper for multilingual-e5-base.

Critical detail that's easy to get wrong: E5 models are trained with
"query: " and "passage: " prefixes on their inputs. This isn't
cosmetic — the model was contrastively trained to treat query-prefixed
and passage-prefixed text differently, so that a query and its matching
passage end up close in vector space specifically BECAUSE of the
prefix convention, not despite it. Skip the prefix and cosine
similarity scores get measurably worse and less discriminative — you
won't get an error, you'll just get worse retrieval that's hard to
attribute to this specific mistake. This is the single most common
E5-specific bug: forgetting to prefix consistently between
ingestion-time (passage) and query-time (query) embedding calls.
"""

from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-base"
EMBEDDING_DIM = 768  # must match the `vector(768)` column in init.sql

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_passages(texts: list[str]) -> list[list[float]]:
    """Embed chunk text for storage. Use at ingestion time."""
    model = get_model()
    prefixed = [f"passage: {t}" for t in texts]
    vectors = model.encode(prefixed, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    """Embed a user question for retrieval. Use at query time."""
    model = get_model()
    vector = model.encode(f"query: {text}", normalize_embeddings=True, show_progress_bar=False)
    return vector.tolist()
