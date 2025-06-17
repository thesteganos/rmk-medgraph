# src/utils.py

import os
from langchain_community.embeddings import HuggingFaceEmbeddings
import sys

def get_embedding_model():
    """
    Initializes and returns the HuggingFace sentence transformer embedding model.
    Allows model selection via EMBEDDING_MODEL_NAME environment variable.
    Defaults to a medical-specific model if not set.
    Falls back to a general-purpose model if the specified or default model fails.
    """
    # CRITICAL WARNING: If you change the embedding model, especially to one with a
    # different embedding dimension (e.g., all-MiniLM-L6-v2 has dimension 384,
    # BioBERT-based models often have 768), you MUST re-run
    # `python populate_repository_graph.py` (which will clear existing graph data)
    # AND `python tagging_pipeline.py --build-hierarchy` to ensure all data and
    # vector indexes are consistent. Failure to do so will likely lead to errors
    # or incorrect results.

    # To use the general-purpose model (dimension 384) set EMBEDDING_MODEL_NAME="all-MiniLM-L6-v2"
    # DEFAULT_MEDICAL_MODEL = "all-MiniLM-L6-v2"

    # Current default model (BioBERT based, likely dimension 768):
    DEFAULT_MEDICAL_MODEL = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
    FALLBACK_MODEL = "all-MiniLM-L6-v2" # Fallback general-purpose model (dimension 384)

    model_name_from_env = os.getenv("EMBEDDING_MODEL_NAME")

    if model_name_from_env:
        model_name = model_name_from_env
        print(f"INFO: Using embedding model specified by environment variable EMBEDDING_MODEL_NAME: {model_name}")
    else:
        model_name = DEFAULT_MEDICAL_MODEL
        print(f"INFO: EMBEDDING_MODEL_NAME not set. Using default medical model: {model_name}")

    model_kwargs = {'device': 'cpu'}

    try:
        print(f"INFO: Attempting to load HuggingFace embedding model: '{model_name}'...")
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs
        )
        # Test embedding a short string to ensure model loads correctly
        embeddings.embed_query("test")
        print(f"INFO: Successfully loaded embedding model: '{model_name}'.")
        return embeddings
    except Exception as e:
        print(f"ERROR: Failed to load specified HuggingFace embedding model '{model_name}'. Error: {e}", file=sys.stderr)
        print(f"INFO: Falling back to: '{FALLBACK_MODEL}'")
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=FALLBACK_MODEL,
                model_kwargs=model_kwargs
            )
            # Test embedding
            embeddings.embed_query("test")
            print(f"INFO: Successfully loaded fallback embedding model: '{FALLBACK_MODEL}'.")
            return embeddings
        except Exception as fallback_e:
            print(f"FATAL ERROR: Failed to load fallback HuggingFace embedding model '{FALLBACK_MODEL}'. Error: {fallback_e}", file=sys.stderr)
            raise fallback_e
