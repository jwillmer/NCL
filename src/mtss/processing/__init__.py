"""Processing layer for mtss.

Intentionally empty of eager re-exports. The old re-exports pulled in
``..ingest`` and ``..parsers`` transitively — those live in the ``ingest``
extras and are absent from the API image, so loading them at import time
crashes API startup with ``ModuleNotFoundError: langchain_text_splitters``.

Consumers import submodules directly (``mtss.processing.entity_cache``,
``mtss.processing.embeddings``, ``mtss.processing.image_processor``, …).
"""
