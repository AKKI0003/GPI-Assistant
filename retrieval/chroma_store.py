"""
retrieval/chroma_store.py

Persistent vector-DB backed storage for the PDF knowledge base,
replacing the single big JSON cache file that retrieval/pdf_retriever.py
used to read/write in full on every ingest. Chroma stores each chunk
(text + embedding + metadata) as its own record in an on-disk,
incrementally-updatable collection -- adding/updating one file's chunks
no longer requires reading and rewriting every OTHER file's chunks too,
and lookups (checksums, per-source deletes) are indexed instead of a
linear scan through one giant in-memory dict.

This module is purely the persistence layer. All retrieval/scoring
logic (financial-vs-ESG boosts, statement-matching, round-robin-by-
source, etc.) in pdf_retriever.py is unchanged -- it still operates on
an in-memory list of (chunk_dict, embedding) pairs; that list is just
now loaded from/written to Chroma instead of a JSON file.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import chromadb
    _CHROMA_AVAILABLE = True
    _CHROMA_IMPORT_ERROR: Optional[Exception] = None
except ImportError as exc:  # pragma: no cover - environment-dependent
    _CHROMA_AVAILABLE = False
    _CHROMA_IMPORT_ERROR = exc

_CHUNKS_COLLECTION = "pdf_knowledge_chunks"
_META_COLLECTION = "pdf_knowledge_meta"
_VERSION_DOC_ID = "__ingestion_logic_version__"

# Chroma refuses to .get()/.upsert() with an empty `ids` list, and a
# read-then-batch pattern is friendlier to it than one call per chunk.
_BATCH_SIZE = 200


class ChromaStore:
    """Thin persistence wrapper around a local, on-disk Chroma database."""

    def __init__(
        self,
        persist_dir: str,
        chunks_collection: str = _CHUNKS_COLLECTION,
        meta_collection: str = _META_COLLECTION,
    ):
        if not _CHROMA_AVAILABLE:
            raise RuntimeError(
                f"chromadb is not installed ({_CHROMA_IMPORT_ERROR}); "
                "install it with `pip install chromadb`."
            )
        os.makedirs(persist_dir, exist_ok=True)
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._chunks_name = chunks_collection
        self._meta_name = meta_collection
        self._chunks = self._client.get_or_create_collection(chunks_collection)
        self._meta = self._client.get_or_create_collection(meta_collection)
        self._cached_dim: Optional[int] = None

    # -- ingestion-logic version ---------------------------------------------

    def get_logic_version(self) -> int:
        try:
            result = self._meta.get(ids=[_VERSION_DOC_ID], include=["metadatas"])
        except Exception:
            return 0
        metas = result.get("metadatas") or []
        if metas and metas[0] and "version" in metas[0]:
            try:
                return int(metas[0]["version"])
            except (TypeError, ValueError):
                return 0
        return 0

    def set_logic_version(self, version: int) -> None:
        self._meta.upsert(
            ids=[_VERSION_DOC_ID], documents=[""], metadatas=[{"version": int(version)}],
            embeddings=[[0.0]],
        )

    def clear_all(self) -> None:
        """Wipe every stored chunk and checksum (used when the
        ingestion-logic version has changed and old chunks can no
        longer be trusted)."""
        for collection, name in ((self._chunks, self._chunks_name), (self._meta, self._meta_name)):
            try:
                self._client.delete_collection(name)
            except Exception:
                pass
        self._chunks = self._client.get_or_create_collection(self._chunks_name)
        self._meta = self._client.get_or_create_collection(self._meta_name)
        self._cached_dim = None

    # -- per-file checksums ---------------------------------------------------

    def get_checksums(self) -> Dict[str, str]:
        try:
            result = self._meta.get(include=["metadatas"])
        except Exception as exc:
            logger.warning("Failed to read checksums from Chroma: %s", exc)
            return {}
        checksums: Dict[str, str] = {}
        for doc_id, meta in zip(result.get("ids", []), result.get("metadatas", [])):
            if doc_id == _VERSION_DOC_ID or not meta:
                continue
            checksum = meta.get("checksum")
            if checksum:
                checksums[doc_id] = checksum
        return checksums

    def set_checksum(self, path: str, checksum: str) -> None:
        self._meta.upsert(
            ids=[path], documents=[""], metadatas=[{"checksum": checksum}],
            embeddings=[[0.0]],
        )

    def delete_checksum(self, path: str) -> None:
        try:
            self._meta.delete(ids=[path])
        except Exception as exc:
            logger.debug("Failed to delete checksum entry for %s: %s", path, exc)

    # -- chunks ---------------------------------------------------------------

    def load_all_chunks(self) -> List[Tuple[Dict[str, Any], Optional[List[float]]]]:
        """Returns (chunk_dict, embedding) pairs for every stored chunk."""
        try:
            result = self._chunks.get(include=["metadatas", "embeddings", "documents"])
        except Exception as exc:
            logger.warning("Failed to load chunks from Chroma: %s", exc)
            return []

        out: List[Tuple[Dict[str, Any], Optional[List[float]]]] = []
        ids = result.get("ids") or []
        metadatas = result.get("metadatas") or []
        embeddings = result.get("embeddings")
        documents = result.get("documents") or []

        for i in range(len(ids)):
            meta = metadatas[i] if i < len(metadatas) and metadatas[i] else {}
            raw = meta.get("chunk_json")
            chunk_dict: Dict[str, Any] = {}
            if raw:
                try:
                    chunk_dict = json.loads(raw)
                except (TypeError, ValueError) as exc:
                    logger.warning("Corrupt chunk_json for id %s: %s", ids[i], exc)
            if "text" not in chunk_dict and i < len(documents):
                chunk_dict["text"] = documents[i]

            embedding: Optional[List[float]] = None
            # embeddings can come back as a numpy array; len()/indexing
            # both work, but an explicit None-check on the array itself
            # raises ("truth value of an array is ambiguous"), so check
            # the container first. `has_embedding` distinguishes a real
            # stored embedding from the [0.0] placeholder written when
            # embed() failed at ingest time (see upsert_chunks).
            if (
                meta.get("has_embedding", True)
                and embeddings is not None
                and i < len(embeddings)
                and embeddings[i] is not None
            ):
                embedding = list(embeddings[i])
            out.append((chunk_dict, embedding))
        return out

    def delete_chunks_for_source(self, source: str) -> None:
        try:
            self._chunks.delete(where={"source": source})
        except Exception as exc:
            logger.warning("Failed to delete existing Chroma chunks for %s: %s", source, exc)

    def _existing_dim(self) -> int:
        """
        The embedding dimension the collection was already created with
        (inferred from any one already-stored real embedding), used as
        the fallback size for a placeholder vector when an entire batch
        has no real embedding of its own to measure. Cached after the
        first successful lookup since a collection's dimensionality
        never changes once real data exists in it.

        Falls back to 768 (nomic-embed-text's dimension -- the embedding
        model this project's ingestion docs call for) only when the
        collection is CURRENTLY EMPTY, i.e. there is nothing yet to
        conflict with; this is a reasonable default for that specific
        case, not a guess that overrides real stored data.
        """
        cached = getattr(self, "_cached_dim", None)
        if cached is not None:
            return cached
        try:
            peek = self._chunks.peek(limit=1)
            embeds = peek.get("embeddings")
            if embeds is not None and len(embeds) > 0 and embeds[0] is not None:
                dim = len(embeds[0])
                self._cached_dim = dim
                return dim
        except Exception as exc:
            logger.debug("Could not peek existing embedding dimension: %s", exc)
        return 768

    def upsert_chunks(
        self,
        chunk_dicts: List[Dict[str, Any]],
        embeddings: List[Optional[List[float]]],
    ) -> None:
        if not chunk_dicts:
            return

        for start in range(0, len(chunk_dicts), _BATCH_SIZE):
            batch_dicts = chunk_dicts[start:start + _BATCH_SIZE]
            batch_embeds = embeddings[start:start + _BATCH_SIZE]

            ids: List[str] = []
            documents: List[str] = []
            metadatas: List[Dict[str, Any]] = []
            embed_batch: List[List[float]] = []

            # Chroma requires every embedding in one upsert call to share
            # the same dimensionality, AND to match whatever dimension
            # the collection was already created with from earlier
            # upserts. Real embeddings in a batch always share the
            # embedding model's dimension, so normally that's enough to
            # size the placeholder correctly -- but if EVERY embedding in
            # this particular batch is missing (e.g. Ollama was
            # unreachable for this whole batch during ingestion, even
            # though earlier/later batches succeeded), there's no real
            # embedding in the batch to measure, and falling back to a
            # bare `1` produces a 1-dim placeholder that conflicts with
            # the 768-dim (or whatever the embedding model produces)
            # vectors already stored in the collection from previous
            # runs -- Chroma then rejects the ENTIRE batch with
            # "Collection expecting embedding with dimension of 768, got
            # 1", silently losing every chunk in it. Falling back to the
            # collection's own already-established dimension (queried
            # once and cached) instead of a hardcoded `1` fixes this.
            real_dim = next((len(e) for e in batch_embeds if e is not None), None)
            if real_dim is None:
                real_dim = self._existing_dim()

            for offset, (chunk_dict, embedding) in enumerate(zip(batch_dicts, batch_embeds)):
                chunk_id = str(
                    chunk_dict.get("chunk_id")
                    or f"{chunk_dict.get('source')}:{chunk_dict.get('page')}:{start + offset}"
                )
                ids.append(chunk_id)
                documents.append(chunk_dict.get("text") or "")
                metadatas.append({
                    "source": chunk_dict.get("source") or "",
                    "page": chunk_dict.get("page") if chunk_dict.get("page") is not None else -1,
                    "is_table": bool(chunk_dict.get("is_table", False)),
                    "company": chunk_dict.get("company") or "",
                    "chunk_json": json.dumps(chunk_dict, ensure_ascii=False),
                    "has_embedding": embedding is not None,
                })
                # A chunk with no real embedding (embed() failed/backend
                # unreachable at ingest time) still needs SOME vector
                # here -- Chroma requires one value per id in this call,
                # and if any id in the batch lacks one, Chroma silently
                # falls back to its OWN default embedding function,
                # which tries to download an ONNX model over the network
                # on first use. A same-dimension zero-vector placeholder
                # plus the `has_embedding` flag above lets
                # load_all_chunks() correctly restore `embedding=None`
                # for these, exactly matching how the old JSON cache
                # stored a bare `null`.
                embed_batch.append(embedding if embedding is not None else [0.0] * real_dim)

            try:
                self._chunks.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embed_batch)
            except Exception as exc:
                logger.warning("Failed to upsert %d chunk(s) into Chroma: %s", len(ids), exc)

    def count(self) -> int:
        try:
            return self._chunks.count()
        except Exception:
            return 0

    # -- generic JSON records (used by the ingest/ pipeline's document chunks,
    #    which don't carry their own embedding -- knowledge_index._embed_items()
    #    computes/caches those uniformly for every item, curated JSON and
    #    document-sourced alike, via the untouched embedding_store disk cache) --

    def upsert_records(self, records: List[Dict[str, Any]]) -> None:
        """
        Store arbitrary JSON-able record dicts (shape:
        {category, section, source_file, chunk_type, record: {...}}, as
        produced by ingest/ingest.py). No real embedding is stored here —
        a 1-dim placeholder is used since this collection is pure
        key/value persistence, not vector search; retrieval still goes
        through knowledge_index's own embedding pipeline exactly as
        before. Each id is derived from source_file + position so
        re-ingesting one document only touches its own ids.
        """
        if not records:
            return
        for start in range(0, len(records), _BATCH_SIZE):
            batch = records[start:start + _BATCH_SIZE]
            ids: List[str] = []
            documents: List[str] = []
            metadatas: List[Dict[str, Any]] = []
            embeds: List[List[float]] = []
            for offset, rec in enumerate(batch):
                source = rec.get("source_file") or "unknown"
                ids.append(f"{source}:{start + offset}")
                documents.append(((rec.get("record") or {}).get("content")) or "")
                metadatas.append({
                    "source": source,
                    "record_json": json.dumps(rec, ensure_ascii=False),
                })
                embeds.append([0.0])
            try:
                self._chunks.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeds)
            except Exception as exc:
                logger.warning(
                    "Failed to upsert %d record(s) into Chroma collection %s: %s",
                    len(ids), self._chunks_name, exc,
                )

    def load_all_records(self) -> List[Dict[str, Any]]:
        try:
            result = self._chunks.get(include=["metadatas"])
        except Exception as exc:
            logger.warning(
                "Failed to load records from Chroma collection %s: %s",
                self._chunks_name, exc,
            )
            return []
        out: List[Dict[str, Any]] = []
        for meta in (result.get("metadatas") or []):
            if not meta:
                continue
            raw = meta.get("record_json")
            if not raw:
                continue
            try:
                out.append(json.loads(raw))
            except (TypeError, ValueError) as exc:
                logger.warning(
                    "Corrupt record_json in Chroma collection %s: %s",
                    self._chunks_name, exc,
                )
        return out

    def delete_records_for_source(self, source_file: str) -> None:
        try:
            self._chunks.delete(where={"source": source_file})
        except Exception as exc:
            logger.warning(
                "Failed to delete existing records for %s from Chroma collection %s: %s",
                source_file, self._chunks_name, exc,
            )

    def sources(self) -> List[str]:
        """Distinct source_file values currently stored (used to report
        what's in the collection without loading every record's full
        JSON)."""
        try:
            result = self._chunks.get(include=["metadatas"])
        except Exception:
            return []
        seen = set()
        for meta in (result.get("metadatas") or []):
            if meta and meta.get("source"):
                seen.add(meta["source"])
        return sorted(seen)
