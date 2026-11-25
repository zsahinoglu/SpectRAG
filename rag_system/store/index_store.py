from __future__ import annotations
import logging
from pathlib import Path
import faiss
from spectrag.models import Chunk
from spectrag.formats import MANIFEST_FILE, read_chunks, read_manifest, write_chunks

logger = logging.getLogger(__name__)

_FAISS_FILE   = "faiss.index"


class IndexStore:
    """
    Persists the two offline artefacts that live on disk: chunks and FAISS index,
    in the safe format of ``spectrag.formats`` (Parquet + .npy + JSON manifest with
    checksums; no pickle). Legacy ``chunks.pkl`` indexes are never read here; convert
    them once with ``python -m spectrag.formats.migrate <dir>``.
    The graph is stored separately (see InMemoryGraphStore / Neo4jGraphClient).
    """

    def __init__(self, store_path: Path | str) -> None:
        self.path = Path(store_path)

    def save(self, chunks: list[Chunk], faiss_index: faiss.Index) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        # Drop any old manifest first so an interrupted save is refused, not half-read.
        (self.path / MANIFEST_FILE).unlink(missing_ok=True)
        faiss.write_index(faiss_index, str(self.path / _FAISS_FILE))
        # spectral_coords and embeddings are stored as row-aligned .npy matrices with
        # explicit presence masks, so a chunk without coords can never shift the rest
        # (CASE-08).
        write_chunks(self.path, chunks, faiss_file=_FAISS_FILE)
        logger.info("IndexStore saved: %d chunks → %s", len(chunks), self.path)

    def load(self) -> tuple[list[Chunk], faiss.Index]:
        if not (self.path / MANIFEST_FILE).exists() and not (self.path / "chunks.pkl").exists():
            raise FileNotFoundError(
                f"Index artefact missing: {self.path / MANIFEST_FILE}\n"
                "Upload documents to build the index."
            )
        manifest = read_manifest(self.path)      # raises IndexFormatError; verifies checksums
        if _FAISS_FILE not in manifest["files"]:
            raise FileNotFoundError(f"{_FAISS_FILE} not part of the index at {self.path}")
        chunks = read_chunks(self.path)
        faiss_index = faiss.read_index(str(self.path / _FAISS_FILE))
        logger.info("IndexStore loaded: %d chunks from %s", len(chunks), self.path)
        return chunks, faiss_index
