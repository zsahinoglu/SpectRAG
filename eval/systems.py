"""Index building and the two systems under test: SpectRAG and the NaiveRAG baseline.

The harness drives the library's public stage classes; it never modifies them.
"""
from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from config import Config
from rag_system.indexing.cluster_assigner import SpectralClusterAssigner
from rag_system.indexing.edges.adjacency import AdjacencyEdgeExtractor
from rag_system.indexing.edges.section import SectionEdgeExtractor
from rag_system.indexing.edges.semantic import SemanticEdgeExtractor
from rag_system.indexing.graph_builder import GraphBuilder
from rag_system.indexing.spectral import SpectralComputer
from rag_system.models import Chunk
from rag_system.query.cluster_selection import ClusterAwareSelector
from rag_system.query.diffusion import RelevanceDiffuser
from rag_system.query.graph_expansion import GraphExpander
from rag_system.query.seed_retrieval import SeedRetriever
from rag_system.store.memory_graph import InMemoryGraphStore

from eval.data import Dataset


@dataclass
class Index:
    chunks: list[Chunk]
    faiss_index: object
    store: InMemoryGraphStore
    embedder: object
    doc_of: dict[str, str] = field(default_factory=dict)  # chunk_id -> dataset doc_id


@dataclass
class Retrieval:
    ranked_ids: list[str]        # selected chunk ids, best first
    ranked_doc_ids: list[str]    # dataset doc ids of ranked_ids
    chunks: list[Chunk]          # selected chunks in rank order (generator input)
    top10_ids: list[str]         # top-10 candidates by final score (stability metric)
    answer: str = ""             # filled in by the runner when a generator is used


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def build_offline_index(cfg: Config, ds: Dataset, embedder) -> Index:
    """No-network index for smoke runs: one chunk per paragraph and only the edge
    signals that need no LLM (semantic, adjacency, section)."""
    chunks = []
    for doc in ds.docs:
        for pos, para in enumerate(_paragraphs(doc.text)):
            chunks.append(Chunk(chunk_id=f"{doc.doc_id}#{pos}", doc_id=doc.doc_id, text=para,
                                position=pos, section_path=[doc.title or doc.doc_id],
                                references=[], metadata={"source": doc.doc_id}))
    embedder.embed_chunks(chunks)
    faiss_index = embedder.build_faiss_index(chunks)
    semantic = SemanticEdgeExtractor(cfg.indexing).extract(chunks)
    adjacency = AdjacencyEdgeExtractor().extract(chunks)
    section = SectionEdgeExtractor(cfg.indexing).extract(chunks)
    graph = GraphBuilder(cfg).build(chunks, semantic, adjacency, section, [], [], [])
    SpectralComputer(cfg.indexing).compute(graph, chunks)
    SpectralClusterAssigner(cfg.indexing).assign(chunks)
    store = InMemoryGraphStore()
    store.write_graph(chunks, graph)
    return Index(chunks, faiss_index, store, embedder, {c.chunk_id: c.doc_id for c in chunks})


def build_full_index(cfg: Config, ds: Dataset, workdir: Path) -> Index:
    """Full library indexing pass (Ollama embedder, Claude Haiku index-time LLM)."""
    from rag_system.indexing.pipeline import IndexingPipeline

    docs_dir = workdir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    stem_to_doc, paths = {}, []
    for i, doc in enumerate(ds.docs):  # file stem becomes the library's doc_id
        stem = f"d{i:07d}"
        stem_to_doc[stem] = doc.doc_id
        p = docs_dir / f"{stem}.txt"
        p.write_text((f"# {doc.title}\n\n" if doc.title else "") + doc.text, encoding="utf-8")
        paths.append(p)
    cfg.store_path = workdir / "index"
    store = InMemoryGraphStore()
    pipe = IndexingPipeline(cfg, store)
    pipe.index(paths)
    chunks, faiss_index = _reload(cfg.store_path)
    return Index(chunks, faiss_index, store, pipe.embedder,
                 {c.chunk_id: stem_to_doc.get(c.doc_id, c.doc_id) for c in chunks})


def _reload(path: Path):
    from rag_system.store.index_store import IndexStore
    return IndexStore(path).load()


class SpectRAG:
    """Stages 1–4 of the library query pipeline (generation is done by the runner)."""

    def __init__(self, cfg: Config, index: Index, encoder=None) -> None:
        self.cfg, self.index = cfg, index
        self.seeds = SeedRetriever(cfg, index.embedder)
        self.expander = GraphExpander(cfg)
        self.diffuser = RelevanceDiffuser(cfg)
        self.selector = ClusterAwareSelector(cfg)
        if encoder is not None:
            self.selector._enc = encoder

    def retrieve(self, question: str) -> Retrieval:
        ix = self.index
        seeds, all_scores = self.seeds.retrieve(question, ix.chunks, ix.faiss_index)
        cands = self.expander.expand(seeds, ix.chunks, ix.store)
        sub = ix.store.subgraph({ix.chunks[i].chunk_id for i in cands})
        diffused = self.diffuser.diffuse(cands, seeds, all_scores, ix.chunks, sub)
        selected = self.selector.select(cands, diffused, all_scores, ix.chunks, sub, question)
        key = lambda i: (-diffused.get(i, 0.0), ix.chunks[i].chunk_id)  # noqa: E731
        ranked = sorted(selected, key=key)
        top10 = sorted(diffused, key=key)[:10]
        return _result(ix, ranked, top10)


class NaiveRAG:
    """Baseline: dense top-k by cosine, packed into the same token budget as SpectRAG."""

    def __init__(self, cfg: Config, index: Index, encoder) -> None:
        self.index, self.encoder = index, encoder
        self.budget = cfg.retrieval.token_budget
        self.k = max(cfg.retrieval.seed_k, 10)

    def retrieve(self, question: str) -> Retrieval:
        ix = self.index
        q = np.asarray(ix.embedder.embed([question], kind="query"), dtype=np.float32)
        _, idx = ix.faiss_index.search(q, min(self.k, len(ix.chunks)))
        order = [int(i) for i in idx[0] if i >= 0]
        picked, used = [], 0
        for i in order:
            n = len(self.encoder.encode(ix.chunks[i].text))
            if picked and used + n > self.budget:
                break
            picked.append(i)
            used += n
        return _result(ix, picked, order[:10])


def _result(ix: Index, ranked: list[int], top10: list[int]) -> Retrieval:
    ids = [ix.chunks[i].chunk_id for i in ranked]
    return Retrieval(ids, [ix.doc_of.get(c, c) for c in ids], [ix.chunks[i] for i in ranked],
                     [ix.chunks[i].chunk_id for i in top10])


def tiktoken_encoder(cfg: Config):
    import tiktoken
    return tiktoken.get_encoding(cfg.indexing.tiktoken_encoding)


def temp_workdir(prefix: str) -> tempfile.TemporaryDirectory:
    return tempfile.TemporaryDirectory(prefix=f"spectrag-eval-{prefix}-")
