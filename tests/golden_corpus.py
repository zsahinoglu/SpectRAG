"""Fixed 12-chunk, 2-document corpus and an offline index builder for golden tests."""
from __future__ import annotations

from config import Config
from rag_system.indexing.cluster_assigner import SpectralClusterAssigner
from rag_system.indexing.edges.adjacency import AdjacencyEdgeExtractor
from rag_system.indexing.edges.base import RawEdge
from rag_system.indexing.edges.section import SectionEdgeExtractor
from rag_system.indexing.edges.semantic import SemanticEdgeExtractor
from rag_system.indexing.graph_builder import GraphBuilder
from rag_system.indexing.spectral import SpectralComputer
from spectrag.models import Chunk
from rag_system.store.memory_graph import InMemoryGraphStore

from tests.fakes import FakeEmbedder, WhitespaceEncoder

# (doc_id, section_path, text, utility_questions)
_RAW = [
    ("astro", ["Stars"], "A star is a luminous sphere of plasma held together by gravity. Stars form in nebula clouds.", ["What is a star made of?"]),
    ("astro", ["Stars"], "Nuclear fusion in the stellar core converts hydrogen into helium and releases energy.", ["How does stellar fusion produce energy?"]),
    ("astro", ["Stars", "Death"], "Massive stars end as supernova explosions leaving a neutron star or black hole.", ["How do massive stars die?"]),
    ("astro", ["Planets"], "Planets orbit a star. Rocky planets such as Mars have thin atmospheres.", ["What do planets orbit?"]),
    ("astro", ["Planets"], "Gas giants like Jupiter have deep hydrogen atmospheres and many moons.", []),
    ("astro", ["Galaxies"], "A galaxy contains billions of stars bound by gravity; the Milky Way is a spiral galaxy.", ["What is the Milky Way?"]),
    ("cook", ["Bread"], "Bread dough needs flour, water, yeast and salt. Knead the dough until elastic.", ["How do you make bread dough?"]),
    ("cook", ["Bread"], "Let the dough rise in a warm place; yeast ferments sugar and releases gas.", ["Why does bread dough rise?"]),
    ("cook", ["Soup"], "Simmer onions, carrots and celery in stock for a basic vegetable soup.", ["How do you cook vegetable soup?"]),
    ("cook", ["Soup"], "Season the soup with salt and pepper and serve hot with fresh bread.", []),
    ("cook", ["Sauces"], "A roux of butter and flour thickens sauces; whisk in milk for bechamel.", ["How do you thicken a sauce?"]),
    ("cook", ["Sauces"], "Tomato sauce simmers garlic, basil and tomatoes with olive oil.", []),
]

# Cross-document "entity/concept" edges: shared concepts the embedder cannot see.
_ENTITY_EDGES = [RawEdge(1, 6, 0.9), RawEdge(7, 1, 0.7), RawEdge(2, 5, 0.8)]

QUESTIONS = [
    "How do stars produce energy through fusion?",
    "What ingredients does bread dough need?",
    "How do you thicken sauces with flour?",
    "Which planets have atmospheres?",
]


def make_config() -> Config:
    cfg = Config()
    cfg.indexing.n_spectral_components = 4
    cfg.indexing.n_offline_clusters = 3
    cfg.indexing.semantic_threshold = 0.2
    cfg.retrieval.seed_k = 4
    cfg.retrieval.uq_seed_k = 4
    cfg.retrieval.bm25_seed_k = 4
    cfg.retrieval.token_budget = 400
    return cfg


def make_chunks() -> list[Chunk]:
    chunks: list[Chunk] = []
    positions: dict[str, int] = {}
    for doc, section, text, uqs in _RAW:
        pos = positions.get(doc, 0)
        positions[doc] = pos + 1
        chunks.append(Chunk(
            chunk_id=f"{doc}-{pos}", doc_id=doc, text=text, position=pos,
            section_path=list(section), references=[], utility_questions=list(uqs),
        ))
    return chunks


def build_index(cfg: Config | None = None, embedder: FakeEmbedder | None = None):
    """Index the fixed corpus offline. Returns (cfg, embedder, chunks, faiss_index, store)."""
    cfg = cfg or make_config()
    embedder = embedder or FakeEmbedder()
    chunks = make_chunks()
    embedder.embed_chunks(chunks)
    faiss_index = embedder.build_faiss_index(chunks)

    semantic = SemanticEdgeExtractor(cfg.indexing).extract(chunks)
    adjacency = AdjacencyEdgeExtractor().extract(chunks)
    section = SectionEdgeExtractor(cfg.indexing).extract(chunks)
    graph = GraphBuilder(cfg).build(
        chunks, semantic, adjacency, section, _ENTITY_EDGES, [], [],
    )
    SpectralComputer(cfg.indexing).compute(graph, chunks)
    SpectralClusterAssigner(cfg.indexing).assign(chunks)

    store = InMemoryGraphStore()
    store.write_graph(chunks, graph)
    return cfg, embedder, chunks, faiss_index, store


def offline_encoder() -> WhitespaceEncoder:
    return WhitespaceEncoder()
