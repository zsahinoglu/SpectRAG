"""Packaging metadata checks (H2a)."""
import re
import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"
HEAVY = {"torch", "sentence-transformers", "gliner", "transformers"}


def _name(req: str) -> str:
    return re.split(r"[<>=!~\[; ]", req, maxsplit=1)[0].lower()


def _load():
    return tomllib.loads(PYPROJECT.read_text())["project"]


def test_required_extras_exist():
    extras = set(_load()["optional-dependencies"])
    assert {"neo4j", "gliner", "anthropic", "ollama", "rerank", "dev"} <= extras


def test_no_openai_anywhere_in_packaging():
    """ARCH2c: OpenAI backends removed by human decision (ARCH2d dropped)."""
    proj = _load()
    assert "openai" not in proj["optional-dependencies"]
    reqs = list(proj["dependencies"])
    for group in proj["optional-dependencies"].values():
        reqs += group
    assert "openai" not in {_name(r) for r in reqs}
    req_txt = PYPROJECT.with_name("requirements.txt").read_text().splitlines()
    assert not [ln for ln in req_txt if _name(ln.strip()) == "openai"]


def test_anthropic_in_default_install():
    assert "anthropic" in {_name(r) for r in _load()["dependencies"]}


def test_core_install_has_no_pytorch():
    core = {_name(r) for r in _load()["dependencies"]}
    assert not core & HEAVY
    assert not core & {"neo4j", "openai", "ollama"}
