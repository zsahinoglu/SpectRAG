"""Shared pytest setup."""
import pytest

# The conformance suites are helper modules, not test modules; register them so
# their plain `assert`s get pytest's detailed failure messages.
pytest.register_assert_rewrite("tests.conformance.embedder", "tests.conformance.llm")
