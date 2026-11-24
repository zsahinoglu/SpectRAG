# ADR 0003: Module moves and deprecation shims

- Status: accepted (ARCH1)
- Date: 2026-11-24

## Context

Code moves from `rag_system/` and `config.py` into `spectrag/` (ADR 0001) while builders keep
changing those same files (C1 edits `config.py` defaults; H1a pins behavior with golden tests).
Users, scripts, notebooks and pickled indexes refer to the old module paths.

## Decision

1. **One move per submission.** A move relocates a module to its `spectrag` home and leaves the
   old file as a re-export shim (`from spectrag.x import A, B`; same `__all__`). Moved classes
   keep their identity, so `old.Chunk is new.Chunk`.
2. **Files under active builder work are re-exported, not moved.** `spectrag.settings`
   re-exports the dataclasses from `config.py`; the definitions move once C1 and the other
   config-touching items have merged. The `spectrag.settings` import path is already stable.
3. **Shims stay silent while code in this repository still uses the old path.** When the last
   internal caller has been migrated (ARCH2 for the modules moved so far), the shim gains a
   module-level `DeprecationWarning` naming the new path. Warning earlier would only flag our
   own code.
4. **Shims are removed no earlier than v1.0** and only after at least one release in which they
   warned. ARCH3 lists the removal release in `CHANGELOG.md`.
5. **Pickled indexes**: indexes written before a move reference old class paths, which the shims
   keep loadable until H3 replaces pickle. Once H3 lands, nothing on disk depends on Python
   module paths.
6. `tests/api/test_shims.py` pins every shim (identity of the re-exported objects).

ARCH1 applies this to `rag_system/models.py` → `spectrag/models.py` (moved) and
`config.py` → `spectrag/settings.py` (re-exported).

## Consequences

- No caller has to change in ARCH1; old and new imports give the same objects.
- Indexes pickled *after* this change record `spectrag.models.Chunk`, so a checkout from before
  ARCH1 cannot read them (downgrade only; upgrade is unaffected).
- Moves can be scheduled into batches that avoid files other builders are editing.
