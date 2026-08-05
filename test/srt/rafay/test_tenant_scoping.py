"""Rafay: tenant scoping in the KV cache hash chain.

These are the unit-level half of the tenant-isolation contract
(``kubeless-me/docs/model-engine-engine-contract.md``). They run without a GPU
or a live engine, so they gate every PR; the Tier 3a conformance suite proves
the same invariant end to end, per backend.

**Why these exist as regression tests rather than as coverage.** The patches
they guard are five lines in a function upstream rarely touches — which is
exactly the kind of change a merge silently drops. If that happens, requests
keep succeeding and two tenants quietly share a cache. Nothing fails. These
tests are the thing that fails.
"""

import hashlib
import importlib.util
import pathlib
import unittest

# Loaded by path rather than by package import: ``sglang/__init__`` pulls numpy,
# torch and triton, and the point of these tests is that they run anywhere,
# every PR. The module under test depends only on the standard library.
_MODULE = (
    pathlib.Path(__file__).resolve().parents[3]
    / "python/sglang/srt/mem_cache/tenant_scope.py"
)
_spec = importlib.util.spec_from_file_location("tenant_scope", _MODULE)
_tenant_scope = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tenant_scope)
tenant_scope_seed = _tenant_scope.tenant_scope_seed


class TestTenantScopeSeed(unittest.TestCase):
    def test_different_tenants_seed_differently(self):
        """The whole point: two tenants must not share a chain root."""
        self.assertNotEqual(tenant_scope_seed("acme"), tenant_scope_seed("globex"))

    def test_same_tenant_seeds_identically(self):
        """Otherwise a tenant never hits its own cache."""
        self.assertEqual(tenant_scope_seed("acme"), tenant_scope_seed("acme"))

    def test_unset_scope_preserves_upstream_behaviour(self):
        """Single-tenant deployments must be bit-identical to upstream.

        ``compute_node_hash_values`` seeds only when this returns non-None, so
        ``None`` here is what guarantees no behaviour change when the feature is
        off.
        """
        self.assertIsNone(tenant_scope_seed(None))
        self.assertIsNone(tenant_scope_seed(""))

    def test_seed_is_a_32_byte_digest(self):
        """``get_hash_str`` does ``bytes.fromhex(prior_hash)`` and the native
        module expects a SHA-256 digest. A shorter seed would fail at the C++
        boundary, not here."""
        seed = tenant_scope_seed("acme")
        self.assertEqual(len(bytes.fromhex(seed)), 32)

    def test_seed_is_sha256_of_the_scope(self):
        """Pinned so the derivation cannot drift: a change here invalidates every
        stored L2/L3 entry, which must be a deliberate, staged decision rather
        than a side effect."""
        self.assertEqual(
            tenant_scope_seed("acme"),
            hashlib.sha256(b"acme").hexdigest(),
        )

    def test_scopes_that_concatenate_alike_do_not_collide(self):
        """A tenant must not reach another's chain by choosing its own scope
        string."""
        self.assertNotEqual(tenant_scope_seed("ab"), tenant_scope_seed("a") + "b")
        self.assertNotEqual(tenant_scope_seed("acme:prod"), tenant_scope_seed("acme:pro"))


class TestBlockRemovedCarriesScope(unittest.TestCase):
    """R4 — an eviction must be attributable, or a floor cannot be proven.

    Asserted against the source rather than by importing the event module, which
    pulls the full runtime. The Tier 3a conformance suite exercises the real
    object; this only guards the field from being dropped by a merge.
    """

    def test_block_removed_declares_extra_key(self):
        src = (
            pathlib.Path(__file__).resolve().parents[3]
            / "python/sglang/srt/disaggregation/kv_events.py"
        ).read_text()
        block_removed = src[src.index("class BlockRemoved") : src.index("class AllBlocksCleared")]
        self.assertIn("extra_key", block_removed)
        # Optional, so single-tenant payloads are unchanged.
        self.assertIn("extra_key: Optional[str] = None", block_removed)

    def test_eviction_populates_the_scope(self):
        src = (
            pathlib.Path(__file__).resolve().parents[3]
            / "python/sglang/srt/mem_cache/events.py"
        ).read_text()
        self.assertIn('extra_key=getattr(node.key, "extra_key", None)', src)


if __name__ == "__main__":
    unittest.main()


class TestEmbeddingRequestCarriesScope(unittest.TestCase):
    """The embedding path must reach the radix cache with a scope.

    **Why embeddings are not exempt.** An embedding returns no tokens and is
    prefill-only, so it reads like a request that cannot pollute a KV cache.
    It can. ``Scheduler.handle_embedding_request`` builds an ordinary ``Req``
    and calls ``_maybe_namespace_elastic_radix_cache`` on it, the same as the
    generate path — so two tenants embedding identical text share radix
    entries unless ``extra_key`` is threaded through.

    These read source rather than run the scheduler: importing
    ``sglang.srt.managers.scheduler`` pulls torch, and the regression being
    guarded is structural — a merge dropping one keyword argument. The Tier 3a
    conformance suite is what proves the behaviour on a live engine.
    """

    _ROOT = pathlib.Path(__file__).resolve().parents[3] / "python/sglang/srt/managers"

    def _function(self, filename, name):
        import ast

        tree = ast.parse((self._ROOT / filename).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return node
        self.fail(f"{name} not found in {filename} — did upstream rename it?")

    def test_tokenized_embedding_input_declares_extra_key(self):
        import ast

        tree = ast.parse((self._ROOT / "io_struct.py").read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ClassDef)
                and node.name == "TokenizedEmbeddingReqInput"
            ):
                fields = [
                    stmt.target.id
                    for stmt in node.body
                    if isinstance(stmt, ast.AnnAssign)
                    and isinstance(stmt.target, ast.Name)
                ]
                self.assertIn(
                    "extra_key",
                    fields,
                    "TokenizedEmbeddingReqInput lost extra_key; the embedding "
                    "path is now unscoped and tenants share cache entries.",
                )
                return
        self.fail("TokenizedEmbeddingReqInput not found in io_struct.py")

    def test_embedding_handler_forwards_extra_key_to_req(self):
        import ast

        handler = self._function("scheduler.py", "handle_embedding_request")
        forwarded = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "Req"
            and any(kw.arg == "extra_key" for kw in node.keywords)
            for node in ast.walk(handler)
        )
        self.assertTrue(
            forwarded,
            "handle_embedding_request builds Req(...) without extra_key. The "
            "Req still goes through the radix cache, so this silently shares "
            "embedding cache entries across tenants — no error is raised.",
        )

    def test_generate_handler_still_forwards_extra_key(self):
        """The reference the embedding assertion is modelled on.

        If upstream restructures the generate path this fails first, which
        says the embedding assertion above is testing a shape that no longer
        describes how requests are built.
        """
        import ast

        handler = self._function("scheduler.py", "handle_generate_request")
        self.assertTrue(
            any(
                isinstance(node, ast.keyword) and node.arg == "extra_key"
                for node in ast.walk(handler)
            ),
            "handle_generate_request no longer passes extra_key.",
        )
