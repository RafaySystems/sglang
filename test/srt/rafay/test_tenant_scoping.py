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
