"""Rafay: storage-path tenant scoping (contract R2).

The KV hash chain is seeded with the tenant at the radix root, but the
**storage** path is separate: `cache_controller` hashes raw token ids with no
RadixKey, so `extra_key` was not merely unused there — it was unavailable. The
hashes it produced are the keys HiCacheStorage writes under, so two tenants
sending identical tokens collided in host memory and on disk even while the
radix tree kept them apart in GPU memory.

These assert the seeding, without importing torch or the scheduler.
"""

import hashlib
import importlib.util
import pathlib
import unittest

_ROOT = pathlib.Path(__file__).resolve().parents[3]


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, _ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tenant_scope = _load("tenant_scope", "python/sglang/srt/mem_cache/tenant_scope.py")


class TestStorageOperationSeeding(unittest.TestCase):
    """The seeding rule, asserted against the source.

    cache_controller imports torch, so the class cannot be constructed here.
    What must hold is the *rule*: seed only when starting a fresh sequence, and
    only when a scope exists.
    """

    def setUp(self):
        self.src = (_ROOT / "python/sglang/srt/managers/cache_controller.py").read_text()

    def test_storage_operation_accepts_a_scope(self):
        self.assertIn("extra_key: Optional[str] = None", self.src)
        self.assertIn("self.extra_key = extra_key", self.src)

    def test_seed_only_when_no_parent_hash(self):
        """A mid-sequence operation must chain from its parent, not reseed.

        Reseeding partway would break the chain and silently miss every block
        after that point — a cache that looks healthy and never hits.
        """
        self.assertIn("if last_hash is None and extra_key:", self.src)

    def test_prefetch_forwards_the_scope(self):
        """The entry point callers use must pass it through, or the parameter
        exists and does nothing."""
        self.assertIn(
            "request_id, new_input_tokens, last_hash, prefix_keys, extra_key=extra_key",
            self.src,
        )

    def test_prefetch_operation_forwards_to_super(self):
        self.assertIn("None, token_ids, last_hash, prefix_keys=prefix_keys, extra_key=extra_key", self.src)


class TestSeedProperties(unittest.TestCase):
    """The seed the storage path uses must be the same one the radix path uses,
    or a block stored under one is unreachable under the other."""

    def test_storage_and_radix_seed_identically(self):
        self.assertEqual(
            tenant_scope.tenant_scope_seed("acme"),
            hashlib.sha256(b"acme").hexdigest(),
        )

    def test_absent_scope_leaves_the_chain_unseeded(self):
        """Single-tenant behaviour must be byte-identical to upstream: no scope
        means last_hash stays None and the hash is what it always was."""
        self.assertIsNone(tenant_scope.tenant_scope_seed(None))
        self.assertIsNone(tenant_scope.tenant_scope_seed(""))

    def test_tenants_do_not_share_a_storage_chain(self):
        self.assertNotEqual(
            tenant_scope.tenant_scope_seed("acme"),
            tenant_scope.tenant_scope_seed("globex"),
        )


if __name__ == "__main__":
    unittest.main()
