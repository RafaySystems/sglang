"""Rafay: what key strings actually reach Mooncake.

`docs/model-engine-backend-keying.md` concludes Mooncake is tenant-scoped
because it addresses storage by the key strings HiCache hands it, and
`_tag_keys` only prefixes them. **That conclusion was reached by reading code**,
which is the same method that produced the wrong answer about LMCache's NIXL
presence cache. This asserts it instead.

What is under test is `MooncakeStore.batch_exists` and `exists` themselves --
the real methods, with a stub standing in only for the `MooncakeDistributedStore`
at the far end, so the test can record precisely which strings would have gone
over the wire. Nothing here needs a GPU, a Mooncake deployment, or the mooncake
Python package.

## What this proves, and what it does not

It proves the store is asked about **disjoint keys** for two tenants, through
every suffixing branch the method has (MLA, MHA, split-heads), and that
`_tag_keys` cannot collapse a distinction that was present in its input.

It does not prove tenant isolation. That needs the keys reaching here to differ
in the first place, which is the hash chain's job and is covered separately, and
it needs the store to honour the distinction, which is Mooncake's own behaviour
and belongs to Tier 3a. This closes the middle link: the one place our code
could silently discard the scope between the engine and the backend.
"""

import unittest
from typing import List, Optional
from unittest import mock

# Two tenants' key sets for the same content. In production these come out of
# the chain-seeded hash, so they are unequal by construction; here they are
# written literally so a failure points at the transport rather than at hashing.
ALICE_KEYS = ["aaaa1111", "aaaa2222"]
BOB_KEYS = ["bbbb1111", "bbbb2222"]


class _RecordingStore:
    """Stands in for MooncakeDistributedStore, recording what it was asked."""

    def __init__(self, present: Optional[set] = None):
        self.present = present or set()
        self.queried: List[str] = []

    def batch_is_exist(self, key_strs: List[str]) -> List[int]:
        self.queried.extend(key_strs)
        return [1 if k in self.present else 0 for k in key_strs]


def _store(
    *,
    is_mla_backend: bool = False,
    should_split_heads: bool = False,
    extra_backend_tag: Optional[str] = None,
    present: Optional[set] = None,
):
    """A MooncakeStore with only the attributes the key path reads.

    Constructed without `__init__` deliberately: the real constructor connects
    to a Mooncake cluster, and connecting is not what is under test. Every
    method that runs below is the real one.
    """
    from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
        MooncakeStore,
    )

    store = MooncakeStore.__new__(MooncakeStore)
    store.store = _RecordingStore(present=present)
    store.extra_backend_tag = extra_backend_tag
    store.is_mla_backend = is_mla_backend
    store.should_split_heads = should_split_heads
    store.mla_suffix = "mla"
    store.mha_suffix = "mha"
    store.split_factor = 2
    if should_split_heads:
        store.mha_suffix = ["mha0", "mha1"]
    return store


try:  # pragma: no cover - import guard, not logic
    _store()
    _AVAILABLE, _WHY = True, ""
except Exception as exc:  # noqa: BLE001 - reported, not handled
    _AVAILABLE, _WHY = False, f"{type(exc).__name__}: {exc}"


@unittest.skipUnless(_AVAILABLE, f"MooncakeStore not importable -- {_WHY}")
class TestMooncakeReceivesDisjointKeys(unittest.TestCase):
    """Two tenants, identical content, must never address the same object."""

    def _queried_for(self, keys, **kwargs):
        store = _store(**kwargs)
        store.batch_exists(keys)
        return store.store.queried

    def test_mha_keys_are_disjoint(self):
        alice = self._queried_for(ALICE_KEYS)
        bob = self._queried_for(BOB_KEYS)
        self.assertTrue(alice and bob)
        self.assertEqual(set(alice) & set(bob), set())

    def test_mla_keys_are_disjoint(self):
        """MLA takes a different suffixing branch, so it is asserted
        separately -- a branch that dropped the key would be invisible to the
        MHA case."""
        alice = self._queried_for(ALICE_KEYS, is_mla_backend=True)
        bob = self._queried_for(BOB_KEYS, is_mla_backend=True)
        self.assertTrue(alice and bob)
        self.assertEqual(set(alice) & set(bob), set())

    def test_split_head_keys_are_disjoint(self):
        alice = self._queried_for(ALICE_KEYS, should_split_heads=True)
        bob = self._queried_for(BOB_KEYS, should_split_heads=True)
        self.assertTrue(alice and bob)
        self.assertEqual(set(alice) & set(bob), set())

    def test_every_queried_key_contains_the_key_it_came_from(self):
        """Suffixing must *decorate* the key, not replace it. A branch that
        derived its query from anything other than the key it was given is how
        a backend stops being scoped -- which is exactly what LMCache's NIXL
        presence cache does."""
        for kwargs in ({}, {"is_mla_backend": True}, {"should_split_heads": True}):
            with self.subTest(**kwargs):
                queried = self._queried_for(ALICE_KEYS, **kwargs)
                for q in queried:
                    self.assertTrue(
                        any(k in q for k in ALICE_KEYS),
                        f"query {q!r} is not derived from any input key",
                    )


@unittest.skipUnless(_AVAILABLE, f"MooncakeStore not importable -- {_WHY}")
class TestTagKeysPreservesDistinctness(unittest.TestCase):
    """`_tag_keys` is the only transform applied to keys before use."""

    def test_prefix_cannot_merge_distinct_keys(self):
        store = _store(extra_backend_tag="cluster-a")
        tagged = store._tag_keys(ALICE_KEYS + BOB_KEYS)
        self.assertEqual(len(set(tagged)), len(ALICE_KEYS + BOB_KEYS))

    def test_absent_tag_is_identity(self):
        """No `extra_backend_tag` configured must leave keys untouched, or a
        single-tenant deployment's existing objects become unreachable."""
        store = _store(extra_backend_tag=None)
        self.assertEqual(store._tag_keys(ALICE_KEYS), ALICE_KEYS)

    def test_tag_is_not_a_tenant_boundary(self):
        """`extra_backend_tag` comes from static `extra_config`, so it is one
        value for the whole engine -- it separates *clusters sharing a store*,
        not tenants. Asserted so nobody later mistakes it for isolation and
        stops seeding the hash chain."""
        store = _store(extra_backend_tag="cluster-a")
        both = store._tag_keys([ALICE_KEYS[0], BOB_KEYS[0]])
        self.assertTrue(all(k.startswith("cluster-a") for k in both))


@unittest.skipUnless(_AVAILABLE, f"MooncakeStore not importable -- {_WHY}")
class TestOneTenantCannotSeeAnother(unittest.TestCase):
    """The attack, rather than the mechanism."""

    def test_bobs_stored_content_is_a_miss_for_alice(self):
        """Bob's objects are present in the store; Alice asks about the same
        content. `batch_exists` returns the number of leading pages found, so a
        leak shows up as a non-zero answer."""
        bob = _store()
        bob.batch_exists(BOB_KEYS)
        bob_objects = set(bob.store.queried)

        alice = _store(present=bob_objects)
        self.assertEqual(alice.batch_exists(ALICE_KEYS), 0)

    def test_a_tenant_does_find_its_own_content(self):
        """The other half, and not a formality: a scoping change that broke all
        reuse would pass every test above."""
        alice = _store()
        alice.batch_exists(ALICE_KEYS)
        own_objects = set(alice.store.queried)

        again = _store(present=own_objects)
        self.assertEqual(again.batch_exists(ALICE_KEYS), len(ALICE_KEYS))

    def test_exists_agrees_with_batch_exists(self):
        bob = _store()
        bob.exists(BOB_KEYS[0])
        bob_objects = set(bob.store.queried)

        alice = _store(present=bob_objects)
        self.assertFalse(alice.exists(ALICE_KEYS[0]))


if __name__ == "__main__":
    unittest.main()
