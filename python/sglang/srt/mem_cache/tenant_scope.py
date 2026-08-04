# Rafay addition — see kubeless-me/docs/model-engine-engine-contract.md
"""Tenant scoping for the KV cache hash chain.

Kept in its own module for two reasons. It has **no dependencies beyond the
standard library**, so the tests that guard it run on every PR without a GPU or
the triton/torch stack that ``mem_cache.utils`` pulls in. And a separate file
conflicts far less than an edit inside a large upstream file, which matters for a
patch carried across a fast-moving upstream.
"""

import hashlib
from typing import Optional


def tenant_scope_seed(extra_key: Optional[str]) -> Optional[str]:
    """Derive a hash-chain seed from a ``RadixKey.extra_key``.

    ``extra_key`` (lora id, cache salt, tenant scope) participates in **radix
    tree** matching but never reaches the hash: ``_native_hash_input`` reads only
    the token ids. Every layer keyed by that hash is therefore tenant-blind — the
    KV lifecycle events, and every ``HiCacheStorage`` backend, which keys on
    these strings.

    Seeding the chain fixes those at once. ``get_hash_str`` already takes a
    ``prior_hash`` and the hash is a chain, so seeding at the root makes every
    descendant inherit the scope without touching the native hash module.

    Returns ``None`` for an unset key, which reproduces the previous behaviour
    exactly — single-tenant deployments are bit-identical to upstream.

    **Changing this derivation invalidates every stored L2/L3 entry**, so it is
    pinned by test and any change is a staged rollout, not a refactor.
    """
    if not extra_key:
        return None
    return hashlib.sha256(extra_key.encode("utf-8")).hexdigest()
