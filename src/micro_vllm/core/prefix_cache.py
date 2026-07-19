"""Deterministic identities and state for full-block prefix caching."""

from collections.abc import Iterable
from hashlib import blake2b

from micro_vllm.core.block import BLOCK_HASH_BYTES, BlockHash

SINGLE_CACHE_GROUP_ID = 0
_HASH_DOMAIN = b"micro-vllm-kv"


def hash_block(
    parent_block_hash: BlockHash | None,
    token_ids: Iterable[int],
    *,
    block_size: int,
    cache_namespace: str,
    cache_group_id: int = SINGLE_CACHE_GROUP_ID,
) -> BlockHash:
    """Hash one complete token block into a deterministic prefix identity.

    ``cache_namespace`` must identify the model/cache compatibility domain.
    The parent digest chains the full prefix into every child block hash.
    """

    token_block = tuple(token_ids)
    namespace_bytes = cache_namespace.encode("utf-8")
    digest = blake2b(digest_size=BLOCK_HASH_BYTES, person=_HASH_DOMAIN)
    digest.update(len(namespace_bytes).to_bytes(8, "big"))
    digest.update(namespace_bytes)
    digest.update(cache_group_id.to_bytes(8, "big"))
    digest.update(block_size.to_bytes(8, "big"))
    if parent_block_hash is None:
        digest.update(b"\x00")
    else:
        digest.update(b"\x01")
        digest.update(parent_block_hash)
    for token_id in token_block:
        digest.update(token_id.to_bytes(8, "big"))
    return digest.digest()
