"""Deterministic identities and state for full-block prefix caching."""

from collections.abc import Iterable
from hashlib import blake2b

from micro_vllm.core.block import BLOCK_HASH_BYTES, BlockHash

SINGLE_CACHE_GROUP_ID = 0
_HASH_DOMAIN = b"micro-vllm-kv"
_MAX_UINT64 = (1 << 64) - 1


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

    if block_size <= 0:
        raise ValueError("block_size must be greater than zero")
    if not cache_namespace:
        raise ValueError("cache_namespace must not be empty")
    if cache_group_id != SINGLE_CACHE_GROUP_ID:
        raise ValueError(
            f"only cache_group_id {SINGLE_CACHE_GROUP_ID} is supported, "
            f"got {cache_group_id}"
        )
    if parent_block_hash is not None and len(parent_block_hash) != BLOCK_HASH_BYTES:
        raise ValueError(
            f"parent_block_hash must contain {BLOCK_HASH_BYTES} bytes"
        )

    token_block = tuple(token_ids)
    if len(token_block) != block_size:
        raise ValueError(
            f"token block must contain exactly {block_size} tokens, "
            f"got {len(token_block)}"
        )
    for index, token_id in enumerate(token_block):
        if token_id < 0 or token_id > _MAX_UINT64:
            raise ValueError(
                f"token_ids[{index}] must fit an unsigned 64-bit integer"
            )

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
