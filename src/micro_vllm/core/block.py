"""Logical and physical block identifiers, tables, and pool state."""

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

BlockHash = bytes
BLOCK_HASH_BYTES = 32


@dataclass(slots=True)
class KVCacheBlock:
    """Mutable lifecycle metadata for one physical KV-cache block."""

    block_id: int
    ref_count: int = 0
    block_hash: BlockHash | None = None


class BlockPool:
    """Own physical block state and deterministic free-block ordering.

    External mappings use integer IDs in ``[0, num_blocks)``. The pool keeps
    each block's mutable refcount and hash together. Cached zero-reference
    blocks remain reclaimable in the free queue until reused.
    """

    __slots__ = ("_blocks", "_cached_blocks", "_free_block_ids")

    def __init__(self, num_blocks: int) -> None:
        if num_blocks <= 0:
            raise ValueError("num_blocks must be greater than zero")

        self._blocks = [KVCacheBlock(block_id) for block_id in range(num_blocks)]
        self._cached_blocks: dict[BlockHash, KVCacheBlock] = {}
        self._free_block_ids = deque(range(num_blocks))

    @property
    def num_blocks(self) -> int:
        """Total number of physical blocks owned by the pool."""

        return len(self._blocks)

    @property
    def num_free_blocks(self) -> int:
        """Number of physical blocks that can currently be allocated."""

        return len(self._free_block_ids)

    @property
    def free_block_ids(self) -> tuple[int, ...]:
        """Immutable snapshot of free block IDs in allocation order."""

        return tuple(self._free_block_ids)

    def ref_count(self, block_id: int) -> int:
        """Return the current reference count for a physical block."""

        ref_count = self._require_block(block_id).ref_count
        if ref_count < 0:
            raise RuntimeError(
                f"block pool invariant violated for block_id {block_id}: "
                f"negative refcount {ref_count}"
            )
        return ref_count

    def block_hash(self, block_id: int) -> BlockHash | None:
        """Return the content hash currently associated with a physical block."""

        return self._require_block(block_id).block_hash

    @property
    def num_cached_blocks(self) -> int:
        """Number of physical blocks indexed by a prefix hash."""

        return len(self._cached_blocks)

    def cache_block(
        self, block_id: int, block_hash: BlockHash
    ) -> KVCacheBlock:
        """Index one allocated block by hash without adding a reference."""

        block = self._require_block(block_id)
        if len(block_hash) != BLOCK_HASH_BYTES:
            raise ValueError(
                f"block_hash must contain {BLOCK_HASH_BYTES} bytes"
            )

        if block.ref_count == 0:
            raise RuntimeError(
                f"cannot cache block_id {block_id}: block is free"
            )
        if block.block_hash is not None and block.block_hash != block_hash:
            raise RuntimeError(
                f"cannot cache block_id {block_id}: block has a different hash"
            )

        existing = self._cached_blocks.get(block_hash)
        if existing is not None and existing is not block:
            raise RuntimeError(
                f"block hash is already cached by block_id {existing.block_id}"
            )

        block.block_hash = block_hash
        self._cached_blocks[block_hash] = block
        return block

    def get_cached_block(self, block_hash: BlockHash) -> KVCacheBlock | None:
        """Return the physical block indexed by ``block_hash``, if present."""

        return self._cached_blocks.get(block_hash)

    def acquire_cached_block(self, block_hash: BlockHash) -> KVCacheBlock | None:
        """Acquire one request reference to a cached physical block."""

        block = self._cached_blocks.get(block_hash)
        if block is None:
            return None
        if block.block_hash != block_hash:
            raise RuntimeError(
                f"block pool invariant violated for block_id {block.block_id}: "
                "cached hash does not match block"
            )

        ref_count = self.ref_count(block.block_id)
        if ref_count == 0:
            try:
                self._free_block_ids.remove(block.block_id)
            except ValueError:
                raise RuntimeError(
                    f"block pool invariant violated for block_id "
                    f"{block.block_id}: cached free block is missing from queue"
                ) from None
            block.ref_count = 1
        else:
            block.ref_count = ref_count + 1
        return block

    def evict_cached_block(self, block_hash: BlockHash) -> KVCacheBlock | None:
        """Remove one hash mapping without changing request references."""

        block = self._cached_blocks.get(block_hash)
        if block is None:
            return None
        if block.block_hash != block_hash:
            raise RuntimeError(
                f"block pool invariant violated for block_id {block.block_id}: "
                "cached hash does not match block"
            )
        del self._cached_blocks[block_hash]
        block.block_hash = None
        return block

    def allocate(
        self,
        count: int,
        cached_block_ids: Iterable[int] = (),
    ) -> tuple[int, ...] | None:
        """Atomically retain cached blocks and allocate ``count`` new blocks."""

        if count < 0:
            raise ValueError(f"count must be non-negative, got {count}")

        cached_blocks = [self._blocks[block_id] for block_id in cached_block_ids]
        num_cached_free_blocks = sum(
            block.ref_count == 0 for block in cached_blocks
        )
        if count + num_cached_free_blocks > self.num_free_blocks:
            return None

        for block in cached_blocks:
            if block.ref_count == 0:
                self._free_block_ids.remove(block.block_id)
            block.ref_count += 1

        new_block_ids: list[int] = []
        for _ in range(count):
            block_id = self._free_block_ids.popleft()
            block = self._blocks[block_id]
            if block.block_hash is not None:
                self._evict_block_hash(block)
            block.ref_count = 1
            new_block_ids.append(block_id)

        return tuple(new_block_ids)

    def retain(self, block_id: int) -> None:
        """Add one reference to an allocated physical block."""

        ref_count = self.ref_count(block_id)
        if ref_count == 0:
            raise RuntimeError(f"cannot retain block_id {block_id}: block is free")

        self._blocks[block_id].ref_count = ref_count + 1

    def release(self, block_id: int) -> None:
        """Release one reference and recycle the block when it reaches zero."""

        ref_count = self.ref_count(block_id)
        if ref_count == 0:
            raise RuntimeError(f"cannot release block_id {block_id}: block is free")

        block = self._blocks[block_id]
        block.ref_count = ref_count - 1
        if block.ref_count == 0:
            self._free_block_ids.append(block_id)

    def _require_block(self, block_id: int) -> KVCacheBlock:
        if block_id < 0 or block_id >= self.num_blocks:
            raise ValueError(
                f"invalid block_id {block_id}: "
                f"expected 0 <= block_id < {self.num_blocks}"
            )
        return self._blocks[block_id]

    def _evict_block_hash(self, block: KVCacheBlock) -> None:
        block_hash = block.block_hash
        if block_hash is None:
            return
        cached_block = self._cached_blocks.get(block_hash)
        if cached_block is not block:
            raise RuntimeError(
                f"block pool invariant violated for block_id {block.block_id}: "
                "cached hash does not map to block"
            )
        del self._cached_blocks[block_hash]
        block.block_hash = None


class BlockTable:
    """Maintain zero-based logical-to-physical block mappings.

    Each list position is a logical block index. Its value is a physical block
    ID, or ``None`` when that logical block has no physical mapping. The table
    neither validates IDs against a BlockPool nor changes block refcounts.
    """

    __slots__ = ("_block_size", "_physical_block_ids")

    def __init__(self, block_size: int) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be greater than zero")

        self._block_size = block_size
        self._physical_block_ids: list[int | None] = []

    @property
    def block_size(self) -> int:
        """Number of token slots represented by each logical block."""

        return self._block_size

    @property
    def num_logical_blocks(self) -> int:
        """Number of logical block entries currently in the table."""

        return len(self._physical_block_ids)

    @property
    def capacity_tokens(self) -> int:
        """Number of token positions covered by the current mappings."""

        return self.num_logical_blocks * self._block_size

    def append(self, physical_block_id: int | None) -> None:
        """Append one physical mapping at the next logical block index."""

        self._validate_mapping(physical_block_id)
        self._physical_block_ids.append(physical_block_id)

    def extend(self, physical_block_ids: Iterable[int | None]) -> None:
        """Atomically append multiple physical mappings in input order."""

        mappings = tuple(physical_block_ids)
        for physical_block_id in mappings:
            self._validate_mapping(physical_block_id)
        self._physical_block_ids.extend(mappings)

    def physical_block(self, logical_block_index: int) -> int | None:
        """Return a physical mapping, or ``None`` when it is unmapped."""

        self._validate_logical_block_index(logical_block_index)
        return self._physical_block_ids[logical_block_index]

    def replace(
        self,
        logical_block_index: int,
        physical_block_id: int | None,
    ) -> int | None:
        """Replace one mapping and return its previous physical block ID."""

        self._validate_logical_block_index(logical_block_index)
        self._validate_mapping(physical_block_id)
        previous_block_id = self._physical_block_ids[logical_block_index]
        self._physical_block_ids[logical_block_index] = physical_block_id
        return previous_block_id

    def token_slot(self, token_position: int) -> tuple[int, int] | None:
        """Map a token position to ``(physical_block_id, block_offset)``."""

        if token_position < 0:
            raise ValueError("token_position must be non-negative")

        logical_block_index, block_offset = divmod(
            token_position, self._block_size
        )
        if logical_block_index >= len(self._physical_block_ids):
            raise ValueError(
                f"token_position {token_position} is outside table capacity "
                f"{len(self._physical_block_ids) * self._block_size}"
            )

        physical_block_id = self._physical_block_ids[logical_block_index]
        if physical_block_id is None:
            return None
        return physical_block_id, block_offset

    def truncate(self, num_tokens: int) -> tuple[int, ...]:
        """Remove mappings beyond ``num_tokens`` and return mapped block IDs."""

        if num_tokens < 0:
            raise ValueError("num_tokens must be non-negative")
        if num_tokens > self.capacity_tokens:
            raise ValueError(
                f"num_tokens {num_tokens} exceeds table capacity "
                f"{self.capacity_tokens}"
            )

        num_blocks_to_keep = (
            num_tokens + self._block_size - 1
        ) // self._block_size
        removed_mappings = self._physical_block_ids[num_blocks_to_keep:]
        del self._physical_block_ids[num_blocks_to_keep:]
        return tuple(
            physical_block_id
            for physical_block_id in removed_mappings
            if physical_block_id is not None
        )

    def clear(self) -> tuple[int, ...]:
        """Remove all mappings and return their non-empty physical block IDs."""

        return self.truncate(0)

    def snapshot(self) -> tuple[int | None, ...]:
        """Return an immutable snapshot of logical-to-physical mappings."""

        return tuple(self._physical_block_ids)

    @staticmethod
    def _validate_mapping(physical_block_id: int | None) -> None:
        if physical_block_id is not None and physical_block_id < 0:
            raise ValueError("physical_block_id must be non-negative or None")

    def _validate_logical_block_index(self, logical_block_index: int) -> None:
        if (
            logical_block_index < 0
            or logical_block_index >= len(self._physical_block_ids)
        ):
            raise ValueError(
                f"invalid logical_block_index {logical_block_index}: expected "
                f"0 <= logical_block_index < {len(self._physical_block_ids)}"
            )
