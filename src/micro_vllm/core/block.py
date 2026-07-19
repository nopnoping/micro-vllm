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

        return self._blocks[block_id].ref_count

    def block_hash(self, block_id: int) -> BlockHash | None:
        """Return the content hash currently associated with a physical block."""

        return self._blocks[block_id].block_hash

    @property
    def num_cached_blocks(self) -> int:
        """Number of physical blocks indexed by a prefix hash."""

        return len(self._cached_blocks)

    def cache_block(
        self, block_id: int, block_hash: BlockHash
    ) -> KVCacheBlock:
        """Index one allocated block by hash without adding a reference."""

        block = self._blocks[block_id]
        self._cached_blocks.pop(block.block_hash, None)
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

        if block.ref_count == 0:
            self._free_block_ids.remove(block.block_id)
        block.ref_count += 1
        return block

    def evict_cached_block(self, block_hash: BlockHash) -> KVCacheBlock | None:
        """Remove one hash mapping without changing request references."""

        block = self._cached_blocks.pop(block_hash, None)
        if block is None:
            return None
        block.block_hash = None
        return block

    def allocate(
        self,
        count: int,
        cached_block_ids: Iterable[int] = (),
    ) -> tuple[int, ...] | None:
        """Atomically retain cached blocks and allocate ``count`` new blocks."""

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
                self._cached_blocks.pop(block.block_hash, None)
                block.block_hash = None
            block.ref_count = 1
            new_block_ids.append(block_id)

        return tuple(new_block_ids)

    def retain(self, block_id: int) -> None:
        """Add one reference to an allocated physical block."""

        self._blocks[block_id].ref_count += 1

    def release(self, block_id: int) -> None:
        """Release one reference and recycle the block when it reaches zero."""

        block = self._blocks[block_id]
        block.ref_count -= 1
        if block.ref_count == 0:
            self._free_block_ids.append(block_id)


class BlockTable:
    """Maintain zero-based logical-to-physical block mappings.

    Each list position is a logical block index. Its value is a physical block
    ID, or ``None`` when that logical block has no physical mapping. The table
    neither validates IDs against a BlockPool nor changes block refcounts.
    """

    __slots__ = ("_block_size", "_physical_block_ids")

    def __init__(self, block_size: int) -> None:
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

        self._physical_block_ids.append(physical_block_id)

    def extend(self, physical_block_ids: Iterable[int | None]) -> None:
        """Atomically append multiple physical mappings in input order."""

        self._physical_block_ids.extend(physical_block_ids)

    def physical_block(self, logical_block_index: int) -> int | None:
        """Return a physical mapping, or ``None`` when it is unmapped."""

        return self._physical_block_ids[logical_block_index]

    def replace(
        self,
        logical_block_index: int,
        physical_block_id: int | None,
    ) -> int | None:
        """Replace one mapping and return its previous physical block ID."""

        previous_block_id = self._physical_block_ids[logical_block_index]
        self._physical_block_ids[logical_block_index] = physical_block_id
        return previous_block_id

    def token_slot(self, token_position: int) -> tuple[int, int] | None:
        """Map a token position to ``(physical_block_id, block_offset)``."""

        if token_position < 0 or token_position >= self.capacity_tokens:
            return None
        logical_block_index, block_offset = divmod(
            token_position, self._block_size
        )
        physical_block_id = self._physical_block_ids[logical_block_index]
        if physical_block_id is None:
            return None
        return physical_block_id, block_offset

    def truncate(self, num_tokens: int) -> tuple[int, ...]:
        """Remove mappings beyond ``num_tokens`` and return mapped block IDs."""

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
