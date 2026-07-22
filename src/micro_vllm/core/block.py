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

    External mappings use integer IDs in ``[0, num_blocks)``. The final ID is
    permanently reserved as a shared null block. The remaining blocks keep
    mutable refcounts and hashes; cached zero-reference blocks remain
    reclaimable in the free queue until reused.
    """

    __slots__ = (
        "_blocks",
        "_cached_blocks",
        "_free_block_ids",
        "_null_block_id",
    )

    def __init__(self, num_blocks: int) -> None:
        self._blocks = [KVCacheBlock(block_id) for block_id in range(num_blocks)]
        self._cached_blocks: dict[BlockHash, KVCacheBlock] = {}
        self._null_block_id = num_blocks - 1
        self._free_block_ids = deque(range(self._null_block_id))

    @property
    def num_blocks(self) -> int:
        """Total number of physical blocks owned by the pool."""

        return len(self._blocks)

    @property
    def num_free_blocks(self) -> int:
        """Number of physical blocks that can currently be allocated."""

        return len(self._free_block_ids)

    @property
    def null_block_id(self) -> int:
        """Physical block permanently reserved for null table entries."""

        return self._null_block_id

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

    def can_allocate(
        self,
        count: int,
        cached_block_ids: Iterable[int] = (),
    ) -> bool:
        """Whether one allocation fits without changing pool state."""

        num_cached_free_blocks = sum(
            self._blocks[block_id].ref_count == 0
            for block_id in cached_block_ids
        )
        return count + num_cached_free_blocks <= self.num_free_blocks

    def allocate(
        self,
        count: int,
        cached_block_ids: Iterable[int] = (),
    ) -> tuple[int, ...] | None:
        """Atomically retain cached blocks and allocate ``count`` new blocks."""

        cached_ids = tuple(cached_block_ids)
        if not self.can_allocate(count, cached_ids):
            return None

        cached_blocks = [self._blocks[block_id] for block_id in cached_ids]
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

    def release(self, block_id: int) -> None:
        """Release one reference and recycle the block when it reaches zero."""

        block = self._blocks[block_id]
        block.ref_count -= 1
        if block.ref_count == 0:
            self._free_block_ids.append(block_id)


class BlockTable:
    """Maintain zero-based logical-to-physical block mappings.

    Each list position is a logical block index and always contains a physical
    block ID. Unmapped positions point at the shared null block. The table
    neither validates IDs against a BlockPool nor changes block refcounts.
    """

    __slots__ = ("_null_block_id", "_physical_block_ids")

    def __init__(self, null_block_id: int) -> None:
        self._null_block_id = null_block_id
        self._physical_block_ids: list[int] = []

    @property
    def num_logical_blocks(self) -> int:
        """Number of logical block entries currently in the table."""

        return len(self._physical_block_ids)

    def extend(self, physical_block_ids: Iterable[int]) -> None:
        """Atomically append multiple physical mappings in input order."""

        self._physical_block_ids.extend(physical_block_ids)

    def physical_block(self, logical_block_index: int) -> int:
        """Return the physical mapping at one absolute logical index."""

        return self._physical_block_ids[logical_block_index]

    def replace(
        self,
        logical_block_index: int,
        physical_block_id: int,
    ) -> int:
        """Replace one mapping and return its previous physical block ID."""

        previous_block_id = self._physical_block_ids[logical_block_index]
        self._physical_block_ids[logical_block_index] = physical_block_id
        return previous_block_id

    def fill_missing(
        self,
        logical_block_indices: Iterable[int],
        physical_block_ids: Iterable[int],
    ) -> None:
        """Fill absolute logical indices while preserving sparse table holes."""

        for logical_block_index, physical_block_id in zip(
            logical_block_indices,
            physical_block_ids,
        ):
            num_missing_entries = (
                logical_block_index + 1 - self.num_logical_blocks
            )
            if num_missing_entries > 0:
                self._physical_block_ids.extend(
                    [self._null_block_id] * num_missing_entries
                )
            self._physical_block_ids[logical_block_index] = physical_block_id

    def release_before(
        self,
        first_retained_logical_block: int,
    ) -> tuple[int, ...]:
        """Unmap blocks before one absolute index and return their IDs."""

        released_block_ids: list[int] = []
        for logical_block_index in range(
            min(first_retained_logical_block, self.num_logical_blocks)
        ):
            block_id = self._physical_block_ids[logical_block_index]
            if block_id != self._null_block_id:
                released_block_ids.append(block_id)
                self._physical_block_ids[logical_block_index] = self._null_block_id
        return tuple(released_block_ids)

    def clear(self) -> tuple[int, ...]:
        """Remove all mappings and return their non-empty physical block IDs."""

        physical_block_ids = tuple(
            block_id
            for block_id in self._physical_block_ids
            if block_id != self._null_block_id
        )
        self._physical_block_ids.clear()
        return physical_block_ids

    def snapshot(self) -> tuple[int, ...]:
        """Return an immutable snapshot of logical-to-physical mappings."""

        return tuple(self._physical_block_ids)
