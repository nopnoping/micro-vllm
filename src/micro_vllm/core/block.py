"""Logical and physical block identifiers, tables, and pool state."""

from collections import deque
from itertools import islice


class BlockPool:
    """Own physical block refcounts and deterministic free-block ordering.

    Physical blocks are represented only by integer IDs in ``[0, num_blocks)``.
    The pool does not know about requests, tokens, or device tensors.
    """

    __slots__ = ("_free_block_ids", "_num_blocks", "_ref_counts")

    def __init__(self, num_blocks: int) -> None:
        if num_blocks <= 0:
            raise ValueError("num_blocks must be greater than zero")

        self._num_blocks = num_blocks
        self._free_block_ids = deque(range(num_blocks))
        self._ref_counts = [0] * num_blocks

    @property
    def num_blocks(self) -> int:
        """Total number of physical blocks owned by the pool."""

        return self._num_blocks

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

        self._validate_block_id(block_id)
        ref_count = self._ref_counts[block_id]
        if ref_count < 0:
            raise RuntimeError(
                f"block pool invariant violated for block_id {block_id}: "
                f"negative refcount {ref_count}"
            )
        return ref_count

    def allocate(self, count: int) -> tuple[int, ...] | None:
        """Atomically allocate ``count`` blocks, or return ``None`` if full."""

        if count <= 0:
            raise ValueError(f"count must be greater than zero, got {count}")
        if count > self.num_free_blocks:
            return None

        block_ids = tuple(islice(self._free_block_ids, count))
        self._validate_free_candidates(block_ids)

        for block_id in block_ids:
            self._free_block_ids.popleft()
            self._ref_counts[block_id] = 1

        return block_ids

    def retain(self, block_id: int) -> None:
        """Add one reference to an allocated physical block."""

        ref_count = self.ref_count(block_id)
        if ref_count == 0:
            raise RuntimeError(f"cannot retain block_id {block_id}: block is free")

        self._ref_counts[block_id] = ref_count + 1

    def release(self, block_id: int) -> None:
        """Release one reference and recycle the block when it reaches zero."""

        ref_count = self.ref_count(block_id)
        if ref_count == 0:
            raise RuntimeError(f"cannot release block_id {block_id}: block is free")

        self._ref_counts[block_id] = ref_count - 1
        if self._ref_counts[block_id] == 0:
            self._free_block_ids.append(block_id)

    def _validate_block_id(self, block_id: int) -> None:
        if block_id < 0 or block_id >= self._num_blocks:
            raise ValueError(
                f"invalid block_id {block_id}: "
                f"expected 0 <= block_id < {self._num_blocks}"
            )

    def _validate_free_candidates(self, block_ids: tuple[int, ...]) -> None:
        if len(set(block_ids)) != len(block_ids):
            raise RuntimeError(
                f"free queue contains duplicate block IDs: {block_ids}"
            )

        for block_id in block_ids:
            if block_id < 0 or block_id >= self._num_blocks:
                raise RuntimeError(
                    f"free queue contains invalid block_id {block_id}: "
                    f"expected 0 <= block_id < {self._num_blocks}"
                )
            if self._ref_counts[block_id] != 0:
                raise RuntimeError(
                    f"block pool invariant violated for block_id {block_id}: "
                    f"free block has refcount {self._ref_counts[block_id]}"
                )
