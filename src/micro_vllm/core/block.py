"""Logical and physical block identifiers, tables, and pool state."""

from collections import deque
from collections.abc import Iterable
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

        if (
            logical_block_index < 0
            or logical_block_index >= len(self._physical_block_ids)
        ):
            raise ValueError(
                f"invalid logical_block_index {logical_block_index}: expected "
                f"0 <= logical_block_index < {len(self._physical_block_ids)}"
            )
        return self._physical_block_ids[logical_block_index]

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
