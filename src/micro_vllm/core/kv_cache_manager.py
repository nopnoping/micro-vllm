"""Logical KV-cache allocation, sharing, preemption, and release."""

from dataclasses import dataclass

from micro_vllm.config import CacheConfig
from micro_vllm.core.block import BlockPool, BlockTable


@dataclass(frozen=True, slots=True)
class KVCacheAllocation:
    """Immutable result of reserving logical KV-cache slots for one request."""

    block_table: tuple[int | None, ...]
    new_block_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        block_table = tuple(self.block_table)
        new_block_ids = tuple(self.new_block_ids)

        for block_id in block_table:
            if block_id is not None and block_id < 0:
                raise ValueError("block_table IDs must be non-negative or None")
        for block_id in new_block_ids:
            if block_id < 0:
                raise ValueError("new_block_ids must be non-negative")

        object.__setattr__(self, "block_table", block_table)
        object.__setattr__(self, "new_block_ids", new_block_ids)


class KVCacheManager:
    """Own logical KV-cache state for one full-attention cache group.

    The manager owns the physical BlockPool and one BlockTable per registered
    request. Request objects and physical device tensors remain outside it.
    """

    __slots__ = ("_block_pool", "_block_size", "_block_tables")

    def __init__(self, config: CacheConfig) -> None:
        self._block_pool = BlockPool(config.num_blocks)
        self._block_size = config.block_size
        self._block_tables: dict[str, BlockTable] = {}

    def register_request(self, request_id: str) -> None:
        """Create an empty BlockTable owned by ``request_id``."""

        if not request_id:
            raise ValueError("request_id must not be empty")
        if request_id in self._block_tables:
            raise ValueError(f"duplicate request_id: {request_id}")

        self._block_tables[request_id] = BlockTable(self._block_size)

    def get_block_table(self, request_id: str) -> tuple[int | None, ...]:
        """Return an immutable snapshot for a registered request."""

        return self._require_block_table(request_id).snapshot()

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_computed_tokens: int,
        num_scheduled_tokens: int,
    ) -> int:
        """Return missing blocks for computed plus currently scheduled tokens."""

        if num_computed_tokens < 0:
            raise ValueError("num_computed_tokens must be non-negative")
        if num_scheduled_tokens < 0:
            raise ValueError("num_scheduled_tokens must be non-negative")

        block_table = self._require_block_table(request_id)
        target_num_tokens = num_computed_tokens + num_scheduled_tokens
        required_num_blocks = (
            target_num_tokens + self._block_size - 1
        ) // self._block_size
        return max(required_num_blocks - block_table.num_logical_blocks, 0)

    def allocate_slots(
        self,
        request_id: str,
        num_computed_tokens: int,
        num_scheduled_tokens: int,
    ) -> KVCacheAllocation | None:
        """Atomically reserve slots for the currently scheduled token range."""

        num_new_blocks = self.get_num_blocks_to_allocate(
            request_id,
            num_computed_tokens,
            num_scheduled_tokens,
        )
        block_table = self._require_block_table(request_id)
        if num_new_blocks == 0:
            return KVCacheAllocation(
                block_table=block_table.snapshot(),
                new_block_ids=(),
            )

        new_block_ids = self._block_pool.allocate(num_new_blocks)
        if new_block_ids is None:
            return None

        block_table.extend(new_block_ids)
        return KVCacheAllocation(
            block_table=block_table.snapshot(),
            new_block_ids=new_block_ids,
        )

    def free_request(self, request_id: str) -> None:
        """Release all request block references and remove its owned table.

        Raises ``KeyError`` when the request is not registered, including when
        it has already been freed.
        """

        block_table = self._require_block_table(request_id)
        for block_id in block_table.clear():
            self._block_pool.release(block_id)
        del self._block_tables[request_id]

    def _require_block_table(self, request_id: str) -> BlockTable:
        try:
            return self._block_tables[request_id]
        except KeyError:
            raise KeyError(f"unknown request_id: {request_id}") from None
