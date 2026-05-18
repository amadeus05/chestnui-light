from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

from src_refactor.core.types import AccountSnapshot, Fill, OrderRequest, OrderSnapshot

ExecutionJournalEventType = Literal[
    "ACCOUNT_SNAPSHOT",
    "ORDER_SUBMITTED",
    "ORDER_ACCEPTED",
    "ORDER_REJECTED",
    "FILL",
    "TRADE_CLOSED",
]


@dataclass(frozen=True, slots=True)
class ExecutionJournalEvent:
    sequence: int
    event_type: ExecutionJournalEventType
    timestamp: pd.Timestamp | None = None
    account: AccountSnapshot | None = None
    order: OrderRequest | OrderSnapshot | None = None
    fill: Fill | None = None
    trade: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExecutionJournal:
    events: list[ExecutionJournalEvent] = field(default_factory=list)
    _next_sequence: int = 1

    def record_account_snapshot(
        self,
        account: AccountSnapshot,
        *,
        timestamp: pd.Timestamp | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ExecutionJournalEvent:
        return self._append(
            "ACCOUNT_SNAPSHOT",
            timestamp=timestamp,
            account=account,
            metadata=metadata,
        )

    def record_order_submitted(
        self,
        order: OrderRequest,
        *,
        timestamp: pd.Timestamp | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ExecutionJournalEvent:
        return self._append(
            "ORDER_SUBMITTED",
            timestamp=timestamp or order.created_at,
            order=order,
            metadata=metadata,
        )

    def record_order_accepted(
        self,
        order: OrderSnapshot,
        *,
        timestamp: pd.Timestamp | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ExecutionJournalEvent:
        return self._append(
            "ORDER_ACCEPTED",
            timestamp=timestamp or order.created_at,
            order=order,
            metadata=metadata,
        )

    def record_order_rejected(
        self,
        order: OrderSnapshot,
        *,
        timestamp: pd.Timestamp | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ExecutionJournalEvent:
        return self._append(
            "ORDER_REJECTED",
            timestamp=timestamp or order.created_at,
            order=order,
            metadata=metadata,
        )

    def record_fill(
        self,
        fill: Fill,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> ExecutionJournalEvent:
        return self._append(
            "FILL",
            timestamp=fill.timestamp,
            fill=fill,
            metadata=metadata,
        )

    def record_trade_closed(
        self,
        trade: Any,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> ExecutionJournalEvent:
        timestamp = getattr(trade, "closed_at", None)
        return self._append(
            "TRADE_CLOSED",
            timestamp=timestamp,
            trade=trade,
            metadata=metadata,
        )

    def clear(self) -> None:
        self.events.clear()
        self._next_sequence = 1

    def _append(
        self,
        event_type: ExecutionJournalEventType,
        *,
        timestamp: pd.Timestamp | None = None,
        account: AccountSnapshot | None = None,
        order: OrderRequest | OrderSnapshot | None = None,
        fill: Fill | None = None,
        trade: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ExecutionJournalEvent:
        event = ExecutionJournalEvent(
            sequence=self._next_sequence,
            event_type=event_type,
            timestamp=pd.to_datetime(timestamp) if timestamp is not None else None,
            account=account,
            order=order,
            fill=fill,
            trade=trade,
            metadata=dict(metadata or {}),
        )
        self._next_sequence += 1
        self.events.append(event)
        return event
