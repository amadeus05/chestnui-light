from __future__ import annotations

from dataclasses import dataclass

from src_refactor.core.types import MarketExecutionSnapshot, OrderRequest
from src_refactor.domain.signals import SignalCandidate


@dataclass(frozen=True, slots=True)
class OrderFactory:
    def from_signal_candidate(
        self,
        candidate: SignalCandidate,
        *,
        position_notional: float,
        stop_pct: float,
        take_pct: float,
        snapshot: MarketExecutionSnapshot,
    ) -> OrderRequest:
        side = "buy" if candidate.direction == 1 else "sell"
        quantity = position_notional / snapshot.next_open
        return OrderRequest(
            order_id=f"{candidate.symbol}:{snapshot.next_timestamp.value}:{side}",
            symbol=candidate.symbol,
            side=side,
            order_type="market",
            quantity=quantity,
            stop_pct=stop_pct,
            take_pct=take_pct,
            created_at=snapshot.current_timestamp,
        )
