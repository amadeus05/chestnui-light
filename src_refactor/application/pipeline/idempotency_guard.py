from __future__ import annotations

from dataclasses import dataclass, field


class IdempotencyGuard:
    def record_once(self, key: str) -> bool:
        raise NotImplementedError


@dataclass(slots=True)
class InMemoryIdempotencyGuard(IdempotencyGuard):
    seen_keys: set[str] = field(default_factory=set)

    def record_once(self, key: str) -> bool:
        if key in self.seen_keys:
            return False
        self.seen_keys.add(key)
        return True
