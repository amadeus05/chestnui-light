from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

import pandas as pd

from src_refactor.infrastructure.exchanges import BybitMarketDataClient
from src_refactor.infrastructure.market_data import MarketDataIngestionService, ParquetMarketDataStore


logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.handler(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Market data ingestion CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    backfill = subparsers.add_parser(
        "backfill",
        help="Download raw exchange market data into parquet files.",
    )
    backfill.add_argument("--exchange", choices=["bybit"], default="bybit")
    backfill.add_argument("--data-root", default="_data")
    backfill.add_argument("--symbols", nargs="+", required=True)
    backfill.add_argument("--timeframes", nargs="+", required=True)
    backfill.add_argument("--start", required=True)
    backfill.add_argument("--end", default=None)
    backfill.add_argument("--no-premium-index", dest="include_premium_index", action="store_false", default=True)
    backfill.add_argument("--no-funding", dest="include_funding", action="store_false", default=True)
    backfill.add_argument("--no-open-interest", dest="include_open_interest", action="store_false", default=True)
    backfill.set_defaults(handler=run_backfill)
    return parser


def run_backfill(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
    client = _client(args.exchange)
    store = ParquetMarketDataStore(root=Path(args.data_root), exchange_code=client.exchange_code)
    end = pd.to_datetime(args.end) if args.end else pd.Timestamp.now(tz="UTC")
    logger.info(
        "Market data backfill: exchange=%s, data_root=%s, symbols=%s, timeframes=%s, start=%s, end=%s",
        client.exchange_code,
        Path(args.data_root),
        ",".join(args.symbols),
        ",".join(args.timeframes),
        pd.to_datetime(args.start),
        end,
    )
    summary = MarketDataIngestionService(client=client, store=store).backfill_raw(
        symbols=tuple(args.symbols),
        timeframes=tuple(args.timeframes),
        start=pd.to_datetime(args.start),
        end=end,
        include_premium_index=args.include_premium_index,
        include_funding=args.include_funding,
        include_open_interest=args.include_open_interest,
    )
    print(json.dumps(summary.written_rows, indent=2, sort_keys=True))
    print(f"Total written rows: {summary.total_written_rows}")


def _client(exchange: str) -> BybitMarketDataClient:
    if exchange == "bybit":
        return BybitMarketDataClient()
    raise ValueError(f"Unsupported exchange: {exchange}")


if __name__ == "__main__":
    main()
