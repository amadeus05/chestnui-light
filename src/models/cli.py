from __future__ import annotations

import argparse
from typing import Sequence

from src.models.factory import create_runner
from src.models.registry import get_model_spec, list_model_specs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="model.py",
        description="Unified model runner. Extra arguments after model are passed to the selected legacy runner.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List registered models.")
    list_parser.set_defaults(handler=handle_list)

    for command in ("train", "production", "wfv", "backtest"):
        command_parser = subparsers.add_parser(command, help=f"Run {command} for a registered model.")
        command_parser.add_argument("model", choices=[spec.key for spec in list_model_specs()])
        command_parser.add_argument(
            "model_args",
            nargs=argparse.REMAINDER,
            help="Arguments forwarded to the model runner. Use '--' before forwarded flags if needed.",
        )
        command_parser.set_defaults(handler=handle_run)

    return parser


def normalize_forwarded_args(args: Sequence[str]) -> list[str]:
    forwarded = list(args)
    if forwarded and forwarded[0] == "--":
        return forwarded[1:]
    return forwarded


def handle_list(_: argparse.Namespace) -> None:
    for spec in list_model_specs():
        production = "yes" if spec.supports_production else "no"
        wfv = "yes" if spec.supports_walk_forward else "no"
        replay = "yes" if spec.supports_backtest_replay else "no"
        print(
            f"{spec.key}: {spec.display_name} | "
            f"type={spec.model_type} | source={spec.feature_source} | "
            f"production={production} | wfv={wfv} | replay={replay}"
        )


def handle_run(args: argparse.Namespace) -> None:
    spec = get_model_spec(args.model)
    runner = create_runner(spec)
    forwarded = normalize_forwarded_args(args.model_args)

    if args.command == "train":
        runner.run_train(forwarded)
    elif args.command == "production":
        runner.run_production(forwarded)
    elif args.command == "wfv":
        runner.run_walk_forward(forwarded)
    elif args.command == "backtest":
        runner.run_backtest(forwarded)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.handler(args)
