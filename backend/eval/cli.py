"""CLI entry point for pipeline evaluation."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from app.core.config import get_settings

from eval.report import print_report, report_to_json
from eval.runner import (
    DEFAULT_DATASET,
    bootstrap_eval_env,
    load_dataset,
    run_dataset,
    shutdown_eval_env,
)

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval",
        description="Run AiCrateDigger pipeline edge-case evaluation against a JSON dataset.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Path to edge-case JSON (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--case",
        action="append",
        dest="case_ids",
        metavar="ID",
        help="Run only these case id(s); repeatable",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "parse", "full"],
        default="all",
        help="Filter cases by mode (default: all)",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write machine-readable report to this path",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Allow Redis cache hits (default: bypass cache for reproducible stage traces)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List case ids and exit",
    )
    return parser


async def _async_main(args: argparse.Namespace) -> int:
    dataset_path: Path = args.dataset
    if not dataset_path.is_file():
        logger.error("dataset_not_found", extra={"path": str(dataset_path)})
        return 2

    dataset = load_dataset(dataset_path)

    if args.list:
        for case in dataset.cases:
            print(f"{case.id}\t{case.mode}\t{case.query[:60]}")
        return 0

    selected = dataset.cases
    if args.case_ids:
        selected = [c for c in selected if c.id in set(args.case_ids)]
    if args.mode != "all":
        selected = [c for c in selected if c.mode == args.mode]

    needs_live = any(c.mode == "full" for c in selected)
    settings = get_settings()
    if needs_live:
        missing: list[str] = []
        if not (settings.openai_api_key or "").strip():
            missing.append("OPENAI_API_KEY")
        if not (settings.tavily_api_key or "").strip():
            missing.append("TAVILY_API_KEY")
        if missing:
            logger.error(
                "eval_missing_api_keys",
                extra={"required": missing},
            )
            print(
                f"Full pipeline cases require: {', '.join(missing)}",
                file=sys.stderr,
            )
            return 2

    await bootstrap_eval_env()
    try:
        report = await run_dataset(
            dataset,
            case_ids=set(args.case_ids) if args.case_ids else None,
            mode_filter=args.mode,
            bypass_cache=not args.use_cache,
        )
    finally:
        await shutdown_eval_env()

    print_report(report)

    if args.json_out is not None:
        args.json_out.write_text(report_to_json(report), encoding="utf-8")
        print(f"Wrote JSON report to {args.json_out}", file=sys.stderr)

    return 1 if report.failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return asyncio.run(_async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
