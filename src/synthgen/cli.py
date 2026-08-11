"""Command line interface.

    synthgen --schema person --rows 200 --out people.csv --report
    synthgen --schema-file examples/product_schema.json --rows 50 --out products.jsonl
    synthgen --list-schemas
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from synthgen.config import MissingAPIKeyError, Settings
from synthgen.exporters import export
from synthgen.generator import SyntheticDataGenerator
from synthgen.quality import report
from synthgen.schemas import SCHEMAS, get_schema, schema_from_spec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="synthgen",
        description="Generate schema-validated synthetic datasets with an LLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-s", "--schema", default="person", help=f"one of: {', '.join(sorted(SCHEMAS))}"
    )
    parser.add_argument("--schema-file", help="JSON spec for a custom schema (overrides --schema)")
    parser.add_argument("-n", "--rows", type=int, default=50, help="number of records")
    parser.add_argument(
        "-o", "--out", help="output path (.csv/.jsonl/.json/.parquet); omit to print"
    )
    parser.add_argument("--model", help="override the model")
    parser.add_argument("--batch-size", type=int, help="records per API call")
    parser.add_argument("--concurrency", type=int, help="parallel API calls")
    parser.add_argument("--temperature", type=float, help="higher means more varied")
    parser.add_argument(
        "--seed", type=int, help="seed for the batch-hint RNG (prompts, not the model)"
    )
    parser.add_argument(
        "--instructions", default="", help="extra steering, e.g. 'all based in Italy'"
    )
    parser.add_argument("--report", action="store_true", help="print a data-quality report")
    parser.add_argument(
        "--list-schemas", action="store_true", help="show built-in schemas and exit"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.list_schemas:
        for name, model in sorted(SCHEMAS.items()):
            print(f"{name:<14}{', '.join(model.model_fields)}")
        return 0

    try:
        schema = schema_from_spec(args.schema_file) if args.schema_file else get_schema(args.schema)
    except (KeyError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        settings = Settings.from_env()
    except MissingAPIKeyError as exc:
        print(exc, file=sys.stderr)
        return 1

    from openai import OpenAI

    generator = SyntheticDataGenerator(
        client=OpenAI(api_key=settings.api_key, timeout=settings.request_timeout),
        schema=schema,
        model=args.model or settings.model,
        batch_size=args.batch_size or settings.batch_size,
        concurrency=args.concurrency or settings.concurrency,
        max_retries=settings.max_retries,
        temperature=settings.temperature if args.temperature is None else args.temperature,
        seed=args.seed,
    )

    result = generator.generate(args.rows, extra_instructions=args.instructions)
    rows = result.to_dicts()

    if args.out:
        path = export(rows, args.out)
        print(f"wrote {len(rows)} records to {path}")
    else:
        print(json.dumps(rows, ensure_ascii=False, indent=2))

    print(json.dumps(result.stats.as_dict(), indent=2), file=sys.stderr)
    for error in result.errors:
        print(f"batch error: {error}", file=sys.stderr)

    if args.report:
        print("\n" + report(rows).render(), file=sys.stderr)

    return 0 if len(rows) == args.rows else 3


if __name__ == "__main__":
    raise SystemExit(main())
