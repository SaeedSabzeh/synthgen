# synthgen

[![CI](https://github.com/SaeedSabzeh/synthgen/actions/workflows/ci.yml/badge.svg)](https://github.com/SaeedSabzeh/synthgen/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Generate synthetic tabular datasets with an LLM, where **every row is validated against a
Pydantic schema before it reaches your file**. Batched, concurrent, de-duplicated, and it
tells you afterwards whether the data is actually varied.

```bash
synthgen --schema person --rows 500 --out out/people.csv --report
```

```
wrote 500 records to out/people.csv
{"requested": 500, "returned": 500, "invalid": 14, "duplicates": 63, "api_calls": 24, "yield_rate": 0.867}

field                 unique   uniq%  top value
------------------------------------------------------------------
full_name                500   100%  Amara Okafor (1)
date_of_birth            496    99%  1987-03-12 (2)
city                     212    42%  Milan (9)
country                   38     8%  Italy (41)
occupation                87    17%  Teacher (23)
```

## Why not just ask the model for 500 rows?

Because you get maybe 40 usable ones. Four failure modes this handles:

| Problem | What happens without it | What synthgen does |
| --- | --- | --- |
| **Response limits** | Ask for 500, get 30 silently truncated, or malformed JSON | splits into batches, runs them concurrently, tops up until the count is met |
| **Mode collapse** | The same twenty names, three cities, birthdays all in the 80s | each batch is steered along rotating region / decade / register axes, plus an avoid-list of what's already been generated |
| **Silent bad rows** | `"date of birth": "DD/MM/YY"` lands in your CSV as a literal string | every row validated against Pydantic; loose date formats normalised; invalid rows counted and dropped |
| **No visibility** | You ship 500 rows that are really 40 repeated | `--report` gives duplicate rate and per-field cardinality, and flags near-constant columns |

## Quickstart

```bash
git clone https://github.com/SaeedSabzeh/synthgen.git
cd synthgen
python -m venv .venv && source .venv/bin/activate
pip install -e ".[pandas,dev]"

cp .env.example .env      # paste your OpenAI key
synthgen --list-schemas
synthgen --schema person --rows 50 --out out/people.csv --report

make test                 # 45 tests, no network, no key needed
```

### As a library

```python
from openai import OpenAI
from synthgen import SyntheticDataGenerator, Person

gen = SyntheticDataGenerator(OpenAI(), Person, batch_size=25, concurrency=4)
result = gen.generate(200, extra_instructions="All based in northern Italy.")

print(result.stats.as_dict())     # yield rate, invalid, duplicates, api_calls
df = result.to_dataframe()        # or result.to_dicts()
```

## Built-in schemas

| Name | Fields |
| --- | --- |
| `person` | full_name, date_of_birth, email, street_address, city, country, occupation |
| `transaction` | transaction_id, customer_name, amount, currency, category, timestamp, status |
| `ticket` | ticket_id, subject, body, product_area, sentiment, priority |

### Your own schema, no Python

```json
{
  "name": "Product",
  "dedup_keys": ["sku"],
  "fields": {
    "sku":      {"type": "string"},
    "price_eur":{"type": "number", "gt": 0.99, "lt": 2000},
    "category": {"type": "string", "enum": ["kitchen", "outdoor", "office"]},
    "in_stock": {"type": "boolean"}
  }
}
```

```bash
synthgen --schema-file examples/product_schema.json --rows 100 --out out/products.jsonl
```

Constraints (`gt`, `lt`, `min_length`, `enum`, …) become real validation, and they're sent
to the model as JSON Schema so it aims at the right shape in the first place.

Or subclass `BaseModel` directly and pass it in — `dedup_keys` as a `ClassVar` controls what
counts as a duplicate.

## How it works

```
generate(n)
   │
   ├─ split into ceil(n / batch_size) batches
   │
   ├─ ThreadPoolExecutor(concurrency)
   │     each batch: system prompt (JSON Schema + rules)
   │                 user prompt  (count + rotating diversity hint + avoid-list)
   │                 response_format = json_object
   │                 retry with exponential backoff
   │
   ├─ extract_records()  — tolerant of whichever wrapper key the model picked
   ├─ schema.model_validate() per row — invalid rows counted, dropped
   ├─ dedup on the schema's dedup_keys
   │
   └─ short? top-up round (up to max_topup_rounds), then return with stats
```

| Module | Responsibility |
| --- | --- |
| `schemas.py` | Pydantic models, loose-date coercion, runtime schema from a JSON spec |
| `prompts.py` | system/user prompts, diversity hints, tolerant response parsing |
| `generator.py` | batching, concurrency, retries, validation, dedup, top-up, stats |
| `quality.py` | duplicate rate, per-field cardinality, near-constant warnings |
| `exporters.py` | csv / jsonl / json / parquet, chosen by file extension |
| `cli.py` | `synthgen` command |

## CLI reference

```
synthgen --schema person --rows 200 --out people.csv
         --schema-file spec.json      custom schema
         --instructions "..."         extra steering, e.g. "all based in Italy"
         --batch-size 25              records per API call
         --concurrency 4              parallel calls
         --temperature 1.0            higher = more varied
         --seed 42                    reproducible batch hints
         --report                     data-quality report
         --list-schemas
```

Exit code `3` means fewer rows came back than requested — useful in a pipeline.

## Design notes

The table above covers the four failure modes. A few decisions behind the fixes:

**The schema is the source of truth.** It validates the output *and* generates the
prompt — `field_summary` and `model_json_schema` are fed to the model, so the shape it
aims at and the shape it's judged against can't drift apart. Adding a field is a
one-line change.

**Liberal in, strict out.** `extract_records` accepts whatever wrapper key the model
invented (`records`, `data`, `people`, or a lone unnamed list) because that choice is
arbitrary and not worth failing over. Individual rows get no such leniency:
`extra="forbid"`, real constraints, and a plausibility check on dates.

**Duplicates are defined by the schema.** `dedup_keys` as a `ClassVar` — two people
share a city, but not a name and birthday. Falling back to whole-row equality would
miss near-duplicates entirely, which is the shape mode collapse actually takes.

**Failures are data, not exceptions.** A batch that dies after its retries is recorded
in `result.errors` and the run continues; you get 480 usable rows and a note, not a
traceback and nothing. `GenerationStats` reports yield rate so you can see the run's
real cost.

**The client is injected.** Which is why 45 tests covering batch splitting, dedup,
invalid rows, top-up rounds and retried failures run in 0.17s without a key.

## Testing approach

The client is injected, so `tests/conftest.py` swaps in a fake that returns scripted JSON.
That makes the interesting paths deterministic and free: batch splitting, dedup, invalid-row
handling, top-up rounds, retried failures.

```bash
pytest -q
pytest --cov=synthgen
```

## Caveats

- LLM-generated data inherits the model's biases — name and location distributions are not
  representative of any real population. Fine for testing and demos; not for training a model
  you'll draw conclusions from, and not a substitute for a differential-privacy pipeline.
- Records are invented, but a model can still emit a real person's name by chance. Don't treat
  the output as anonymised real data.
- Cost scales with rows. Start with `--rows 20` and check the report before scaling up.

## Roadmap

- [ ] Structured Outputs (`response_format={"type": "json_schema", "strict": true}`) to cut the invalid-row rate
- [ ] Hybrid mode: Faker for names/addresses, LLM for the free-text fields (cheaper, more diverse)
- [ ] Referential integrity across linked schemas (customers ↔ transactions)
- [ ] Distribution targets ("60% negative sentiment")
- [ ] Async client and streaming to disk for very large runs

## License

MIT — see [LICENSE](LICENSE).
