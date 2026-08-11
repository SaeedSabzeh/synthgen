"""synthgen — schema-validated synthetic tabular data from an LLM."""

from synthgen.config import MissingAPIKeyError, Settings
from synthgen.generator import GenerationResult, GenerationStats, SyntheticDataGenerator
from synthgen.schemas import SCHEMAS, Person, get_schema, schema_from_spec

__all__ = [
    "SyntheticDataGenerator",
    "GenerationResult",
    "GenerationStats",
    "Person",
    "SCHEMAS",
    "get_schema",
    "schema_from_spec",
    "Settings",
    "MissingAPIKeyError",
]
__version__ = "0.2.0"
