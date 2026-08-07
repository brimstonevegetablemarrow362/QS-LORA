"""Self-contained ingest, Q/A generation, split, train, and merge scripts for platform Docker images."""

from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
PLATFORM_ROOT = PIPELINE_DIR.parent
