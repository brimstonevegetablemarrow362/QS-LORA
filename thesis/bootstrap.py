"""Put domain_v1 on sys.path so thesis code can reuse generator prompts and vLLM helpers."""

from __future__ import annotations

import sys
from pathlib import Path

_DOMAIN_V1 = Path(__file__).resolve().parent.parent / "domain_v1"
if str(_DOMAIN_V1) not in sys.path:
    sys.path.insert(0, str(_DOMAIN_V1))
