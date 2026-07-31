#!/usr/bin/env python3
"""Download real, publicly released DoD issuances into data/raw/.

Run this from a network with access to esd.whs.mil / media.defense.gov
(the dev sandbox may block these hosts). All documents are public releases.
"""

import sys
import urllib.request
from pathlib import Path

CORPUS = {
    # DoDD 3000.09 — Autonomy in Weapon Systems
    "dodd_3000_09.pdf": "https://www.esd.whs.mil/Portals/54/Documents/DD/issuances/dodd/300009p.pdf",
    # DoDI 5000.89 — Test and Evaluation
    "dodi_5000_89.pdf": "https://www.esd.whs.mil/Portals/54/Documents/DD/issuances/dodi/500089p.pdf",
    # DoDI 5200.48 — Controlled Unclassified Information
    "dodi_5200_48.pdf": "https://www.esd.whs.mil/Portals/54/Documents/DD/issuances/dodi/520048p.pdf",
    # DoD Data, Analytics, and AI Adoption Strategy (2023)
    "dod_daai_strategy_2023.pdf": "https://media.defense.gov/2023/Nov/02/2003333300/-1/-1/1/DOD_DATA_ANALYTICS_AI_ADOPTION_STRATEGY.PDF",
}

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    failures = 0
    for name, url in CORPUS.items():
        dest = RAW / name
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            data = urllib.request.urlopen(req, timeout=60).read()
            if len(data) < 10_000:
                raise ValueError(f"response too small ({len(data)} bytes) — likely blocked")
            dest.write_bytes(data)
            print(f"OK  {name} ({len(data):,} bytes)")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {exc}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
