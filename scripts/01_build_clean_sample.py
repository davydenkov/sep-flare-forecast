"""Build the cleaned event sample from the upstream table and NOAA catalogues (offline)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sepforecast import sample  # noqa: E402


def main() -> None:
    events, spe, qc = sample.build(ROOT)
    out = ROOT / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    events.to_csv(out / "sep_m2_events_clean.csv", index=False)
    spe.to_csv(out / "noaa_spe_1997_2022_with_parent_flare.csv", index=False)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "qc_summary.json").write_text(json.dumps(qc, indent=2, default=str), encoding="utf-8")
    print(json.dumps(qc, indent=2, default=str))


if __name__ == "__main__":
    main()
