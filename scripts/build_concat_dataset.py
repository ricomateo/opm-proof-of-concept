"""Concatenate Norne and Volve baseline into a single CSV.

Reads `datasets/dataset_norne.csv` and `datasets/dataset_volve.csv`, tags
each row with a `reservoir_id` column, writes
`datasets/dataset_volve_norne.csv`. sim_ids stay independent per
reservoir; the compound key (reservoir_id, sim_id) uniquely identifies
each simulation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NORNE_PATH = PROJECT_ROOT / "datasets" / "dataset_norne.csv"
VOLVE_PATH = PROJECT_ROOT / "datasets" / "dataset_volve.csv"
OUT_PATH = PROJECT_ROOT / "datasets" / "dataset_volve_norne.csv"


def main() -> int:
    norne = pd.read_csv(NORNE_PATH)
    volve = pd.read_csv(VOLVE_PATH)

    norne["reservoir_id"] = "norne"
    volve["reservoir_id"] = "volve"

    combined = pd.concat([norne, volve], ignore_index=True)

    cols = list(combined.columns)
    cols.remove("reservoir_id")
    cols.insert(cols.index("sim_id") + 1, "reservoir_id")
    combined = combined[cols]

    _validate(combined, expected_rows=len(norne) + len(volve))

    combined.to_csv(OUT_PATH, index=False)

    n_norne = (combined["reservoir_id"] == "norne").sum()
    n_volve = (combined["reservoir_id"] == "volve").sum()
    sims_norne = combined.loc[combined["reservoir_id"] == "norne", "sim_id"].nunique()
    sims_volve = combined.loc[combined["reservoir_id"] == "volve", "sim_id"].nunique()

    print(f"wrote {OUT_PATH}")
    print(f"  total: {len(combined)} rows, {len(combined.columns)} cols")
    print(f"  norne: {n_norne} rows across {sims_norne} sims")
    print(f"  volve: {n_volve} rows across {sims_volve} sims")
    return 0


def _validate(df: pd.DataFrame, expected_rows: int) -> None:
    assert len(df) == expected_rows, f"row count {len(df)} != expected {expected_rows}"
    assert df["reservoir_id"].nunique() == 2, (
        f"expected 2 distinct reservoir_id values, got {df['reservoir_id'].nunique()}"
    )
    assert not df.isnull().any().any(), "NaN values present in concatenated dataset"

    for (rid, sid), group in df.groupby(["reservoir_id", "sim_id"]):
        diffs = group["tiempo_dias"].diff().dropna()
        if not (diffs >= 0).all():
            raise AssertionError(f"tiempo_dias decreases within {rid} sim {sid}")


if __name__ == "__main__":
    sys.exit(main())
