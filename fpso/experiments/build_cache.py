"""One-time CRSP pull into a Parquet cache.

Run this once. Every backtest thereafter reads the cache, which is what makes
the study (a) fast enough to run 7 arms x 30 seeds, and (b) reproducible — WRDS
query results can change under you, so a cached panel is the only way a reviewer
re-running the code gets the numbers in the paper.

    python -m fpso.experiments.build_cache                 # full CRSP pull
    python -m fpso.experiments.build_cache --source synthetic   # offline demo

The WRDS pull needs network access to wrds-pgdata.wharton.upenn.edu:9737 and
credentials in .env (see .env.example).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fpso.config.schema import DataConfig
from fpso.data.source import SyntheticPanelSource, WRDSPanelSource


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["wrds", "synthetic"], default="wrds")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--start", default="2005-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument(
        "--universe-pool",
        type=int,
        default=500,
        help="Names sampled per year for the cached universe union.",
    )
    args = parser.parse_args()

    config = DataConfig(
        source=args.source,
        cache_dir=args.cache_dir,
        panel_start=args.start,
        panel_end=args.end,
    )

    print(f"Building {args.source} panel {args.start} -> {args.end}", flush=True)
    if args.source == "wrds":
        panel = WRDSPanelSource(config, universe_pool=args.universe_pool).load()
    else:
        panel = SyntheticPanelSource(args.start, args.end).load()

    panel.to_parquet(args.cache_dir)
    print(
        f"Wrote {panel.returns.shape[0]} trading days x {panel.returns.shape[1]} "
        f"assets to {Path(args.cache_dir).resolve()}"
    )
    print(f"  date range : {panel.dates[0].date()} .. {panel.dates[-1].date()}")
    print(f"  coverage   : {panel.returns.notna().mean().mean():.1%} of cells populated")


if __name__ == "__main__":
    main()
