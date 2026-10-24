"""Report review-scraping progress from what's on disk.

Reads the shop list, the per-shop review CSVs and the review page cache
rather than the log: the log rotates (10MB x 5), so on a long run the start
of the review phase is gone from it, while the files are the actual state.
"""

import argparse
import json
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

REVIEW_KEY_COLUMNS = ["author", "product", "comment", "city", "meta_info"]
_LOG_PROGRESS_RE = re.compile(
    r"\] Shop .*(starting|reviews ->|incomplete)|found \d+ review pages"
)


def _fmt_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def report(input_file: Path, cache_dir: Path, output_dir: Path, log_file: Path) -> None:
    shops = pd.read_csv(input_file, dtype=str)["shop_id"]
    unique_shops = shops.nunique()

    done = set()
    rows = 0
    dups = 0
    for f in output_dir.glob("*.csv"):
        done.add(f.stem)
        df = pd.read_csv(f, dtype=str)
        rows += len(df)
        if len(df):
            dups += int(df.duplicated(subset=REVIEW_KEY_COLUMNS).sum())

    # shop_id -> [(page_num, found, scraped_at)]
    pages = defaultdict(list)
    cached_reviews = 0
    corrupt = []
    for f in cache_dir.rglob("*.json"):
        try:
            with open(f) as fh:
                entry = json.load(fh)
        except (OSError, json.JSONDecodeError):
            corrupt.append(f)
            continue
        pages[entry.get("shop_id")].append(
            (entry.get("page_num", 0), entry.get("found", 0), entry.get("scraped_at", 0))
        )
        cached_reviews += entry.get("found", 0)
    cached_pages = sum(len(v) for v in pages.values())

    print(f"shops in list:    {len(shops)} ({unique_shops} unique)")
    print(f"shops finished:   {len(done)} ({len(done) / max(unique_shops, 1):.1%})")
    print(
        f"reviews in CSVs:  {rows} (duplicates {dups}, {dups / max(rows, 1):.2%})"
    )
    print(f"cached pages:     {cached_pages} ({cached_reviews} reviews)")

    if corrupt:
        print(f"\nCORRUPT cache files ({len(corrupt)}), will crash the next start:")
        for f in corrupt:
            print(f"  {f}")

    in_progress = {sid: v for sid, v in pages.items() if sid not in done}
    print(f"\nshops with cache but no CSV (in progress / incomplete): {len(in_progress)}")
    for sid, v in sorted(in_progress.items(), key=lambda kv: -max(p[2] for p in kv[1])):
        print(
            f"  {sid}  pages_cached={len(v)}  max_page={max(p[0] for p in v)}  "
            f"last_page_at={_fmt_ts(max(p[2] for p in v))}"
        )

    timestamps = sorted(p[2] for v in pages.values() for p in v if p[2])
    if len(timestamps) > 1:
        hours = (timestamps[-1] - timestamps[0]) / 3600
        print(
            f"\nfirst cached page: {_fmt_ts(timestamps[0])}, "
            f"last: {_fmt_ts(timestamps[-1])} "
            f"({(time.time() - timestamps[-1]) / 60:.0f} min ago)"
        )
        if hours > 0:
            print(f"throughput:        ~{len(timestamps) / hours:.0f} pages/hour")

    if log_file.exists():
        with open(log_file, errors="replace") as fh:
            lines = fh.readlines()
        progress = [line.rstrip() for line in lines if _LOG_PROGRESS_RE.search(line)]
        print(f"\nlatest progress in {log_file}:")
        for line in progress[-5:]:
            print(f"  {line}")
        if lines:
            print(f"last log line:\n  {lines[-1].rstrip()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("./config.yaml"))
    args = parser.parse_args()

    # Same path resolution as darknet_scraper.main(): relative to the config.
    repo_root = args.config.resolve().parent

    def resolve(p: str | Path) -> Path:
        p = Path(p)
        return p if p.is_absolute() else (repo_root / p).resolve()

    with open(args.config) as f:
        reviews = yaml.safe_load(f)["scrape_reviews"]

    report(
        resolve(reviews["input_file"]),
        resolve(reviews["cache_dir"]),
        resolve(reviews["output_dir"]),
        resolve("logs/darknet-scraper.log"),
    )


if __name__ == "__main__":
    main()
