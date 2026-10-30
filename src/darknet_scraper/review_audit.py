"""Check scraped reviews for duplication the scraper itself could cause.

Read-only. Looks for the patterns a scraping bug would leave behind:
pages holding more reviews than the site's page size (a review parsed
twice), pages within one shop with identical content (the ?p= parameter
ignored), the same review in several shops (the shop id ignored), and
review counts far above the shop's own deal counter.
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REVIEW_KEY_COLUMNS = ["author", "product", "comment", "city", "meta_info"]
_SHOP_RE = re.compile(r"/shop/comments/([^/?]+)")


def _digits(value) -> int | None:
    found = re.sub(r"\D", "", str(value)) if isinstance(value, str) else ""
    return int(found) if found else None


def audit_pages(cache_dir: Path) -> dict[str, int]:
    """Reviews per page and repeated page content, from the page cache."""
    per_page = Counter()
    # shop -> page content -> page numbers with that content
    contents = defaultdict(lambda: defaultdict(list))
    pages_per_shop = Counter()
    for path in cache_dir.rglob("*.json"):
        try:
            with open(path) as f:
                entry = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        m = _SHOP_RE.search(entry.get("url", ""))
        shop = entry.get("shop_id") or (m.group(1) if m else "?")
        result = entry.get("result", [])
        per_page[len(result)] += 1
        pages_per_shop[shop] += 1
        if result:
            key = json.dumps(result, sort_keys=True, ensure_ascii=False)
            contents[shop][key].append(entry.get("page_num"))

    total = sum(per_page.values())
    print(f"1. reviews per cached page ({total} pages)")
    for n, count in sorted(per_page.items()):
        print(f"  {n:4} reviews: {count:8} pages ({count / max(total, 1):.1%})")
    print("  -> one dominant size (the site's page size) and nothing above it = no double parsing")

    repeated_pages = 0
    examples = []
    for shop, by_content in contents.items():
        for page_nums in by_content.values():
            if len(page_nums) > 1:
                repeated_pages += len(page_nums) - 1
                if len(examples) < 5:
                    examples.append(f"{shop}: pages {sorted(page_nums)[:10]}")
    print(f"\n2. pages identical to another page of the same shop: {repeated_pages}")
    for example in examples:
        print(f"  {example}")
    print("  -> near 0 = the ?p= parameter really pages through different reviews")
    return pages_per_shop


def audit_csvs(output_dir: Path) -> Counter:
    """Duplicates within each shop and across shops, from the review CSVs."""
    files = sorted(output_dir.glob("*.csv"))
    rows_per_shop = Counter()
    within = 0
    hashes = []
    shop_index = []
    for i, path in enumerate(files):
        df = pd.read_csv(path, dtype=str, usecols=REVIEW_KEY_COLUMNS)
        rows_per_shop[path.stem] = len(df)
        if not len(df):
            continue
        h = pd.util.hash_pandas_object(df[REVIEW_KEY_COLUMNS], index=False).to_numpy()
        within += len(h) - len(np.unique(h))
        h = np.unique(h)
        hashes.append(h)
        shop_index.append(np.full(len(h), i, dtype=np.int32))

    rows = sum(rows_per_shop.values())
    print(f"\n3. review rows in {len(files)} CSVs: {rows}")
    print(f"  duplicates within the same shop: {within} ({within / max(rows, 1):.2%})")
    if not hashes:
        return rows_per_shop

    all_hashes = np.concatenate(hashes)
    all_shops = np.concatenate(shop_index)
    unique, counts = np.unique(all_hashes, return_counts=True)
    shared = unique[counts > 1]
    shared_rows = int(counts[counts > 1].sum())
    print(
        f"  reviews found in more than one shop: {len(shared)} distinct, "
        f"{shared_rows} rows ({shared_rows / max(rows, 1):.2%})"
    )
    print("  -> near 0 = each shop's page really shows that shop's reviews")

    # A few of the most repeated ones, to judge by eye whether they're
    # generic texts that genuinely repeat or copies of the same review.
    top = unique[np.argsort(counts)[::-1][: min(3, len(shared))]]
    for h in top:
        shop_files = [files[i] for i in all_shops[all_hashes == h]]
        df = pd.read_csv(shop_files[0], dtype=str, usecols=REVIEW_KEY_COLUMNS)
        row_hashes = pd.util.hash_pandas_object(df, index=False).to_numpy()
        row = df[row_hashes == h].iloc[0].to_dict()
        print(f"  in {len(shop_files)} shops: {row}")
    return rows_per_shop


def audit_deals(products_dir: Path, rows_per_shop: Counter, pages_per_shop: Counter) -> None:
    """Our review count next to the deal counter the shop page shows."""
    deals = {}
    for path in products_dir.glob("*.csv"):
        try:
            df = pd.read_csv(path, dtype=str, usecols=["shop_deals"], nrows=1)
        except (ValueError, pd.errors.EmptyDataError):
            continue
        if len(df):
            deals[path.stem] = (df["shop_deals"].iloc[0], _digits(df["shop_deals"].iloc[0]))

    print(f"\n4. reviews vs the shop's deal counter ({len(deals)} shops with product data)")
    if not deals:
        print("  no product CSVs with shop_deals")
        return
    print(f"  {'shop':38} {'reviews':>8} {'pages':>6} {'deals (raw)':>20} {'reviews/deals':>14}")
    ratios = []
    for shop, n in rows_per_shop.items():
        _, value = deals.get(shop, (None, None))
        if value:
            ratios.append(n / value)
    for shop, n in rows_per_shop.most_common(15):
        raw, value = deals.get(shop, (None, None))
        ratio = f"{n / value:.2f}" if value else "-"
        print(f"  {shop:38} {n:8} {pages_per_shop[shop]:6} {str(raw):>20} {ratio:>14}")
    if ratios:
        ratios = np.array(ratios)
        print(
            f"  reviews/deals over {len(ratios)} shops: median {np.median(ratios):.2f}, "
            f"shops above 1: {(ratios > 1).sum()}"
        )
        print("  -> reviews well above deals in many shops would point to over-counting")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("./config.yaml"))
    args = parser.parse_args()

    # Same path resolution as darknet_scraper.main(): relative to the config.
    repo_root = args.config.resolve().parent
    with open(args.config) as f:
        config = yaml.safe_load(f)

    def resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else repo_root / path

    reviews = config["scrape_reviews"]
    pages_per_shop = audit_pages(resolve(reviews["cache_dir"]))
    rows_per_shop = audit_csvs(resolve(reviews["output_dir"]))
    audit_deals(resolve(config["scrape_products"]["output_dir"]), rows_per_shop, pages_per_shop)


if __name__ == "__main__":
    main()
