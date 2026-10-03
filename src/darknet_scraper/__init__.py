import json
import logging
import os
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import pandas as pd
import yaml
from bs4 import BeautifulSoup
from tqdm import tqdm

from darknet_scraper.config import ScraperConfig
from darknet_scraper.scraper import Scraper, SessionExpired

logger = logging.getLogger(__name__)

REVIEW_COLUMNS = ["author", "product", "comment", "city", "meta_info", "shop_id"]
PRODUCT_COLUMNS = [
    "product_url",
    "title",
    "price",
    "rating",
    "shop_id",
    "shop_title",
    "shop_deals",
]


headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:156.0) Gecko/20100101 Firefox/156.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Sec-GPC": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-origin",
    "Priority": "u=0, i",
    "TE": "trailers",
}

cookies = {}

url = ""


def log_progress(message: str) -> None:
    """Write a shop-level milestone to both the console and the log file.

    Neither alone gives the full picture: tqdm's live bar (and tqdm.write's
    console output) often doesn't survive Docker's log pipe reliably, while
    the file-side per-request logs have no per-shop progress context of
    their own — just a flat stream of URLs with no sense of how far through
    the shop list we are.
    """
    tqdm.write(message)
    logger.info(message)


def load_with_reauth(s: Scraper, url: str) -> str | None:
    """Load a page, re-authenticating once if the session has expired.

    reauth() can resolve to a different mirror, so the retry swaps the URL's
    host for the fresh s.base_url instead of hitting the stale one. A page
    that is still rejected right after a fresh login is treated as
    page-specific (e.g. a removed shop redirecting away) and returns None,
    instead of raising and taking the whole run down.
    """
    try:
        return s.load_url(url)
    except SessionExpired as exc:
        logger.warning(f"Session expired ({exc}), re-authenticating")
        s.reauth()

    parts = urlsplit(url)
    url = urlunsplit(parts._replace(netloc=urlsplit(s.base_url).netloc))
    try:
        return s.load_url(url)
    except SessionExpired as exc:
        logger.error(f"Still rejected right after re-auth, skipping: {exc}")
        return None


def scrape_all_shops(
    s: Scraper, start_url: str, cache_dir: Path, output_file: Path
) -> None:
    shop_ids = []
    cache = {}
    cache_dir.mkdir(parents=True, exist_ok=True)

    for i in cache_dir.rglob("*.json"):
        with open(i) as f:
            data = json.load(f)
            cache[i.name] = data

    if not cache:
        logger.info("No cache data found, starting from scratch")
    else:
        for i in cache:
            shop_ids.extend(cache[i]["result"])

    page_html = load_with_reauth(s, start_url)
    if not page_html:
        logger.error(f"Failed to load first page {start_url}, stopping discovery")
        return

    soup = BeautifulSoup(page_html, "html.parser")
    last_page = s.get_last_page_number(soup)
    if not last_page:
        logger.error("Could not determine last page number, stopping discovery")
        return

    logger.info(f"Found {last_page} pages to scrape")
    time.sleep(s.request_delay)

    pbar = tqdm(range(1, last_page + 1), desc="Shop pages", unit="page")
    for page_num in pbar:
        pbar.set_postfix(shops=len(shop_ids))
        page_url = f"{start_url}?p={page_num}"
        url_hash = s.hash_url(page_url)
        cache_filename = f"{url_hash}.json"

        if cache_filename in cache:
            continue

        page_html = load_with_reauth(s, page_url)
        if not page_html:
            tqdm.write(f"Failed to load page {page_num}, stopping discovery")
            break

        soup = BeautifulSoup(page_html, "html.parser")
        found = s.scrape_shops_from_main(soup)

        entry = {
            "url": page_url,
            "page_num": page_num,
            "scraped_at": datetime.now().timestamp(),
            "found": len(found),
            "result": found,
        }
        with open(cache_dir / cache_filename, "w") as f:
            json.dump(entry, f)
        cache[cache_filename] = entry

        shop_ids.extend(found)
        pbar.set_postfix(shops=len(shop_ids))

        time.sleep(s.request_delay)

    pbar.close()
    logger.info(
        f"Shop discovery finished: {len(shop_ids)} shops across {page_num} pages"
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"shop_id": shop_ids}).to_csv(output_file, index=False)


def scrape_all_reviews(
    s: Scraper, input_file: Path, cache_dir: Path, output_dir: Path
) -> None:
    if not input_file.exists():
        raise RuntimeError("Shop file not found")

    shop_ids = pd.read_csv(input_file)["shop_id"].to_list()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_pages = {p.name for p in cache_dir.rglob("*.json")}

    output_dir.mkdir(parents=True, exist_ok=True)

    if not cached_pages:
        logger.info("No review cache data found, starting from scratch")

    total_reviews = 0
    scraped_shops = 0
    total_shops = len(shop_ids)
    shop_pbar = tqdm(shop_ids, desc="Shops", unit="shop")
    for shop_num, shop_id in enumerate(shop_pbar, start=1):
        shop_pbar.set_postfix(reviews=total_reviews)
        shop_csv = output_dir / f"{shop_id}.csv"
        if shop_csv.exists():
            continue

        page_html = load_with_reauth(s, s.build_url("shop_comments", shop_id))
        comments_url = s.build_url("shop_comments", shop_id)
        if not page_html:
            log_progress(
                f"[{shop_num}/{total_shops}] Failed to load first review page for "
                f"{shop_id}, stopping"
            )
            continue

        soup = BeautifulSoup(page_html, "html.parser")
        last_page = s.get_last_page_number(soup)
        if not last_page:
            log_progress(f"[{shop_num}/{total_shops}] Could not determine last page for {shop_id}")
            last_page = 1

        shop_reviews = []
        complete = True
        page_pbar = tqdm(
            range(1, last_page + 1), desc="  Review pages", unit="page", leave=False
        )
        for page_num in page_pbar:
            page_url = f"{comments_url}?p={page_num}"
            url_hash = s.hash_url(page_url)
            cache_filename = f"{url_hash}.json"

            if cache_filename in cached_pages:
                with open(cache_dir / cache_filename) as f:
                    shop_reviews.extend(json.load(f)["result"])
                page_pbar.set_postfix(reviews=len(shop_reviews))
                continue

            page_html = load_with_reauth(s, page_url)
            comments_url = s.build_url("shop_comments", shop_id)
            page_url = f"{comments_url}?p={page_num}"
            if not page_html:
                log_progress(
                    f"[{shop_num}/{total_shops}] Failed to load review page "
                    f"{page_num} for {shop_id}, stopping"
                )
                complete = False
                break

            soup = BeautifulSoup(page_html, "html.parser")
            reviews = s.scrape_shop_reviews(soup)
            for review in reviews:
                review["shop_id"] = shop_id

            entry = {
                "url": page_url,
                "page_num": page_num,
                "scraped_at": datetime.now().timestamp(),
                "found": len(reviews),
                "result": reviews,
                "shop_id": shop_id,
            }
            with open(cache_dir / cache_filename, "w") as f:
                json.dump(entry, f)
            cached_pages.add(cache_filename)

            shop_reviews.extend(reviews)
            page_pbar.set_postfix(reviews=len(shop_reviews))

            time.sleep(s.request_delay)

        page_pbar.close()
        total_reviews += len(shop_reviews)
        shop_pbar.set_postfix(reviews=total_reviews)

        # Only a fully paged shop gets a CSV, so an interrupted one is retried
        # on the next run (cheaply, from the page cache) instead of looking done.
        if not complete:
            log_progress(f"[{shop_num}/{total_shops}] Shop {shop_id}: incomplete, CSV not written")
            continue

        pd.DataFrame(shop_reviews, columns=REVIEW_COLUMNS).to_csv(shop_csv, index=False)
        scraped_shops += 1
        log_progress(
            f"[{shop_num}/{total_shops}] Shop {shop_id}: {len(shop_reviews)} reviews -> {shop_csv}"
        )

    shop_pbar.close()
    logger.info(
        f"Review scraping finished: {total_reviews} reviews written across "
        f"{scraped_shops} shop files in {output_dir}"
    )


def scrape_all_products(
    s: Scraper, input_file: Path, cache_dir: Path, output_dir: Path
) -> None:
    if not input_file.exists():
        raise RuntimeError("Shops file not found")

    shop_ids = pd.read_csv(input_file)["shop_id"].to_list()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_pages = {p.name for p in cache_dir.rglob("*.json")}

    output_dir.mkdir(parents=True, exist_ok=True)

    if not cached_pages:
        logger.info("No product cache data found, starting from scratch")

    total_products = 0
    scraped_shops = 0
    total_shops = len(shop_ids)
    shop_pbar = tqdm(shop_ids, desc="Shops", unit="shop")
    for shop_num, shop_id in enumerate(shop_pbar, start=1):
        shop_pbar.set_postfix(products=total_products)
        shop_csv = output_dir / f"{shop_id}.csv"
        if shop_csv.exists():
            continue

        page_html = load_with_reauth(s, s.build_url("shop_catalog", shop_id))
        shop_url = s.build_url("shop_catalog", shop_id)
        if not page_html:
            log_progress(
                f"[{shop_num}/{total_shops}] Failed to load first product page for "
                f"{shop_id}, stopping"
            )
            continue

        soup = BeautifulSoup(page_html, "html.parser")
        shop_info = s.scrape_shop_info(soup)
        last_page = s.get_last_page_number(soup)
        if not last_page:
            log_progress(f"[{shop_num}/{total_shops}] Could not determine last page for {shop_id}")
            last_page = 1

        shop_products = []
        complete = True
        page_pbar = tqdm(
            range(1, last_page + 1), desc="  Product pages", unit="page", leave=False
        )
        for page_num in page_pbar:
            page_url = f"{shop_url}?p={page_num}"
            url_hash = s.hash_url(page_url)
            cache_filename = f"{url_hash}.json"

            if cache_filename in cached_pages:
                with open(cache_dir / cache_filename) as f:
                    shop_products.extend(json.load(f)["result"])
                page_pbar.set_postfix(products=len(shop_products))
                continue

            page_html = load_with_reauth(s, page_url)
            shop_url = s.build_url("shop_catalog", shop_id)
            page_url = f"{shop_url}?p={page_num}"
            if not page_html:
                log_progress(
                    f"[{shop_num}/{total_shops}] Failed to load product page "
                    f"{page_num} for {shop_id}, stopping"
                )
                complete = False
                break

            soup = BeautifulSoup(page_html, "html.parser")
            products = s.scrape_shop_products(soup)
            for product in products:
                product["shop_id"] = shop_id

            entry = {
                "url": page_url,
                "page_num": page_num,
                "scraped_at": datetime.now().timestamp(),
                "found": len(products),
                "result": products,
                "shop_id": shop_id,
            }
            with open(cache_dir / cache_filename, "w") as f:
                json.dump(entry, f)
            cached_pages.add(cache_filename)

            shop_products.extend(products)
            page_pbar.set_postfix(products=len(shop_products))

            time.sleep(s.request_delay)

        page_pbar.close()
        total_products += len(shop_products)
        shop_pbar.set_postfix(products=total_products)

        if not complete:
            log_progress(f"[{shop_num}/{total_shops}] Shop {shop_id}: incomplete, CSV not written")
            continue

        for product in shop_products:
            product.update(shop_info)

        pd.DataFrame(shop_products, columns=PRODUCT_COLUMNS).to_csv(
            shop_csv, index=False
        )
        scraped_shops += 1
        log_progress(
            f"[{shop_num}/{total_shops}] Shop {shop_id}: {len(shop_products)} products -> {shop_csv}"
        )

    shop_pbar.close()
    logger.info(
        f"Product scraping finished: {total_products} products written across "
        f"{scraped_shops} shop files in {output_dir}"
    )


def main() -> None:
    config_file = Path("./config.yaml")
    repo_root = config_file.resolve().parent

    def resolve(p: Path) -> Path:
        return p if p.is_absolute() else (repo_root / p).resolve()

    log_dir = resolve(Path("logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    # 10MB x 5 backups: bounded disk usage for a long-running/unattended scrape
    # instead of one log file growing forever.
    file_handler = RotatingFileHandler(
        log_dir / "darknet-scraper.log", maxBytes=10 * 1024 * 1024, backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # WARNING-only on the console: per-request INFO logs would otherwise
    # interleave with tqdm's \r-redrawn progress bars on the same stream and
    # corrupt both. Full detail still goes to the file.
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.WARNING)
    stream_handler.setFormatter(formatter)

    logging.basicConfig(level=logging.INFO, handlers=[file_handler, stream_handler])
    logging.getLogger("httpx2").setLevel(logging.WARNING)

    with open(config_file) as f:
        config = ScraperConfig(**yaml.safe_load(f))

    from darknet_scraper.tor_client import DEFAULT_CONTROL_PORT, DEFAULT_SOCKS_PROXY, TorScraper

    scraper = TorScraper(
        url,
        headers,
        cookies,
        request_delay=1.0,
        socks_proxy=os.environ.get("TOR_SOCKS_PROXY", DEFAULT_SOCKS_PROXY),
        control_port=int(os.environ.get("TOR_CONTROL_PORT", DEFAULT_CONTROL_PORT)),
    )
    with scraper as s:
        logger.info("Phase 0: authenticating")
        s.auth(
            config.auth.captcha_url,
            config.auth.login,
            config.auth.password.get_secret_value(),
            config.auth.captcha_key.get_secret_value(),
            config.auth.max_captcha_retries,
        )
        logger.info(f"Authenticated, main site resolved to {s.base_url}")

        shops_file = resolve(config.scrape_shops.output_file)
        if shops_file.exists():
            logger.info(f"Phase 1: skipped, {shops_file} already exists")
        else:
            # Use the path from config, but the domain auth() actually
            # resolved to — the configured domain goes stale as soon as the
            # main-site mirror rotates, which happens between runs (and
            # sometimes within one).
            configured = urlsplit(config.scrape_shops.start_url)
            resolved = urlsplit(s.base_url)
            shops_start_url = urlunsplit(
                (resolved.scheme, resolved.netloc, configured.path, configured.query, "")
            )
            if shops_start_url != config.scrape_shops.start_url:
                logger.info(
                    f"Phase 1: configured start_url domain is stale, using "
                    f"{shops_start_url} instead of {config.scrape_shops.start_url}"
                )

            logger.info("Phase 1: discovering shops from main catalog")
            scrape_all_shops(
                s,
                shops_start_url,
                resolve(config.scrape_shops.cache_dir),
                shops_file,
            )

        logger.info("Phase 2: scraping products for each shop")
        scrape_all_products(
            s,
            resolve(config.scrape_products.input_file),
            resolve(config.scrape_products.cache_dir),
            resolve(config.scrape_products.output_dir),
        )

        logger.info("Phase 3: scraping reviews for each shop")
        scrape_all_reviews(
            s,
            resolve(config.scrape_reviews.input_file),
            resolve(config.scrape_reviews.cache_dir),
            resolve(config.scrape_reviews.output_dir),
        )
