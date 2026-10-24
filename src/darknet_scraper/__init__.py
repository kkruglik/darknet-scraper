import json
import logging
import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
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

# Seconds between worker logins at startup. Each login is a burst of
# requests (TCK hops, two captcha POSTs) to the one shared captcha mirror,
# and four at once tripped the login form's brute-force 429s.
LOGIN_STAGGER = 60.0

# A crashed worker is restarted in place rather than leaving its shops idle.
# Crashes soon after a (re)start count as "quick"; after enough in a row the
# worker gives up, since restarting into the same deterministic bug forever
# would only hide it.
WORKER_RESTART_BASE = 60.0
WORKER_RESTART_MAX = 600.0
WORKER_MIN_UPTIME = 600.0
WORKER_MAX_QUICK_CRASHES = 5


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


def run_supervised(worker: int, run_once: Callable[[int], None]) -> None:
    """Run a worker, restarting it from scratch whenever it crashes.

    run_once(generation) builds a fresh scraper (new client, empty cookies,
    new Tor circuit, new login) and scrapes the worker's shops; finished
    CSVs are skipped and pages come from the cache, so a restart resumes
    cheaply. A crash after a healthy run restarts after WORKER_RESTART_BASE;
    crashes within WORKER_MIN_UPTIME of starting back off further and
    re-raise after WORKER_MAX_QUICK_CRASHES in a row.
    """
    generation = 0
    quick_crashes = 0
    while True:
        started = time.monotonic()
        try:
            run_once(generation)
            return
        except Exception as exc:
            uptime = time.monotonic() - started
            quick_crashes = quick_crashes + 1 if uptime < WORKER_MIN_UPTIME else 0
            if quick_crashes >= WORKER_MAX_QUICK_CRASHES:
                logger.error(
                    f"Worker {worker} crashed {quick_crashes} times in a row within "
                    f"{WORKER_MIN_UPTIME:.0f}s of starting, giving up"
                )
                raise
            wait = min(
                WORKER_RESTART_BASE * 2 ** max(quick_crashes - 1, 0), WORKER_RESTART_MAX
            )
            generation += 1
            logger.error(
                f"Worker {worker} crashed after {uptime:.0f}s, restarting in {wait:.0f}s "
                f"with a new Tor circuit (generation {generation})",
                exc_info=exc,
            )
            time.sleep(wait)


def migrate_cache_keys(s: Scraper, cache_dir: Path) -> None:
    """Rename cache files keyed by an older hash_url() scheme to the current one.

    Each entry stores the URL it was scraped from, so its current key can be
    recomputed. When the same page was cached under several mirrors, they all
    map to one key: the first one renamed wins and the rest are removed as
    duplicates.
    """
    renamed = 0
    duplicates = 0
    for path in cache_dir.rglob("*.json"):
        with open(path) as f:
            page_url = json.load(f).get("url")
        if not page_url:
            continue

        target = path.with_name(f"{s.hash_url(page_url)}.json")
        if target == path:
            continue
        if target.exists():
            path.unlink()
            duplicates += 1
        else:
            path.rename(target)
            renamed += 1

    if renamed or duplicates:
        logger.info(
            f"Migrated cache keys in {cache_dir}: renamed={renamed}, "
            f"removed duplicates={duplicates}"
        )


def scrape_all_shops(
    s: Scraper, start_url: str, cache_dir: Path, output_file: Path
) -> None:
    started_at = time.monotonic()
    logger.info(f"Starting shop discovery: cache={cache_dir}, output={output_file}")
    shop_ids = []
    cache = {}
    cache_dir.mkdir(parents=True, exist_ok=True)
    migrate_cache_keys(s, cache_dir)

    for i in cache_dir.rglob("*.json"):
        with open(i) as f:
            data = json.load(f)
            cache[i.name] = data

    if not cache:
        logger.info("No cache data found, starting from scratch")
    else:
        for i in cache:
            shop_ids.extend(cache[i]["result"])
        logger.info(f"Loaded {len(cache)} cached shop pages with {len(shop_ids)} shop IDs")

    logger.info("Loading first catalog page to determine pagination")
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

    pbar = tqdm(range(1, last_page + 1), desc="Shop pages", unit="page")
    for page_num in pbar:
        pbar.set_postfix(shops=len(shop_ids))
        page_url = f"{start_url}?p={page_num}"
        url_hash = s.hash_url(page_url)
        cache_filename = f"{url_hash}.json"

        if cache_filename in cache:
            logger.info(f"Shop page {page_num}/{last_page}: using cache, skipping request")
            continue

        logger.info(f"Loading shop page {page_num}/{last_page}")
        page_html = load_with_reauth(s, page_url)
        if not page_html:
            tqdm.write(f"Failed to load page {page_num}, stopping discovery")
            logger.error(f"Failed to load shop page {page_num}/{last_page}, stopping discovery")
            break

        soup = BeautifulSoup(page_html, "html.parser")
        found = s.scrape_shops_from_main(soup)
        if not found:
            logger.warning(f"Shop page {page_num}/{last_page}: no shop IDs found")

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
        logger.info(
            f"Shop page {page_num}/{last_page}: found {len(found)} shop IDs, "
            f"total={len(shop_ids)}, cached to {cache_dir / cache_filename}"
        )

    pbar.close()
    logger.info(
        f"Shop discovery finished: {len(shop_ids)} shops across {page_num} pages"
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"shop_id": shop_ids}).to_csv(output_file, index=False)
    logger.info(
        f"Saved {len(shop_ids)} shop IDs to {output_file} "
        f"in {time.monotonic() - started_at:.1f}s"
    )


def scrape_all_reviews(
    s: Scraper,
    input_file: Path,
    cache_dir: Path,
    output_dir: Path,
    worker: int = 0,
    workers: int = 1,
) -> None:
    started_at = time.monotonic()
    logger.info(
        f"Starting review scraping: input={input_file}, cache={cache_dir}, output={output_dir}"
    )
    if not input_file.exists():
        logger.error(f"Cannot scrape reviews: shop file not found: {input_file}")
        raise RuntimeError("Shop file not found")

    shop_ids = pd.read_csv(input_file)["shop_id"].to_list()[worker::workers]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_pages = {p.name for p in cache_dir.rglob("*.json")}

    output_dir.mkdir(parents=True, exist_ok=True)

    if not cached_pages:
        logger.info("No review cache data found, starting from scratch")
    else:
        logger.info(f"Found {len(cached_pages)} cached review pages")

    total_reviews = 0
    written_reviews = 0
    scraped_shops = 0
    skipped_shops = 0
    incomplete_shops = 0
    total_shops = len(shop_ids)
    logger.info(f"Loaded {total_shops} shops for review scraping")
    shop_pbar = tqdm(shop_ids, desc=f"W{worker + 1} shops", unit="shop", position=worker)
    for shop_num, shop_id in enumerate(shop_pbar, start=1):
        shop_pbar.set_postfix(reviews=total_reviews)
        shop_csv = output_dir / f"{shop_id}.csv"
        if shop_csv.exists():
            skipped_shops += 1
            logger.info(
                f"[{shop_num}/{total_shops}] Shop {shop_id}: skipping existing CSV {shop_csv}"
            )
            continue

        log_progress(f"[{shop_num}/{total_shops}] Shop {shop_id}: starting review collection")
        first_page_html = page_html = load_with_reauth(s, s.build_url("shop_comments", shop_id))
        comments_url = s.build_url("shop_comments", shop_id)
        if not page_html:
            log_progress(
                f"[{shop_num}/{total_shops}] Failed to load first review page for "
                f"{shop_id}, skipping shop"
            )
            logger.error(f"Shop {shop_id}: first review page unavailable, CSV not written")
            incomplete_shops += 1
            continue

        soup = BeautifulSoup(page_html, "html.parser")
        last_page = s.get_last_page_number(soup)
        if not last_page:
            log_progress(f"[{shop_num}/{total_shops}] Could not determine last page for {shop_id}")
            logger.warning(f"Shop {shop_id}: review pagination unavailable, falling back to one page")
            last_page = 1

        logger.info(f"Shop {shop_id}: found {last_page} review pages")
        shop_reviews = []
        complete = True
        page_pbar = tqdm(
            range(1, last_page + 1),
            desc="  Review pages",
            unit="page",
            leave=False,
            disable=workers > 1,
        )
        for page_num in page_pbar:
            page_url = f"{comments_url}?p={page_num}"
            url_hash = s.hash_url(page_url)
            cache_filename = f"{url_hash}.json"

            if cache_filename in cached_pages:
                with open(cache_dir / cache_filename) as f:
                    cached_reviews = json.load(f)["result"]
                shop_reviews.extend(cached_reviews)
                logger.info(
                    f"Shop {shop_id}, review page {page_num}/{last_page}: "
                    f"loaded {len(cached_reviews)} reviews from cache"
                )
                page_pbar.set_postfix(reviews=len(shop_reviews))
                continue

            # Page 1 is the page already loaded to read the pagination, so
            # reuse it instead of requesting the same content twice.
            if page_num == 1:
                page_html = first_page_html
            else:
                logger.info(f"Shop {shop_id}: loading review page {page_num}/{last_page}")
                page_html = load_with_reauth(s, page_url)
            comments_url = s.build_url("shop_comments", shop_id)
            page_url = f"{comments_url}?p={page_num}"
            if not page_html:
                log_progress(
                    f"[{shop_num}/{total_shops}] Failed to load review page "
                    f"{page_num} for {shop_id}, stopping this shop"
                )
                logger.error(f"Shop {shop_id}: review page {page_num}/{last_page} unavailable")
                complete = False
                break

            soup = BeautifulSoup(page_html, "html.parser")
            reviews = s.scrape_shop_reviews(soup)
            if not reviews:
                logger.warning(f"Shop {shop_id}, review page {page_num}/{last_page}: no reviews found")
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
            logger.info(
                f"Shop {shop_id}, review page {page_num}/{last_page}: "
                f"found {len(reviews)} reviews, total={len(shop_reviews)}, "
                f"cached to {cache_dir / cache_filename}"
            )

        page_pbar.close()
        total_reviews += len(shop_reviews)
        shop_pbar.set_postfix(reviews=total_reviews)

        # Only a fully paged shop gets a CSV, so an interrupted one is retried
        # on the next run (cheaply, from the page cache) instead of looking done.
        if not complete:
            incomplete_shops += 1
            log_progress(f"[{shop_num}/{total_shops}] Shop {shop_id}: incomplete, CSV not written")
            continue

        pd.DataFrame(shop_reviews, columns=REVIEW_COLUMNS).to_csv(shop_csv, index=False)
        scraped_shops += 1
        written_reviews += len(shop_reviews)
        log_progress(
            f"[{shop_num}/{total_shops}] Shop {shop_id}: {len(shop_reviews)} reviews -> {shop_csv}"
        )

    shop_pbar.close()
    logger.info(
        f"Review scraping finished: {written_reviews} reviews written across "
        f"{scraped_shops} shop files in {output_dir}; collected={total_reviews}, "
        f"skipped={skipped_shops}, incomplete={incomplete_shops}, "
        f"elapsed={time.monotonic() - started_at:.1f}s"
    )


def scrape_all_products(
    s: Scraper,
    input_file: Path,
    cache_dir: Path,
    output_dir: Path,
    worker: int = 0,
    workers: int = 1,
) -> None:
    if not input_file.exists():
        raise RuntimeError("Shops file not found")

    shop_ids = pd.read_csv(input_file)["shop_id"].to_list()[worker::workers]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_pages = {p.name for p in cache_dir.rglob("*.json")}

    output_dir.mkdir(parents=True, exist_ok=True)

    if not cached_pages:
        logger.info("No product cache data found, starting from scratch")

    total_products = 0
    scraped_shops = 0
    total_shops = len(shop_ids)
    shop_pbar = tqdm(shop_ids, desc=f"W{worker + 1} shops", unit="shop", position=worker)
    for shop_num, shop_id in enumerate(shop_pbar, start=1):
        shop_pbar.set_postfix(products=total_products)
        shop_csv = output_dir / f"{shop_id}.csv"
        if shop_csv.exists():
            continue

        first_page_html = page_html = load_with_reauth(s, s.build_url("shop_catalog", shop_id))
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
            range(1, last_page + 1),
            desc="  Product pages",
            unit="page",
            leave=False,
            disable=workers > 1,
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

            # Page 1 is the page already loaded for shop info and pagination.
            if page_num == 1:
                page_html = first_page_html
            else:
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

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(threadName)s %(name)s: %(message)s"
    )

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

    accounts = config.auth.accounts
    workers = len(accounts)

    def make_scraper(worker: int, generation: int = 0) -> TorScraper:
        # A distinct SOCKS username per worker makes Tor (IsolateSOCKSAuth,
        # on by default) put each worker on its own circuit, so the site sees
        # separate clients instead of one client with several sessions. A new
        # generation after a crash gets a new username, so only that worker
        # moves to a fresh circuit (NEWNYM would reset every worker's).
        proxy = urlsplit(os.environ.get("TOR_SOCKS_PROXY", DEFAULT_SOCKS_PROXY))
        proxy = urlunsplit(
            proxy._replace(
                netloc=f"worker{worker}-{generation}:x@{proxy.hostname}:{proxy.port}"
            )
        )
        return TorScraper(
            url,
            headers,
            cookies,
            request_delay=3.0,
            socks_proxy=proxy,
            control_port=int(os.environ.get("TOR_CONTROL_PORT", DEFAULT_CONTROL_PORT)),
        )

    def login(s: TorScraper, worker: int) -> None:
        account = accounts[worker]
        logger.info(f"Worker {worker}: authenticating as {account.login}")
        s.auth_until_success(
            config.auth.captcha_url,
            account.login,
            account.password.get_secret_value(),
            config.auth.captcha_key.get_secret_value(),
            config.auth.max_captcha_retries,
        )
        logger.info(f"Worker {worker}: authenticated, main site resolved to {s.base_url}")

    def run_worker(worker: int) -> None:
        if worker:
            logger.info(f"Worker {worker}: waiting {worker * LOGIN_STAGGER:.0f}s before login")
            time.sleep(worker * LOGIN_STAGGER)
        run_supervised(worker, lambda generation: run_worker_once(worker, generation))

    def run_worker_once(worker: int, generation: int) -> None:
        with make_scraper(worker, generation) as s:
            login(s, worker)

            logger.info(f"Phase 2: worker {worker} scraping products for its shops")
            scrape_all_products(
                s,
                resolve(config.scrape_products.input_file),
                resolve(config.scrape_products.cache_dir),
                resolve(config.scrape_products.output_dir),
                worker,
                workers,
            )

            logger.info(f"Phase 3: worker {worker} scraping reviews for its shops")
            scrape_all_reviews(
                s,
                resolve(config.scrape_reviews.input_file),
                resolve(config.scrape_reviews.cache_dir),
                resolve(config.scrape_reviews.output_dir),
                worker,
                workers,
            )

    # Once, before the workers start: concurrent renames in one cache dir
    # would race each other.
    for cache_dir in (config.scrape_products.cache_dir, config.scrape_reviews.cache_dir):
        cache_dir = resolve(cache_dir)
        if cache_dir.exists():
            migrate_cache_keys(Scraper, cache_dir)

    shops_file = resolve(config.scrape_shops.output_file)
    if shops_file.exists():
        logger.info(f"Phase 1: skipped, {shops_file} already exists")
    else:
        # Single-threaded: the page count is only known after the first page,
        # and it's a one-time phase.
        with make_scraper(0) as s:
            login(s, 0)

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

    # Each worker gets every workers-th shop, so they never share a shop,
    # cache file or CSV and need no locks.
    logger.info(f"Starting {workers} worker(s), one per account")
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="worker") as pool:
        futures = [pool.submit(run_worker, worker) for worker in range(workers)]

    failed = 0
    for worker, future in enumerate(futures):
        if exc := future.exception():
            failed += 1
            logger.error(f"Worker {worker} failed", exc_info=exc)
    if failed:
        raise RuntimeError(f"{failed}/{workers} worker(s) failed")
