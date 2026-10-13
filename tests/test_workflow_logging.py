import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import httpx2 as httpx
import pandas as pd

from darknet_scraper import load_with_reauth, migrate_cache_keys, scrape_all_reviews, scrape_all_shops
from darknet_scraper.scraper import RateLimited, Scraper
from darknet_scraper.tor_client import TorScraper


class WorkflowLoggingTests(unittest.TestCase):
    def setUp(self):
        temporary_dir = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(temporary_dir.cleanup)
        self.root = Path(temporary_dir.name)
        self.cache_dir = self.root / "cache"
        self.cache_dir.mkdir()
        self.output_dir = self.root / "output"
        self.input_file = self.root / "shops.csv"
        pd.DataFrame({"shop_id": [101]}).to_csv(self.input_file, index=False)
        self.scraper = Mock(spec=Scraper)
        self.scraper.request_delay = 0
        self.scraper.load_url.return_value = "<html></html>"
        self.scraper.get_last_page_number.return_value = 2
        self.scraper.hash_url.side_effect = lambda url: hashlib.sha256(url.encode()).hexdigest()
        self.scraper.build_url.side_effect = lambda route, shop_id: f"https://example.test/{route}/{shop_id}"
        self.enterContext(patch("sys.stderr", new=io.StringIO()))
        self.enterContext(patch("sys.stdout", new=io.StringIO()))

    def cache_page(self, url, result):
        filename = self.scraper.hash_url(url) + ".json"
        (self.cache_dir / filename).write_text(json.dumps({"result": result}))

    def scrape_reviews(self):
        scrape_all_reviews(self.scraper, self.input_file, self.cache_dir, self.output_dir)

    def test_shop_discovery_logs_cached_and_fresh_pages(self):
        url = "https://example.test/catalog"
        self.cache_page(f"{url}?p=1", [101])
        self.scraper.scrape_shops_from_main.return_value = [202]
        output_file = self.root / "discovered.csv"

        with self.assertLogs("darknet_scraper", level="INFO") as logs:
            scrape_all_shops(self.scraper, url, self.cache_dir, output_file)

        messages = "\n".join(logs.output)
        self.assertIn("Loaded 1 cached shop pages with 1 shop IDs", messages)
        self.assertIn("Shop page 1/2: using cache, skipping request", messages)
        self.assertIn("Shop page 2/2: found 1 shop IDs, total=2", messages)
        self.assertIn("Saved 2 shop IDs", messages)
        self.assertEqual(pd.read_csv(output_file)["shop_id"].to_list(), [101, 202])
        self.assertEqual(self.scraper.load_url.call_count, 2)

    def test_shop_page_failure_is_logged_to_file_handler(self):
        self.scraper.load_url.side_effect = ["<html></html>", None]
        output_file = self.root / "discovered.csv"

        with self.assertLogs("darknet_scraper", level="INFO") as logs:
            scrape_all_shops(self.scraper, "https://example.test/catalog", self.cache_dir, output_file)

        self.assertIn(
            "ERROR:darknet_scraper:Failed to load shop page 1/2, stopping discovery",
            logs.output,
        )
        self.assertTrue(pd.read_csv(output_file).empty)

    def test_review_logs_cache_progress_and_written_count(self):
        url = self.scraper.build_url("shop_comments", 101)
        self.cache_page(f"{url}?p=1", [{"author": "cached", "shop_id": 101}])
        self.scraper.scrape_shop_reviews.return_value = [{"author": "fresh"}]

        with self.assertLogs("darknet_scraper", level="INFO") as logs:
            self.scrape_reviews()

        messages = "\n".join(logs.output)
        self.assertIn("Shop 101: starting review collection", messages)
        self.assertIn("Shop 101: found 2 review pages", messages)
        self.assertIn("review page 1/2: loaded 1 reviews from cache", messages)
        self.assertIn("review page 2/2: found 1 reviews, total=2", messages)
        self.assertIn("2 reviews written across 1 shop files", messages)
        self.assertIn("skipped=0, incomplete=0, elapsed=", messages)
        rows = pd.read_csv(self.output_dir / "101.csv")
        self.assertEqual(rows["author"].to_list(), ["cached", "fresh"])
        self.assertEqual(rows["shop_id"].to_list(), [101, 101])
        self.assertEqual(self.scraper.load_url.call_count, 2)

    def test_existing_csv_is_logged_and_not_requested(self):
        self.output_dir.mkdir()
        output_file = self.output_dir / "101.csv"
        output_file.write_text("author,shop_id\ncached,101\n")

        with self.assertLogs("darknet_scraper", level="INFO") as logs:
            self.scrape_reviews()

        messages = "\n".join(logs.output)
        self.assertIn("Shop 101: skipping existing CSV", messages)
        self.assertIn("skipped=1, incomplete=0", messages)
        self.scraper.load_url.assert_not_called()
        self.assertEqual(output_file.read_text(), "author,shop_id\ncached,101\n")

    def test_incomplete_reviews_are_not_reported_as_written(self):
        self.scraper.load_url.side_effect = ["<html></html>", "<html></html>", None]
        self.scraper.scrape_shop_reviews.return_value = [{"author": "fresh"}]

        with self.assertLogs("darknet_scraper", level="INFO") as logs:
            self.scrape_reviews()

        messages = "\n".join(logs.output)
        self.assertIn("ERROR:darknet_scraper:Shop 101: review page 2/2 unavailable", messages)
        self.assertIn("incomplete, CSV not written", messages)
        self.assertIn("0 reviews written across 0 shop files", messages)
        self.assertIn("collected=1, skipped=0, incomplete=1", messages)
        self.assertFalse((self.output_dir / "101.csv").exists())

    def test_unavailable_first_review_page_counts_incomplete_shop(self):
        self.scraper.load_url.return_value = None

        with self.assertLogs("darknet_scraper", level="INFO") as logs:
            self.scrape_reviews()

        messages = "\n".join(logs.output)
        self.assertIn("first review page unavailable, CSV not written", messages)
        self.assertIn("skipped=0, incomplete=1", messages)
        self.assertFalse((self.output_dir / "101.csv").exists())

    def test_missing_review_pagination_and_empty_page_are_warnings(self):
        self.scraper.get_last_page_number.return_value = None
        self.scraper.scrape_shop_reviews.return_value = []

        with self.assertLogs("darknet_scraper", level="INFO") as logs:
            self.scrape_reviews()

        messages = "\n".join(logs.output)
        self.assertIn("WARNING:darknet_scraper:Shop 101: review pagination unavailable", messages)
        self.assertIn("WARNING:darknet_scraper:Shop 101, review page 1/1: no reviews found", messages)
        self.assertTrue(pd.read_csv(self.output_dir / "101.csv").empty)


class CacheKeyTests(unittest.TestCase):
    def setUp(self):
        temporary_dir = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(temporary_dir.cleanup)
        self.cache_dir = Path(temporary_dir.name)
        self.scraper = Mock(spec=Scraper)
        self.scraper.hash_url.side_effect = Scraper.hash_url

    def legacy_cache_page(self, url):
        filename = hashlib.md5(url.encode()).hexdigest() + ".json"
        (self.cache_dir / filename).write_text(json.dumps({"url": url, "result": [url]}))
        return self.cache_dir / filename

    def test_hash_url_ignores_mirror_host(self):
        self.assertEqual(
            Scraper.hash_url("http://mirror-a.onion/shop/comments/101/?p=2"),
            Scraper.hash_url("http://mirror-b.onion/shop/comments/101/?p=2"),
        )
        self.assertNotEqual(
            Scraper.hash_url("http://mirror-a.onion/shop/comments/101/?p=1"),
            Scraper.hash_url("http://mirror-a.onion/shop/comments/101/?p=2"),
        )

    def test_migration_renames_legacy_files_and_drops_mirror_duplicates(self):
        self.legacy_cache_page("http://mirror-a.onion/shop/comments/101/?p=1")
        self.legacy_cache_page("http://mirror-b.onion/shop/comments/101/?p=1")
        (self.cache_dir / "no_url.json").write_text(json.dumps({"result": []}))

        migrate_cache_keys(self.scraper, self.cache_dir)

        expected = Scraper.hash_url("/shop/comments/101/?p=1") + ".json"
        self.assertEqual(
            sorted(p.name for p in self.cache_dir.glob("*.json")),
            sorted([expected, "no_url.json"]),
        )

    def test_migration_is_noop_for_current_keys(self):
        legacy = self.legacy_cache_page("http://mirror-a.onion/shop/comments/101/?p=1")
        migrate_cache_keys(self.scraper, self.cache_dir)
        names = sorted(p.name for p in self.cache_dir.glob("*.json"))

        migrate_cache_keys(self.scraper, self.cache_dir)

        self.assertFalse(legacy.exists())
        self.assertEqual(sorted(p.name for p in self.cache_dir.glob("*.json")), names)


class RateLimitTests(unittest.TestCase):
    def setUp(self):
        self.scraper = Scraper("https://example.test", {}, {}, request_delay=1.0, backoff_base=3.0)
        self.scraper.client = Mock()
        self.sleep = self.enterContext(patch("darknet_scraper.scraper.time.sleep"))

    def respond(self, *statuses, headers=None):
        self.scraper.client.request.side_effect = [
            status if isinstance(status, Exception) else Mock(status_code=status, headers=headers or {})
            for status in statuses
        ]

    def request(self, url="https://example.test/page", **kwargs):
        with self.assertLogs("darknet_scraper", level="INFO"):
            return self.scraper.request("GET", url, **kwargs)

    def waits(self):
        return [c.args[0] for c in self.sleep.call_args_list]

    def test_429_backs_off_and_raises_delay_then_recovers(self):
        self.respond(429, 429, 200)

        response = self.request()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.waits(), [3.0, 6.0])
        self.assertAlmostEqual(self.scraper.request_delay, 4.0 * 0.95)

    def test_retry_after_is_used_as_is_below_the_ceiling(self):
        self.respond(429, 200, headers={"Retry-After": "7"})

        self.request()

        self.assertEqual(self.waits(), [7.0])

    def test_retry_after_above_max_backoff_is_still_honoured(self):
        self.scraper.max_backoff = 60.0
        self.respond(429, 200, headers={"Retry-After": "300"})

        self.request()

        self.assertEqual(self.waits(), [300.0])

    def test_retry_after_is_capped_at_max_retry_after(self):
        self.scraper.max_retry_after = 120.0
        self.respond(429, 200, headers={"Retry-After": "900"})

        self.request()

        self.assertEqual(self.waits(), [120.0])

    def test_retry_after_zero_still_waits(self):
        self.respond(429, 200, headers={"Retry-After": "0"})

        self.request()

        self.assertEqual(self.waits(), [1.0])

    def test_non_numeric_retry_after_falls_back_to_backoff(self):
        for value in ["Wed, 21 Oct 2026 07:28:00 GMT", "\u00b2", "nan"]:
            with self.subTest(value=value):
                self.sleep.reset_mock()
                self.respond(429, 200, headers={"Retry-After": value})

                self.request()

                self.assertEqual(self.waits(), [3.0])

    def test_gives_up_after_rate_limit_retries(self):
        self.scraper.rate_limit_retries = 2
        self.respond(429, 429, 429)

        response = self.request()

        self.assertEqual(response.status_code, 429)
        self.assertEqual(self.scraper.client.request.call_count, 3)
        self.assertEqual(self.waits(), [3.0, 6.0])
        self.assertEqual(self.scraper.request_delay, 8.0)

    def test_per_call_override_disables_429_retries(self):
        self.respond(429)

        response = self.request(rate_limit_retries=0)

        self.assertEqual(response.status_code, 429)
        self.assertEqual(self.scraper.client.request.call_count, 1)
        self.sleep.assert_not_called()

    def test_repeatedly_exhausted_budget_raises_rate_limited(self):
        self.scraper.rate_limit_retries = 0
        self.scraper.rate_limit_strikes = 3
        self.respond(429, 429, 429)

        self.assertEqual(self.request("https://example.test/a").status_code, 429)
        self.assertEqual(self.request("https://example.test/b").status_code, 429)
        with self.assertRaises(RateLimited):
            self.request("https://example.test/c")

    def test_any_other_response_resets_the_strike_count(self):
        self.scraper.rate_limit_retries = 0
        self.scraper.rate_limit_strikes = 2
        self.respond(429, 200, 429)

        self.request("https://example.test/a")
        self.request("https://example.test/b")

        self.assertEqual(self.request("https://example.test/c").status_code, 429)

    def test_delay_is_capped_at_max_delay(self):
        self.scraper.max_delay = 5.0
        self.respond(*[429] * 4, 200)

        self.request()

        self.assertEqual(self.scraper.request_delay, 5.0 * 0.95)

    def test_delay_never_drops_below_configured_minimum(self):
        self.respond(200, 200)

        self.request("https://example.test/a")
        self.request("https://example.test/b")

        self.assertEqual(self.scraper.request_delay, 1.0)

    def test_max_delay_is_never_below_configured_delay(self):
        scraper = Scraper("https://example.test", {}, {}, request_delay=60.0, max_delay=30.0)

        scraper._slow_down()

        self.assertEqual(scraper.request_delay, 60.0)

    def test_network_errors_exhaust_their_own_budget(self):
        self.scraper.retries = 3
        error = httpx.ConnectError("boom")
        self.respond(error, error, error)

        with self.assertRaises(httpx.ConnectError):
            self.request()

        self.assertEqual(self.waits(), [3.0, 6.0])

    def test_received_response_resets_network_failures(self):
        self.scraper.retries = 2
        error = httpx.ReadTimeout("slow")
        self.respond(error, 429, error, 200)

        response = self.request()

        self.assertEqual(response.status_code, 200)

    def test_tor_scraper_maps_positional_args(self):
        scraper = TorScraper("https://example.test", {}, {})

        self.assertEqual(
            (scraper.timeout, scraper.request_delay, scraper.retries, scraper.backoff_base),
            (60, 3.0, 5, 3.0),
        )
        self.assertEqual(scraper.rate_limit_retries, 10)


class LoadWithReauthTests(unittest.TestCase):
    def setUp(self):
        self.scraper = Mock(spec=Scraper)
        self.scraper.base_url = "https://mirror-b.test"
        self.scraper.rate_limit_cooldown = 600
        self.sleep = self.enterContext(patch("darknet_scraper.time.sleep"))

    def test_rate_limited_session_pauses_reauths_and_retries(self):
        self.scraper.load_url.side_effect = [RateLimited("blocked"), "<html></html>"]

        with self.assertLogs("darknet_scraper", level="WARNING"):
            html = load_with_reauth(self.scraper, "https://mirror-a.test/shop/comments/1/")

        self.assertEqual(html, "<html></html>")
        self.sleep.assert_called_once_with(600)
        self.scraper.reauth.assert_called_once()
        self.scraper.load_url.assert_called_with("https://mirror-b.test/shop/comments/1/")

    def test_still_rate_limited_after_reauth_propagates(self):
        self.scraper.load_url.side_effect = [RateLimited("blocked"), RateLimited("blocked")]

        with self.assertLogs("darknet_scraper", level="WARNING"), self.assertRaises(RateLimited):
            load_with_reauth(self.scraper, "https://mirror-a.test/shop/comments/1/")


if __name__ == "__main__":
    unittest.main()