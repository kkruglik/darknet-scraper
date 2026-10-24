import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from darknet_scraper import migrate_cache_keys, scrape_all_reviews, scrape_all_shops
from darknet_scraper.scraper import Scraper


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
        self.scraper.load_url.side_effect = ["<html></html>", None]
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

    def test_first_review_page_is_not_requested_twice(self):
        self.scraper.scrape_shop_reviews.return_value = [{"author": "fresh"}]

        self.scrape_reviews()

        url = self.scraper.build_url("shop_comments", 101)
        self.assertEqual(
            [c.args[0] for c in self.scraper.load_url.call_args_list],
            [url, f"{url}?p=2"],
        )
        self.assertEqual(len(pd.read_csv(self.output_dir / "101.csv")), 2)

    def test_worker_scrapes_only_its_share_of_shops(self):
        pd.DataFrame({"shop_id": [101, 102, 103, 104]}).to_csv(self.input_file, index=False)
        self.scraper.get_last_page_number.return_value = 1
        self.scraper.scrape_shop_reviews.return_value = []

        scrape_all_reviews(
            self.scraper, self.input_file, self.cache_dir, self.output_dir, worker=1, workers=2
        )

        self.assertEqual(
            [c.args[0] for c in self.scraper.load_url.call_args_list],
            [
                "https://example.test/shop_comments/102",
                "https://example.test/shop_comments/104",
            ],
        )
        self.assertEqual(sorted(p.name for p in self.output_dir.glob("*.csv")), ["102.csv", "104.csv"])

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
        self.scraper = Scraper(
            "https://example.test", {}, {}, request_delay=1.0, backoff_base=3.0, jitter=0.0
        )
        self.scraper.client = Mock()
        # Fake clock: sleeping advances monotonic time, so pacing waits are exact.
        self.now = 1000.0
        self.enterContext(
            patch("darknet_scraper.scraper.time.monotonic", side_effect=lambda: self.now)
        )
        self.sleep = self.enterContext(
            patch("darknet_scraper.scraper.time.sleep", side_effect=self.advance)
        )

    def advance(self, seconds):
        self.now += seconds

    def respond(self, *statuses, headers=None):
        self.scraper.client.request.side_effect = [
            Mock(status_code=status, headers=headers or {}) for status in statuses
        ]

    def test_429_backs_off_and_raises_delay_then_recovers(self):
        self.respond(429, 429, 200)

        response = self.scraper.request("GET", "https://example.test/page")

        self.assertEqual(response.status_code, 200)
        self.assertEqual([c.args[0] for c in self.sleep.call_args_list], [3.0, 6.0])
        # 1.0 -> x1.5 -> x1.5 on the 429s, then one 0.2s step off on the 200.
        self.assertAlmostEqual(self.scraper.request_delay, 2.25 - 0.2)

    def test_delay_recovers_in_fixed_steps_after_a_429(self):
        self.scraper.request_delay = 10.0
        self.respond(429, *[200] * 25)

        self.scraper.request("GET", "https://example.test/limited")
        self.assertAlmostEqual(self.scraper.request_delay, 15.0 - 0.2)
        for i in range(24):
            self.scraper.request("GET", f"https://example.test/{i}")

        # 2.5 * d successes to get back to the delay that tripped the limit.
        self.assertAlmostEqual(self.scraper.request_delay, 10.0)

    def test_retry_after_header_is_honoured_and_capped(self):
        self.respond(429, 429, 200, headers={"Retry-After": "900"})
        self.scraper.max_backoff = 120.0

        self.scraper.request("GET", "https://example.test/page")

        self.assertEqual([c.args[0] for c in self.sleep.call_args_list], [120.0, 120.0])

    def test_429_log_names_host_retry_after_and_limit_headers(self):
        self.respond(
            429,
            429,
            200,
            headers={"Retry-After": "7", "X-RateLimit-Remaining": "0", "Server": "nginx"},
        )

        with self.assertLogs("darknet_scraper", level="INFO") as logs:
            self.scraper.request("GET", "https://mirror-a.onion/shop/comments/101/?p=2")

        warnings = [m for m in logs.output if m.startswith("WARNING")]
        self.assertEqual(len(warnings), 2)
        self.assertIn("host=mirror-a.onion, retry_after=7", warnings[0])
        self.assertIn("{'X-RateLimit-Remaining': '0'}", warnings[0])
        self.assertNotIn("nginx", warnings[0])
        self.assertEqual(sum("First 429 body" in m for m in logs.output), 1)

    def test_gives_up_after_rate_limit_retries(self):
        self.scraper.rate_limit_retries = 2
        self.respond(429, 429, 429)

        with self.assertLogs("darknet_scraper", level="ERROR"):
            response = self.scraper.request("GET", "https://example.test/page")

        self.assertEqual(response.status_code, 429)
        self.assertEqual(self.scraper.client.request.call_count, 3)

    def test_one_shot_request_is_not_replayed_on_429(self):
        self.respond(429, 200)

        with self.assertLogs("darknet_scraper", level="ERROR"):
            response = self.scraper.request("POST", "https://example.test/login", rate_limit_retries=0)

        self.assertEqual(response.status_code, 429)
        self.assertEqual(self.scraper.client.request.call_count, 1)

    def test_only_2xx_responses_lower_the_delay(self):
        self.scraper.request_delay = 10.0
        self.respond(302, 404, 500, 503, 200)

        for i in range(4):
            response = self.scraper.request("GET", f"https://example.test/{i}")
            self.assertNotEqual(response.status_code, 200)
            self.assertEqual(self.scraper.request_delay, 10.0)
        self.scraper.request("GET", "https://example.test/ok")

        self.assertAlmostEqual(self.scraper.request_delay, 10.0 - 0.2)

    def test_delay_never_drops_below_configured_minimum(self):
        self.respond(200, 200)

        self.scraper.request("GET", "https://example.test/a")
        self.scraper.request("GET", "https://example.test/b")

        self.assertEqual(self.scraper.request_delay, 1.0)

    def test_consecutive_requests_are_paced_by_request_delay(self):
        self.respond(200, 200, 200)

        self.scraper.request("GET", "https://example.test/a")
        self.scraper.request("GET", "https://example.test/b")
        self.advance(5.0)
        self.scraper.request("GET", "https://example.test/c")

        # b waits out the delay after a; c comes after a 5s gap, so no wait.
        self.assertEqual([c.args[0] for c in self.sleep.call_args_list], [1.0])

    def test_jitter_only_stretches_the_delay(self):
        self.scraper.jitter = 0.5
        self.respond(200, 200)

        with patch("darknet_scraper.scraper.random.uniform", return_value=1.3) as uniform:
            self.scraper.request("GET", "https://example.test/a")
            self.scraper.request("GET", "https://example.test/b")

        uniform.assert_called_with(1.0, 1.5)
        self.assertEqual(self.sleep.call_count, 1)
        self.assertAlmostEqual(self.sleep.call_args.args[0], 1.3)

    def test_429_wait_is_never_shorter_than_request_delay(self):
        self.scraper.request_delay = 16.0
        self.respond(429, 200)

        self.scraper.request("GET", "https://example.test/page")

        self.assertEqual([c.args[0] for c in self.sleep.call_args_list], [24.0])


if __name__ == "__main__":
    unittest.main()