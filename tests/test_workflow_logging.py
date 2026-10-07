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


if __name__ == "__main__":
    unittest.main()