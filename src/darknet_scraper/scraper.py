import hashlib
import logging
import re
import time
from dataclasses import dataclass
from typing import Literal, Self
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx2 as httpx
from bs4 import BeautifulSoup
from Crypto.Cipher import AES

from darknet_scraper.auth import AuthError, solve_image_captcha

logger = logging.getLogger(__name__)

_TCK_VARS_RE = re.compile(
    r'var a=toNumbers\("([0-9a-f]+)"\),b=toNumbers\("([0-9a-f]+)"\),'
    r'c=toNumbers\("([0-9a-f]+)"\)'
)
_LOCATION_HREF_RE = re.compile(r'location\.href\s*=\s*"([^"]+)"')


def _unpad_lenient(data: bytes, block_size: int = 16) -> bytes:
    """Replicates aes.min.js's unpadBytesOut exactly.

    Not standard PKCS7: it only strips trailing bytes if they're all <=
    block_size and equal to each other; otherwise it leaves the data
    untouched (even a full, "unpadded-looking" 16-byte block is valid).
    """
    n = len(data)
    pad_count = 0
    pad_byte = -1
    i = n - 1
    limit = n - 1 - block_size
    while i >= limit and i >= 0:
        if data[i] <= block_size:
            if pad_byte == -1:
                pad_byte = data[i]
            if data[i] != pad_byte:
                pad_count = 0
                break
            pad_count += 1
        else:
            break
        if pad_count == pad_byte:
            break
        i -= 1
    return data[: n - pad_count] if pad_count > 0 else data


def _decrypt_tck(key_hex: str, iv_hex: str, ciphertext_hex: str) -> str:
    key = bytes.fromhex(key_hex)
    iv = bytes.fromhex(iv_hex)
    ciphertext = bytes.fromhex(ciphertext_hex)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    plaintext = _unpad_lenient(cipher.decrypt(ciphertext))
    return plaintext.hex()


@dataclass
class ImageCaptchaForm:
    ref: str
    user_id: str
    fate: str
    image_b64: str


class SessionExpired(Exception):
    """Raised when a request comes back 403 or 3xx — the session needs a fresh auth().

    The site answers an expired session (stale TCK cookie, logged out,
    rotated mirror) with a redirect to its challenge/login flow instead of
    the page, so a redirect on a plain page GET means the same thing as 403.
    """


class Scraper:
    def __init__(
        self,
        base_url: str,
        headers: dict,
        cookies: dict,
        timeout: int = 30,
        request_delay: float = 2.0,
        retries: int = 3,
        backoff_base: float = 5.0,
    ):
        self.base_url = base_url
        self.client = None
        self.headers = headers
        self.cookies = cookies
        self.timeout = timeout
        self.request_delay = request_delay
        self.retries = retries
        self.backoff_base = backoff_base

    def build_url(
        self,
        url_type: Literal["shop_catalog", "shop_comments", "shop_item"],
        shop_id: str,
        item_id: str | None = None,
    ) -> str:
        match url_type:
            case "shop_catalog":
                return f"{self.base_url}/shop/catalog/{shop_id}/"
            case "shop_comments":
                return f"{self.base_url}/shop/comments/{shop_id}/"
            case "shop_item" if item_id:
                return f"{self.base_url}/shop/item/{item_id}/?type=instant&city=any&region=any"
            case _:
                raise ValueError(f"Unknown URL type: {url_type}")

    def __enter__(self) -> Self:
        self.client = httpx.Client(
            timeout=self.timeout, headers=self.headers, cookies=self.cookies
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            self.client.close()

    @staticmethod
    def hash_url(url: str) -> str:
        return hashlib.md5(url.encode()).hexdigest()

    @staticmethod
    def to_relative(url: str) -> str:
        parts = urlsplit(url)
        return urlunsplit(("", "", parts.path, parts.query, parts.fragment))

    def request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Make a request, retrying on network errors and 429s with backoff.

        Uses self.retries/self.backoff_base — set once per instance (e.g.
        TorScraper defaults these higher, since Tor circuits flake more
        often than a direct connection) rather than per-call, since nothing
        in this codebase ever needs a one-off override.

        Returns the raw response without checking its status otherwise —
        callers decide what counts as success via response.raise_for_status().
        Re-raises the last httpx.RequestError once retries are exhausted;
        a still-429 response after exhausting retries is returned as-is.
        """
        if not self.client:
            raise RuntimeError("Scraper must be used within a 'with' context manager.")

        for attempt in range(1, self.retries + 1):
            logger.info(f"{method} {url} (attempt {attempt}/{self.retries})")
            try:
                response = self.client.request(method, url, **kwargs)
            except httpx.RequestError as exc:
                if attempt == self.retries:
                    logger.error(
                        f"{type(exc).__name__} on final attempt {attempt}/{self.retries} "
                        f"for {method} {url}: {exc}"
                    )
                    raise
                wait = self.backoff_base * (2 ** (attempt - 1))
                logger.warning(
                    f"{type(exc).__name__} on attempt {attempt}/{self.retries} for "
                    f"{method} {url}: {exc}, waiting {wait}s"
                )
                time.sleep(wait)
                continue

            logger.info(f"{method} {url} -> {response.status_code}")

            if response.status_code == 429 and attempt < self.retries:
                wait = self.backoff_base * (2 ** (attempt - 1))
                logger.warning(
                    f"Rate limited (429) on attempt {attempt}/{self.retries}, "
                    f"waiting {wait}s"
                )
                time.sleep(wait)
                continue

            return response

        return response

    def load_url(self, url: str) -> str | None:
        try:
            response = self.request("GET", url)
            if response.is_redirect:
                location = response.headers.get("Location")
                raise SessionExpired(
                    f"{response.status_code} for {url} -> {location}, session needs re-auth"
                )
            response.raise_for_status()
            return response.text
        except httpx.RequestError as exc:
            logger.error(f"Network error while requesting {url}: {exc}")
            return None
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 403:
                raise SessionExpired(f"403 for {url}, session needs re-auth") from exc
            logger.error(f"HTTP error {exc.response.status_code} for {url}")
            return None

    def get_next_page_url(
        self, soup: BeautifulSoup, current_page_url: str
    ) -> str | None:
        next_btn = soup.select_one("a.next-page[href]")

        if not next_btn:
            return None

        relative_href = next_btn["href"]

        return urljoin(current_page_url, relative_href)

    def scrape_shops_from_main(self, soup: BeautifulSoup) -> list[str]:
        pattern = re.compile(r"^/?shop/catalog/([^/?]+)")
        shop_ids = []
        for a in soup.find_all("a", href=pattern):
            match = pattern.match(a["href"])
            shop_ids.append(match.group(1))
        return shop_ids

    def get_last_page_number(self, soup: BeautifulSoup) -> int | None:
        pagination_container = soup.find("div", class_="pagination-container")
        if not pagination_container:
            return None

        pagination_links = pagination_container.find_all("a", class_="pagination-links")
        if not pagination_links:
            return None

        last_link = pagination_links[-1]
        page_text = last_link.get_text(strip=True)
        try:
            return int(page_text)
        except ValueError:
            return None

    def scrape_shop_reviews(self, soup: BeautifulSoup) -> list[dict]:
        comments_data = []
        comment_blocks = soup.find_all("div", class_="review_comments_info")

        for block in comment_blocks:
            # Skip admin replies, e.g. <div class="review_comments_item admin_comment">.
            item = block.find_parent("div", class_="review_comments_item")
            if item and "admin_comment" in item.get("class", []):
                continue

            # Anonymous reviews show the author as <span class="review_comments_name">.
            # Registered users show it as <a class="review_comments_link"> instead,
            # in which case review_comments_title only has one <span> (the product).
            name_tag = block.find("span", class_="review_comments_name")
            if name_tag:
                author = name_tag.get_text(strip=True)
            else:
                link_tag = block.find("a", class_="review_comments_link")
                author = link_tag.get_text(strip=True) if link_tag else None

            product = None
            title_p = block.find("p", class_="review_comments_title")
            if title_p:
                spans = title_p.find_all("span")
                if spans:
                    product = spans[-1].get_text(strip=True)

            text_tag = block.find("p", class_="review_comments_text")
            comment_text = text_tag.get_text(strip=True) if text_tag else None

            city_tag = block.find("p", class_="city")
            city = city_tag.get_text(strip=True) if city_tag else None

            date_tag = block.find("p", class_="review_comments_data")
            date_str = date_tag.get_text(strip=True) if date_tag else None

            comments_data.append(
                {
                    "author": author,
                    "product": product,
                    "comment": comment_text,
                    "city": city,
                    "meta_info": date_str,
                }
            )

        return comments_data

    def scrape_shop_products(self, soup: BeautifulSoup) -> list[dict]:
        products_data = []
        product_links = soup.find_all("a", class_="product_item")

        for link in product_links:
            href = link.get("href")
            product_url = (
                self.to_relative(urljoin(self.base_url, href)) if href else None
            )

            title_tag = link.find("p", class_="product_title")
            title = title_tag.get_text(strip=True) if title_tag else None

            price_tag = link.find("p", class_="product_price")
            price = price_tag.get_text(" ", strip=True) if price_tag else None

            rating_tag = link.find("span", class_="product_star_col")
            rating = rating_tag.get_text(strip=True) if rating_tag else None

            products_data.append(
                {
                    "product_url": product_url,
                    "title": title,
                    "price": price,
                    "rating": rating,
                }
            )

        return products_data

    def scrape_shop_info(self, soup: BeautifulSoup) -> dict:
        panel = soup.find("div", class_="shop_panel_info")
        if not panel:
            return {"shop_title": None, "shop_deals": None}

        title_tag = panel.find("h1", class_="shop_title")
        shop_title = title_tag.get_text(strip=True) if title_tag else None

        deals_tag = panel.find("p", class_="shop_rait_text")
        deals_span = deals_tag.find("span") if deals_tag else None
        if deals_span:
            shop_deals = deals_span.get_text(strip=True)
        elif deals_tag:
            shop_deals = deals_tag.get_text(" ", strip=True)
        else:
            shop_deals = None

        return {"shop_title": shop_title, "shop_deals": shop_deals}

    def is_tck_challenge_page(self, html: str) -> bool:
        return "slowAES.decrypt" in html or "Пожалуйста, подождите" in html

    def parse_captcha_form(self, html: str) -> ImageCaptchaForm:
        soup = BeautifulSoup(html, "html.parser")
        form = soup.select_one("form.form-captcha")
        if not form:
            raise AuthError("Expected an image-captcha form but didn't find one")

        def hidden(name: str) -> str:
            tag = form.select_one(f'input[name="{name}"]')
            return tag["value"] if tag else ""

        img = form.select_one("#captcha-img")
        if not img or not img.get("src", "").startswith("data:"):
            raise AuthError("Captcha form found but no embedded image")

        return ImageCaptchaForm(
            ref=hidden("ref"),
            user_id=hidden("userId"),
            fate=hidden("fate"),
            image_b64=img["src"].split(",", 1)[1].strip(),
        )

    def parse_login_form(self, html: str) -> dict:
        """Parse the real username/password login form.

        Confirmed live against the real page: fields are timezoneoffset,
        login, password, captcha, posting to /entry/post/login. The page
        also has a second, unrelated registration form (class
        "register_form") — select the login one explicitly rather than
        relying on document order.
        """
        soup = BeautifulSoup(html, "html.parser")
        form = soup.select_one("form.authorization-block:not(.register_form)")
        if not form:
            raise AuthError("Expected a login form but didn't find one")

        img = form.select_one("img")
        if not img or not img.get("src", "").startswith("data:"):
            raise AuthError("Login form found but no embedded captcha image")

        return {
            "action": form.get("action") or "/entry/post/login",
            "image_b64": img["src"].split(",", 1)[1].strip(),
        }

    def solve_tck_bootstrap(self, url: str, max_hops: int = 15) -> str:
        """Follow the TCK 'please wait' AES-challenge chain until past it.

        Each hop's page embeds its own AES key/IV/ciphertext in plaintext
        JS and expects the client to decrypt it, set the result as the TCK
        cookie, and follow the page's own location.href. Returns the final
        (non-challenge) page's HTML.
        """
        logger.info(f"TCK bootstrap: starting at {url}")
        for hop in range(1, max_hops + 1):
            resp = self.request("GET", url, follow_redirects=True)
            resp.raise_for_status()
            html = resp.text

            if not self.is_tck_challenge_page(html):
                logger.info(
                    f"TCK bootstrap: no challenge on this page, treating as done "
                    f"({hop - 1} challenge hop(s) solved)"
                )
                return html

            logger.info(f"TCK bootstrap: challenge hop {hop}/{max_hops} at {url}")

            vars_match = _TCK_VARS_RE.search(html)
            if not vars_match:
                raise AuthError("TCK challenge page didn't match the expected shape")
            key_hex, iv_hex, ciphertext_hex = vars_match.groups()

            href_match = _LOCATION_HREF_RE.search(html)
            if not href_match:
                raise AuthError("TCK challenge page had no location.href to follow")

            self.client.cookies.set(
                "TCK", _decrypt_tck(key_hex, iv_hex, ciphertext_hex)
            )
            url = urljoin(url, href_match.group(1))
            logger.info(f"TCK bootstrap: hop {hop} solved, following redirect to {url}")

        raise AuthError("TCK bootstrap did not resolve within max_hops")

    def auth(
        self,
        captcha_url: str,
        login: str,
        password: str,
        captcha_key: str,
        max_captcha_retries: int = 5,
    ) -> str:
        self._auth_args = (
            captcha_url,
            login,
            password,
            captcha_key,
            max_captcha_retries,
        )

        logger.info(f"auth(): starting login flow at {captcha_url}")
        html = self.solve_tck_bootstrap(captcha_url)
        logger.info("auth(): TCK bootstrap done, parsing image captcha form")

        for attempt in range(1, max_captcha_retries + 1):
            form = self.parse_captcha_form(html)
            logger.info(
                f"auth(): image captcha form parsed (userId={form.user_id}), "
                f"solving via 2captcha"
            )
            answer = solve_image_captcha(
                captcha_key,
                form.image_b64,
                case=False,
                min_length=8,
                max_length=8,
                comment="8-character lowercase alphanumeric code, enter exactly as shown",
            )
            logger.info(f"Image captcha attempt {attempt}: 2captcha answer {answer!r}")

            resp = self.request(
                "POST",
                captcha_url,
                data={
                    "ref": form.ref,
                    "userId": form.user_id,
                    "fate": form.fate,
                    "answer": answer.upper(),
                },
                follow_redirects=True,
            )
            resp.raise_for_status()
            html = resp.text

            if "form-captcha" not in html:
                logger.info(f"auth(): image captcha accepted, landed on {resp.url}")
                break
            logger.warning(f"auth(): image captcha attempt {attempt} rejected, retrying")
            if attempt == max_captcha_retries:
                raise AuthError("Image captcha retries exhausted")

        login_form = self.parse_login_form(html)
        action_url = urljoin(str(resp.url), login_form["action"])
        logger.info(f"auth(): login form parsed, will post credentials to {action_url}")

        for attempt in range(1, max_captcha_retries + 1):
            cyrillic_answer = solve_image_captcha(
                captcha_key,
                login_form["image_b64"],
                numeric=2,
                case=False,
                language_pool="ru",
            )
            logger.info(
                f"Login captcha attempt {attempt}: 2captcha answer {cyrillic_answer!r}"
            )

            resp = self.request(
                "POST",
                action_url,
                data={
                    "timezoneoffset": "0",
                    "login": login,
                    "password": password,
                    "captcha": cyrillic_answer.upper(),
                },
                follow_redirects=True,
            )
            resp.raise_for_status()
            html = resp.text

            # Confirmed live: "no more captcha form on the page" reliably
            # means the login succeeded and landed on the real homepage.
            if not BeautifulSoup(html, "html.parser").select_one(
                "input[name='captcha']"
            ):
                logger.info(f"auth(): login accepted, landed on {resp.url}")
                break
            logger.warning(f"auth(): login captcha attempt {attempt} rejected, retrying")
            if attempt == max_captcha_retries:
                raise AuthError(
                    "Login captcha retries exhausted or credentials rejected"
                )
            login_form = self.parse_login_form(html)
            action_url = urljoin(str(resp.url), login_form["action"])

        parts = urlsplit(str(resp.url))
        self.base_url = urlunsplit((parts.scheme, parts.netloc, "", "", ""))
        logger.info(f"auth(): login flow complete, main site resolved to {self.base_url}")
        return self.base_url

    def reauth(self) -> str:
        """Re-run auth() with the same credentials used last time.

        The main-site mirror it resolves to can differ from before (mirrors
        rotate), so self.base_url may change — callers must rebuild any URL
        via build_url() afterward rather than retrying a stale absolute URL.
        """
        if not hasattr(self, "_auth_args"):
            raise RuntimeError("reauth() called before auth() has ever succeeded")
        return self.auth(*self._auth_args)
