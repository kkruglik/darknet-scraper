import logging
import time

import httpx2 as httpx

from darknet_scraper.scraper import Scraper

logger = logging.getLogger(__name__)

# Tor Browser's bundled tor process defaults to these ports (a standalone
# `tor` daemon installed separately defaults to 9050/9051 instead).
DEFAULT_SOCKS_PROXY = "socks5h://127.0.0.1:9150"
DEFAULT_CONTROL_PORT = 9151

# Clearnet endpoint reachable through a Tor exit node, used to confirm
# traffic is actually flowing through Tor before scraping starts.
TOR_CHECK_URL = "https://check.torproject.org/api/ip"


class TorScraper(Scraper):
    """Scraper that routes all requests through a local Tor SOCKS proxy.

    Needs "socks5h" rather than "socks5": .onion names aren't real DNS
    names, so hostname resolution must happen inside Tor, not locally.
    """

    def __init__(
        self,
        base_url: str,
        headers: dict,
        cookies: dict,
        timeout: int = 60,
        request_delay: float = 3.0,
        retries: int = 5,
        backoff_base: float = 3.0,
        socks_proxy: str = DEFAULT_SOCKS_PROXY,
        control_port: int | None = DEFAULT_CONTROL_PORT,
        control_password: str | None = None,
    ):
        super().__init__(base_url, headers, cookies, timeout, request_delay, retries, backoff_base)
        self.socks_proxy = socks_proxy
        self.control_port = control_port
        self.control_password = control_password

    def __enter__(self) -> "TorScraper":
        self.client = httpx.Client(
            timeout=self.timeout,
            headers=self.headers,
            cookies=self.cookies,
            proxy=self.socks_proxy,
        )
        return self

    def verify_tor(self) -> bool:
        """Confirm requests are actually leaving through Tor, not direct."""
        if not self.client:
            raise RuntimeError(
                "TorScraper must be used within a 'with' context manager."
            )
        try:
            response = self.client.get(TOR_CHECK_URL, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            if data.get("IsTor"):
                logger.info(f"Tor connectivity confirmed, exit IP: {data.get('IP')}")
                return True
            logger.error(f"Reached the check endpoint but IsTor=False: {data}")
            return False
        except httpx.RequestError as exc:
            logger.error(f"Failed to reach Tor check endpoint: {exc}")
            return False

    def new_circuit(self) -> bool:
        """Request a fresh Tor circuit via the control port (stem NEWNYM).

        Onion-service failures (dead rendezvous circuit, stuck introduction
        point) are common enough that retrying on the same circuit often
        just repeats the failure; this forces a new one. Best-effort: a
        missing/unauthenticated control port logs a warning and returns
        False instead of crashing the scrape run.
        """
        if self.control_port is None:
            logger.warning("No control_port configured, cannot rotate circuit")
            return False

        from stem import Signal
        from stem.control import Controller

        try:
            with Controller.from_port(port=self.control_port) as controller:
                if self.control_password is not None:
                    controller.authenticate(password=self.control_password)
                else:
                    controller.authenticate()
                controller.signal(Signal.NEWNYM)
                # Tor enforces a minimum gap between NEWNYM signals.
                time.sleep(controller.get_newnym_wait())
            logger.info("Requested new Tor circuit")
            return True
        except Exception as exc:
            logger.error(f"Failed to rotate Tor circuit: {exc}")
            return False
