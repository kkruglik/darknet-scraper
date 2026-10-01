import logging
import time

import httpx2 as httpx

logger = logging.getLogger(__name__)

TWO_CAPTCHA_CREATE_TASK_URL = "https://api.2captcha.com/createTask"
TWO_CAPTCHA_GET_RESULT_URL = "https://api.2captcha.com/getTaskResult"


class AuthError(Exception):
    """Raised when the login flow cannot complete."""


def solve_image_captcha(
    api_key: str,
    image_b64: str,
    *,
    numeric: int = 0,
    case: bool = False,
    min_length: int = 0,
    max_length: int = 0,
    language_pool: str = "en",
    comment: str = "",
    poll_interval: float = 3.0,
    max_wait: float = 90.0,
) -> str:
    """Solve an image captcha via 2captcha's ImageToTextTask API.

    Standalone: takes just the image and solving hints, talks to 2captcha
    over its own plain connection (no reason to route a clearnet API call
    through the scraper's Tor client), and returns just the answer text.
    """
    task = {
        "type": "ImageToTextTask",
        "body": image_b64,
        "numeric": numeric,
        "case": case,
        "minLength": min_length,
        "maxLength": max_length,
    }
    if comment:
        task["comment"] = comment

    logger.info(
        f"2captcha: submitting image (languagePool={language_pool!r}, "
        f"numeric={numeric}, case={case})"
    )
    with httpx.Client(timeout=30) as client:
        resp = client.post(
            TWO_CAPTCHA_CREATE_TASK_URL,
            json={"clientKey": api_key, "task": task, "languagePool": language_pool},
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("errorId"):
            raise AuthError(
                f"2captcha createTask failed: {data.get('errorCode')} "
                f"{data.get('errorDescription')}"
            )
        task_id = data["taskId"]
        logger.info(f"2captcha: task {task_id} created, polling for a result")

        deadline = time.monotonic() + max_wait
        poll_count = 0
        while time.monotonic() < deadline:
            time.sleep(poll_interval)
            poll_count += 1
            resp = client.post(
                TWO_CAPTCHA_GET_RESULT_URL,
                json={"clientKey": api_key, "taskId": task_id},
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("errorId"):
                raise AuthError(
                    f"2captcha getTaskResult failed: {data.get('errorCode')} "
                    f"{data.get('errorDescription')}"
                )
            if data["status"] == "ready":
                logger.info(
                    f"2captcha: task {task_id} solved after {poll_count} poll(s)"
                )
                return data["solution"]["text"]
            logger.info(
                f"2captcha: task {task_id} still processing "
                f"(poll {poll_count}, {deadline - time.monotonic():.0f}s left)"
            )

    raise AuthError("2captcha did not return a result within max_wait")
