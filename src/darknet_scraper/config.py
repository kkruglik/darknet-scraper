from pathlib import Path

from pydantic import BaseModel, Field, SecretStr


class AccountConfig(BaseModel):
    login: str
    password: SecretStr


class AuthConfig(BaseModel):
    captcha_key: SecretStr
    captcha_url: str
    max_captcha_retries: int = Field(default=5, ge=1)
    # One scraping worker per account.
    accounts: list[AccountConfig] = Field(min_length=1)


class ShopsConfig(BaseModel):
    start_url: str
    cache_dir: Path
    output_file: Path


class ReviewsConfig(BaseModel):
    input_file: Path
    cache_dir: Path
    output_dir: Path


class ProductsConfig(BaseModel):
    input_file: Path
    cache_dir: Path
    output_dir: Path


class ScraperConfig(BaseModel):
    scrape_shops: ShopsConfig
    scrape_reviews: ReviewsConfig
    scrape_products: ProductsConfig
    auth: AuthConfig
