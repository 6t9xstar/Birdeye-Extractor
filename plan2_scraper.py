from __future__ import annotations

import argparse
import asyncio
import csv
from collections import OrderedDict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import shutil
import time
from typing import AsyncIterator, Iterable

from extractors.config import ScraperConfig
from extractors.http import AsyncHTTPClient
from extractors.parsing import ProfileData, extract_profile_links, parse_profile_html

LISTING_MARKERS = ("/d/categories/", "/categories/contractors", "?page=")


def read_seed_file(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip() and not line.startswith("#")]


def partition_seed_urls(urls: Iterable[str]) -> tuple[list[str], list[str]]:
    listings: list[str] = []
    profiles: list[str] = []
    for url in urls:
        lowered = url.lower()
        if any(marker in lowered for marker in LISTING_MARKERS):
            listings.append(url)
        else:
            profiles.append(url)
    return listings, profiles


async def crawl_listing_pages(
    client: AsyncHTTPClient,
    listing_urls: list[str],
    concurrency: int,
) -> dict[str, str]:
    semaphore = asyncio.Semaphore(concurrency)
    discovered: OrderedDict[str, str] = OrderedDict()

    async def _fetch(url: str) -> None:
        async with semaphore:
            try:
                html = await client.get_text(url)
            except Exception as exc:  # pragma: no cover - network failure path
                print(f"[listing] ERROR {url}: {exc}")
                return
        links = extract_profile_links(html)
        for link in links:
            if link not in discovered:
                discovered[link] = url
        print(f"[listing] {url} -> {len(links)} profiles (total {len(discovered)})")

    await asyncio.gather(*(_fetch(url) for url in listing_urls))
    return discovered


async def crawl_profile_pages(
    client: AsyncHTTPClient,
    profile_sources: dict[str, str],
    concurrency: int,
) -> AsyncIterator[tuple[str, ProfileData, str]]:
    semaphore = asyncio.Semaphore(concurrency)

    async def _fetch(url: str, source: str) -> tuple[str, ProfileData, str] | None:
        async with semaphore:
            try:
                html = await client.get_text(url)
            except Exception as exc:  # pragma: no cover - network failure path
                print(f"[profile] ERROR {url}: {exc}")
                return None
        data = parse_profile_html(html)
        print(f"[profile] scraped {url}")
        return url, data, source

    tasks = [
        asyncio.create_task(_fetch(url, src)) for url, src in profile_sources.items()
    ]
    if not tasks:
        return

    for task in asyncio.as_completed(tasks):
        result = await task
        if result:
            yield result


def profile_row(url: str, data: ProfileData, source: str) -> dict[str, str | float | int | None]:
    row = {
        "profile_url": url,
        "source_page": source,
        "last_seen": datetime.utcnow().isoformat(),
    }
    row.update(asdict(data))
    return row


class CSVStreamer:
    columns = [
        "profile_url",
        "name",
        "rating",
        "review_count",
        "phone",
        "address",
        "website_url",
        "source_page",
        "last_seen",
    ]

    def __init__(
        self,
        output_path: Path,
        *,
        flush_interval_rows: int = 100,
        flush_interval_seconds: int = 60,
        autosave_seconds: int = 300,
    ) -> None:
        self.output_path = output_path
        self.flush_interval_rows = max(1, flush_interval_rows)
        self.flush_interval_seconds = max(1, flush_interval_seconds)
        self.autosave_seconds = max(1, autosave_seconds)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = output_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=self.columns)
        self._writer.writeheader()
        self.row_count = 0
        now = time.monotonic()
        self._last_flush = now
        self._last_autosave = now
        self._autosave_path = output_path.with_name(
            f"{output_path.stem}.autosave{output_path.suffix}"
        )

    def write_row(self, row: dict[str, str | float | int | None]) -> None:
        self._writer.writerow(row)
        self.row_count += 1
        self._maybe_flush()
        self._maybe_autosave()

    def close(self) -> None:
        self._handle.flush()
        self._copy_autosave(force=True)
        self._handle.close()

    def _maybe_flush(self) -> None:
        now = time.monotonic()
        should_flush = False
        if self.row_count % self.flush_interval_rows == 0:
            should_flush = True
        if now - self._last_flush >= self.flush_interval_seconds:
            should_flush = True
        if should_flush:
            self._handle.flush()
            self._last_flush = now

    def _maybe_autosave(self) -> None:
        now = time.monotonic()
        if now - self._last_autosave >= self.autosave_seconds:
            self._copy_autosave()
            self._last_autosave = now

    def _copy_autosave(self, force: bool = False) -> None:
        self._handle.flush()
        try:
            shutil.copyfile(self.output_path, self._autosave_path)
        except OSError:
            if force:
                raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Birdeye contractor extractor (Plan 2)")
    parser.add_argument("--input", type=Path, default=Path("birdeye.txt"), help="Seed file with URLs")
    parser.add_argument("--output", type=Path, default=Path("Plan 2/output.csv"), help="CSV to write")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--flush-interval", type=int, default=200, help="Rows per CSV flush")
    parser.add_argument("--flush-seconds", type=int, default=60, help="Max seconds between flushes")
    parser.add_argument("--autosave-seconds", type=int, default=300, help="Seconds between autosave copies")
    parser.add_argument(
        "--mode",
        choices=["listing-only", "profile-only", "full"],
        default="full",
        help="Choose which pipeline stages to run",
    )
    return parser.parse_args()


async def run_async(args: argparse.Namespace) -> None:
    config = ScraperConfig(concurrency=args.concurrency, timeout=args.timeout)
    client = AsyncHTTPClient(config)
    streamer: CSVStreamer | None = None
    total_rows = 0
    try:
        seeds = read_seed_file(args.input)
        listing_seeds, profile_seeds = partition_seed_urls(seeds)

        profile_sources: OrderedDict[str, str] = OrderedDict((url, "manual") for url in profile_seeds)

        if args.mode in ("listing-only", "full") and listing_seeds:
            discovered = await crawl_listing_pages(client, listing_seeds, args.concurrency)
            for url, source in discovered.items():
                profile_sources.setdefault(url, source)
            if args.mode == "listing-only":
                print(f"[done] discovered {len(profile_sources)} profile URLs")
                return

        if args.mode in ("profile-only", "full") and profile_sources:
            streamer = CSVStreamer(
                args.output,
                flush_interval_rows=args.flush_interval,
                flush_interval_seconds=args.flush_seconds,
                autosave_seconds=args.autosave_seconds,
            )
            async for url, data, source in crawl_profile_pages(
                client, profile_sources, args.concurrency
            ):
                row = profile_row(url, data, source)
                streamer.write_row(row)
                total_rows += 1
            print(f"[done] wrote {total_rows} rows to {args.output}")
        else:
            print("[done] nothing to scrape")
    finally:
        if streamer:
            streamer.close()
        await client.close()


def main() -> None:
    args = parse_args()
    asyncio.run(run_async(args))


if __name__ == "__main__":
    main()
