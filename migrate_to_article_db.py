#!/usr/bin/env python3
"""Migrate digest callout blocks from the daily-digest Notion DB to a per-article Notion DB."""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Notion API helpers
# ---------------------------------------------------------------------------

def _notion_request(endpoint: str, token: str, payload: dict, method: str = "POST") -> dict:
    """Make a request to the Notion API."""
    url = f"https://api.notion.com/v1/{endpoint}"
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Content-Type", "application/json")
    req.add_header("Notion-Version", "2022-06-28")
    try:
        with urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        log.error("Notion API error (HTTP %d): %s", e.code, body[:500])
        raise
    except URLError as e:
        log.error("Notion API connection error: %s", e.reason)
        raise


def _notion_get(endpoint: str, token: str) -> dict:
    """Make a GET request to the Notion API."""
    url = f"https://api.notion.com/v1/{endpoint}"
    req = Request(url, method="GET")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Notion-Version", "2022-06-28")
    try:
        with urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        log.error("Notion API error (HTTP %d): %s", e.code, body[:500])
        raise
    except URLError as e:
        log.error("Notion API connection error: %s", e.reason)
        raise


# ---------------------------------------------------------------------------
# Notion data fetchers
# ---------------------------------------------------------------------------

def fetch_all_digest_pages(token: str, database_id: str) -> list[dict]:
    """Fetch all pages from the digest database with pagination."""
    pages = []
    has_more = True
    start_cursor = None

    while has_more:
        payload: dict = {"page_size": 100}
        if start_cursor:
            payload["start_cursor"] = start_cursor

        result = _notion_request(f"databases/{database_id}/query", token, payload)
        pages.extend(result.get("results", []))
        has_more = result.get("has_more", False)
        start_cursor = result.get("next_cursor")

    log.info("Fetched %d digest pages", len(pages))
    return pages


def fetch_page_blocks(token: str, page_id: str) -> list[dict]:
    """Fetch all child blocks of a page with pagination."""
    blocks = []
    has_more = True
    start_cursor = None

    while has_more:
        endpoint = f"blocks/{page_id}/children?page_size=100"
        if start_cursor:
            endpoint += f"&start_cursor={start_cursor}"

        result = _notion_get(endpoint, token)
        blocks.extend(result.get("results", []))
        has_more = result.get("has_more", False)
        start_cursor = result.get("next_cursor")

    return blocks


def fetch_existing_links(token: str, article_db_id: str) -> set[str]:
    """Fetch all existing article URLs from the article DB for deduplication."""
    links: set[str] = set()
    has_more = True
    start_cursor = None

    while has_more:
        payload: dict = {"page_size": 100}
        if start_cursor:
            payload["start_cursor"] = start_cursor

        result = _notion_request(f"databases/{article_db_id}/query", token, payload)
        for page in result.get("results", []):
            url = page.get("properties", {}).get("링크", {}).get("url")
            if url:
                links.add(url)
        has_more = result.get("has_more", False)
        start_cursor = result.get("next_cursor")

    log.info("Found %d existing articles in target DB", len(links))
    return links


# ---------------------------------------------------------------------------
# Callout block parser
# ---------------------------------------------------------------------------

def parse_callout_block(block: dict) -> dict | None:
    """Parse a callout block into article fields. Returns None for non-callout blocks."""
    if block.get("type") != "callout":
        return None

    rich_text = block.get("callout", {}).get("rich_text", [])
    if not rich_text:
        return None

    title = ""
    link = ""
    source = ""
    summary = ""
    tags: list[str] = []
    reason = ""

    for segment in rich_text:
        text_obj = segment.get("text", {})
        content = text_obj.get("content", "")
        link_obj = text_obj.get("link")
        annotations = segment.get("annotations", {})
        color = annotations.get("color", "default")
        is_bold = annotations.get("bold", False)
        is_italic = annotations.get("italic", False)

        # Bold + link → title + URL
        if is_bold and link_obj and link_obj.get("url") and not title:
            title = content
            link = link_obj["url"]
            continue

        # Italic + gray with 💬 → reason
        if is_italic and color == "gray":
            if "💬" in content:
                marker = content.find("💬")
                # Extract text after 💬 or "💬 읽어야 할 이유:"
                rest = content[marker:]
                if "읽어야 할 이유:" in rest:
                    reason = rest.split("읽어야 할 이유:", 1)[1].strip()
                else:
                    reason = rest.lstrip("💬").strip()
                continue
            # Italic + gray with "via" → source
            if "via " in content:
                source = content.split("via ", 1)[1].strip()
                continue
            continue

        # Gray (not italic, not bold) with · → tags
        if color == "gray" and not is_italic and not is_bold:
            stripped = content.strip()
            if stripped and " · " in stripped:
                tags = [t.strip() for t in stripped.split(" · ") if t.strip()]
            continue

        # Non-bold, non-italic, default/no color → summary
        if not is_bold and not is_italic and color in ("default", None, ""):
            text = content.strip()
            if text and not summary:
                summary = text

    if not title or not link:
        return None

    return {
        "title": title,
        "link": link,
        "source": source,
        "summary": summary,
        "tags": tags,
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# Payload builder
# ---------------------------------------------------------------------------

def build_article_payload(article_dict: dict, article_db_id: str, curation_date: str) -> dict:
    """Build a Notion page-create payload from a parsed article dict.

    score is set to 0 since it is not available in the digest callout blocks.
    """
    return {
        "parent": {"database_id": article_db_id},
        "properties": {
            "제목": {"title": [{"text": {"content": article_dict["title"]}}]},
            "링크": {"url": article_dict["link"]},
            "출처": {"select": {"name": article_dict["source"]}} if article_dict["source"] else {"select": None},
            "태그": {"multi_select": [{"name": t} for t in article_dict["tags"]]},
            "점수": {"number": 0},
            "요약": {"rich_text": [{"text": {"content": article_dict["summary"]}}]},
            "읽어야 할 이유": {"rich_text": [{"text": {"content": article_dict["reason"]}}]},
            "큐레이션일": {"date": {"start": curation_date}},
        },
    }


def _load_dotenv():
    """Load .env file from the project directory into os.environ."""
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


# ---------------------------------------------------------------------------
# Migration logic
# ---------------------------------------------------------------------------

def migrate(config_path: str, dry_run: bool = False) -> None:
    """Run the full migration from digest DB to article DB."""
    config_file = Path(config_path)
    if not config_file.exists():
        log.error("Config file not found: %s", config_path)
        sys.exit(1)

    with open(config_file, encoding="utf-8") as f:
        config = json.load(f)

    notion = config.get("notion", {})
    token = os.environ.get("NOTION_TOKEN") or notion.get("token")
    digest_db_id = os.environ.get("NOTION_DATABASE_ID") or notion.get("database_id")
    article_db_id = notion.get("article_database_id")

    if not token or not digest_db_id or not article_db_id:
        log.error("NOTION_TOKEN, digest database_id, and article_database_id are all required")
        sys.exit(1)

    # Fetch all digest pages
    digest_pages = fetch_all_digest_pages(token, digest_db_id)

    # Fetch existing links for dedup
    existing_links = fetch_existing_links(token, article_db_id)

    created = 0
    skipped_dup = 0
    skipped_parse = 0

    for page in digest_pages:
        page_id = page["id"]

        # Extract curation date from page properties
        date_prop = page.get("properties", {}).get("작성일", {}).get("date", {})
        curation_date = date_prop.get("start", "") if date_prop else ""
        if not curation_date:
            # Fallback to page created_time
            created_time = page.get("created_time", "")
            curation_date = created_time[:10] if created_time else ""

        # Fetch child blocks
        blocks = fetch_page_blocks(token, page_id)

        for block in blocks:
            article = parse_callout_block(block)
            if article is None:
                continue

            if article["link"] in existing_links:
                skipped_dup += 1
                continue

            if not article["title"]:
                skipped_parse += 1
                continue

            if dry_run:
                log.info("[DRY-RUN] Would create: %s", article["title"])
                created += 1
                existing_links.add(article["link"])
                continue

            payload = build_article_payload(article, article_db_id, curation_date)
            try:
                _notion_request("pages", token, payload)
                created += 1
                existing_links.add(article["link"])
                log.info("Created: %s", article["title"])
                # Rate limiting: Notion API allows ~3 requests/sec
                time.sleep(0.35)
            except Exception:
                log.exception("Failed to create page for: %s", article["title"])

    log.info(
        "Migration complete — created: %d, skipped (dup): %d, skipped (parse): %d",
        created, skipped_dup, skipped_parse,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate digest callout blocks to per-article Notion DB",
    )
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to config.json (default: config.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and log without creating Notion pages",
    )
    args = parser.parse_args()
    migrate(args.config, args.dry_run)


if __name__ == "__main__":
    _load_dotenv()
    main()
