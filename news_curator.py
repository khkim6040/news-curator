#!/usr/bin/env python3
"""Daily RSS News Curator — LLM-powered curation using Claude CLI."""

import json
import logging
import os
import re
import sqlite3
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from html import unescape
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
from xml.etree import ElementTree as ET

BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config.json"
DB_PATH = BASE_DIR / "seen.db"
LOG_PATH = BASE_DIR / "curator.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class Article:
    title: str
    link: str
    description: str
    pub_date: str
    source: str
    categories: list[str] = field(default_factory=list)
    score: int = 0
    summary: str = ""
    tags: list[str] = field(default_factory=list)
    reason: str = ""


def strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def parse_pub_date(date_str: str) -> datetime | None:
    """Parse RSS/Atom date strings into timezone-aware datetime."""
    if not date_str or not date_str.strip():
        return None
    date_str = date_str.strip()
    # RFC 822 (RSS 2.0): "Sat, 22 Feb 2026 10:30:00 +0000"
    try:
        return parsedate_to_datetime(date_str)
    except Exception:
        pass
    # ISO 8601 (Atom): "2026-02-22T10:30:00Z"
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(date_str, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# SQLite dedup DB
# ---------------------------------------------------------------------------

def init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS seen_articles (
            url   TEXT PRIMARY KEY,
            title TEXT,
            source TEXT,
            seen_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.commit()
    return conn


def is_seen(conn: sqlite3.Connection, url: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM seen_articles WHERE url = ?", (url,)
    ).fetchone() is not None


def mark_seen(conn: sqlite3.Connection, articles: list[Article]):
    conn.executemany(
        "INSERT OR IGNORE INTO seen_articles (url, title, source) VALUES (?, ?, ?)",
        [(a.link, a.title, a.source) for a in articles],
    )
    conn.commit()


def cleanup_db(conn: sqlite3.Connection, days: int = 30):
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    n = conn.execute("DELETE FROM seen_articles WHERE seen_at < ?", (cutoff,)).rowcount
    conn.commit()
    if n:
        log.info("Cleaned up %d old DB entries", n)


# ---------------------------------------------------------------------------
# Feed fetching & parsing
# ---------------------------------------------------------------------------

def fetch_feed(feed_cfg: dict) -> str | None:
    url = feed_cfg["url"]
    headers = feed_cfg.get("headers", {})
    req = Request(url)
    for k, v in headers.items():
        req.add_header(k, v)
    try:
        with urlopen(req, timeout=15) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except HTTPError as e:
        if e.code == 403:
            log.warning("Feed %s returned 403 — skipping", feed_cfg["name"])
        else:
            log.error("Failed to fetch %s: HTTP %d", feed_cfg["name"], e.code)
        return None
    except URLError as e:
        log.error("Failed to fetch %s: %s", feed_cfg["name"], e.reason)
        return None


def parse_feed(xml_text: str, source_name: str) -> list[Article]:
    articles = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        log.error("XML parse error for %s: %s", source_name, e)
        return []

    # Detect Atom vs RSS 2.0
    tag = root.tag.lower()
    if "feed" in tag:
        # Atom format
        ns = {"a": "http://www.w3.org/2005/Atom"}
        for entry in root.findall("a:entry", ns):
            title = (entry.findtext("a:title", "", ns) or "").strip()
            link_el = entry.find("a:link[@rel='alternate']", ns)
            if link_el is None:
                link_el = entry.find("a:link", ns)
            link = link_el.get("href", "") if link_el is not None else ""
            desc = strip_html(entry.findtext("a:content", "", ns)
                              or entry.findtext("a:summary", "", ns) or "")
            pub_date = entry.findtext("a:published", "", ns) or entry.findtext("a:updated", "", ns) or ""
            categories = [c.get("term", "") for c in entry.findall("a:category", ns) if c.get("term")]

            if not title or not link:
                continue
            articles.append(Article(
                title=title, link=link, description=desc[:500],
                pub_date=pub_date, source=source_name, categories=categories,
            ))
    else:
        # RSS 2.0 format
        for item in root.iter("item"):
            title = item.findtext("title", "").strip()
            link = item.findtext("link", "").strip()
            desc = strip_html(item.findtext("description", ""))
            pub_date = item.findtext("pubDate", "")
            categories = [c.text for c in item.findall("category") if c.text]

            if not title or not link:
                continue
            articles.append(Article(
                title=title, link=link, description=desc[:500],
                pub_date=pub_date, source=source_name, categories=categories,
            ))
    return articles


# ---------------------------------------------------------------------------
# Claude CLI curation
# ---------------------------------------------------------------------------

def _build_prompt(articles: list[Article], config: dict) -> str:
    """Build a single prompt combining system instructions + article list."""
    cur = config["curator"]
    interests = "\n".join(f"- {i}" for i in cur["interests"])
    min_score = config["scoring"]["min_score"]
    max_articles = cur.get("max_articles", 20)

    # Numbered article list
    lines = []
    for i, a in enumerate(articles):
        cats = f" [{', '.join(a.categories)}]" if a.categories else ""
        date_str = ""
        if a.pub_date:
            dt = parse_pub_date(a.pub_date)
            date_str = f"\n발행일: {dt.strftime('%Y-%m-%d')}" if dt else f"\n발행일: {a.pub_date}"
        lines.append(
            f"[{i}] [{a.source}]{cats}\n"
            f"제목: {a.title}{date_str}\n"
            f"요약: {a.description[:300]}"
        )
    article_text = "\n\n".join(lines)

    return f"""당신은 기술 뉴스 큐레이터입니다.
사용자의 관심사와 역할에 맞는 기사를 선별하고, 각 기사의 관련성을 평가합니다.

사용자 프로필:
{cur["persona"]}

관심 분야:
{interests}

## 채점 루브릭 (반드시 준수)

9-10점 [필독]: 아키텍처 설계, 코드 레벨 해결책, 대규모 트래픽 처리 경험이 상세히 기술된 포스트. 실무에 바로 적용 가능한 깊이.
7-8점 [추천]: 새로운 백엔드/AI 기술 동향, 실제 도입 사례, 깊이 있는 튜토리얼. 새로운 인사이트 제공.
4-6점 [참고]: 가볍게 읽을 만한 IT 업계 동향 및 개념 소개. 배경 지식으로 유용.
1-3점 [제외]: 프로필과 무관하거나 기술적 깊이가 없는 기사. 아래 제외 기준에 해당.

## 제외 기준 (즉시 3점 이하 처리)

다음에 해당하는 기사는 즉시 3점 이하로 평가하고 결과에서 제외할 것:
- 주가 동향, 시세 정보, 투자 전망
- 단순 인사 동정, 경영진 변경
- 암호화폐/NFT 시세 및 투기 관련
- 마케팅성 보도자료, 제품 출시 홍보
- 비기술적인 스타트업 투자 유치 기사
- 기술적 깊이 없이 용어만 나열한 기사

## 규칙

1. 각 기사를 위 루브릭에 따라 1-10점으로 평가
2. {min_score}점 이상인 기사만 선택
3. 최대 {max_articles}개까지 선택
4. 요약: 기사의 '핵심 문제(Problem)'와 도입된 '기술적 해결책(Solution)'을 중심으로 3문장 이내로 한국어 요약
5. 한국어 태그 1-3개 부여
6. 선택 이유를 한 줄로 작성

반드시 아래 JSON 배열 형식으로만 응답하세요 (다른 텍스트 없이):
[
  {{
    "index": 0,
    "score": 8,
    "summary": "한국어 요약 — 문제와 해결책 중심 3문장 이내",
    "tags": ["태그1", "태그2"],
    "reason": "선택 이유 한 줄"
  }}
]

관심 분야와 전혀 관련 없으면 빈 배열 []을 반환하세요.

---

다음 {len(articles)}개 기사를 큐레이션해주세요:

{article_text}"""


def curate_with_claude(articles: list[Article], config: dict) -> list[Article]:
    """Call Claude CLI to curate articles."""
    cur = config["curator"]
    prompt = _build_prompt(articles, config)
    model = cur.get("model")

    log.info("Calling Claude CLI (%s) with %d articles …", model or "default", len(articles))

    try:
        cmd = ["claude", "-p", prompt, "--output-format", "text", "--max-turns", "4"]
        if model:
            cmd.extend(["--model", model])
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
    except subprocess.TimeoutExpired:
        log.error("Claude CLI timed out")
        return []
    except FileNotFoundError:
        log.error("'claude' CLI not found. Install: npm install -g @anthropic-ai/claude-code")
        return []

    if result.returncode != 0:
        log.error("Claude CLI error (exit %d): %s", result.returncode, result.stderr[:500])
        return []

    text = result.stdout.strip()
    if not text:
        log.error("Empty response from Claude CLI")
        return []

    # Strip markdown code fences if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    # Try to extract JSON array from response
    try:
        selections = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                selections = json.loads(match.group())
            except json.JSONDecodeError as e:
                log.error("Failed to parse Claude JSON: %s\nResponse: %.500s", e, text)
                return []
        else:
            log.error("No JSON array in Claude response: %.500s", text)
            return []

    # Map back to articles
    curated = []
    for sel in selections:
        idx = sel.get("index")
        if idx is None or not (0 <= idx < len(articles)):
            continue
        a = articles[idx]
        a.score = sel.get("score", 0)
        a.summary = sel.get("summary", "")
        a.tags = sel.get("tags", [])
        a.reason = sel.get("reason", "")
        curated.append(a)

    curated.sort(key=lambda a: a.score, reverse=True)
    log.info("Claude selected %d / %d articles", len(curated), len(articles))
    return curated


# ---------------------------------------------------------------------------
# Notion uploader
# ---------------------------------------------------------------------------

def _score_emoji(score: int) -> str:
    if score >= 8:
        return "\U0001f525"  # 🔥
    if score >= 6:
        return "\u2b50"      # ⭐
    if score >= 4:
        return "\U0001f4a1"  # 💡
    return "\u2796"          # ➖


def _score_color(score: int) -> str:
    if score >= 8:
        return "red"
    if score >= 6:
        return "orange"
    if score >= 4:
        return "yellow"
    return "gray"


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


def _build_article_blocks(article: Article) -> list[dict]:
    """Build Notion blocks for a single article."""
    emoji = _score_emoji(article.score)
    color = _score_color(article.score)
    tags_text = " · ".join(article.tags) if article.tags else ""

    # Callout block with article info
    rich_text = [
        {
            "type": "text",
            "text": {"content": article.title, "link": {"url": article.link}},
            "annotations": {"bold": True},
        },
        {
            "type": "text",
            "text": {"content": f"  [{article.score}/10]", "link": None},
            "annotations": {"bold": True, "color": color},
        },
        {
            "type": "text",
            "text": {"content": f"  via {article.source}", "link": None},
            "annotations": {"color": "gray", "italic": True},
        },
    ]
    if tags_text:
        rich_text.append(
            {"type": "text", "text": {"content": f"\n{tags_text}", "link": None},
             "annotations": {"color": "gray"}}
        )
    rich_text.append(
        {"type": "text", "text": {"content": f"\n{article.summary}", "link": None}}
    )
    if article.reason:
        rich_text.append(
            {"type": "text", "text": {"content": f"\n💬 {article.reason}", "link": None},
             "annotations": {"italic": True, "color": "gray"}}
        )

    return [
        {
            "object": "block",
            "type": "callout",
            "callout": {
                "rich_text": rich_text,
                "icon": {"type": "emoji", "emoji": emoji},
                "color": "gray_background",
            },
        },
    ]


def _build_notion_blocks(curated: list[Article], errors: list[str]) -> list[dict]:
    """Build all Notion content blocks for the digest page."""
    today = datetime.now().strftime("%Y년 %m월 %d일")
    blocks: list[dict] = []

    # Header paragraph
    blocks.append({
        "object": "block",
        "type": "paragraph",
        "paragraph": {
            "rich_text": [{
                "type": "text",
                "text": {"content": f"총 {len(curated)}건의 관련 기사가 선별되었습니다 · {today}"},
                "annotations": {"color": "gray"},
            }],
        },
    })
    blocks.append({"object": "block", "type": "divider", "divider": {}})

    # Group by score tier
    tiers = [
        ("\U0001f525 필독", 8, 10),
        ("\u2b50 추천", 6, 7),
        ("\U0001f4a1 참고", 4, 5),
    ]

    for tier_label, lo, hi in tiers:
        arts = [a for a in curated if lo <= a.score <= hi]
        if not arts:
            continue
        blocks.append({
            "object": "block",
            "type": "heading_2",
            "heading_2": {
                "rich_text": [
                    {"type": "text", "text": {"content": f"{tier_label} "}},
                    {"type": "text", "text": {"content": f"({len(arts)}건)"},
                     "annotations": {"color": "gray"}},
                ],
            },
        })

        for a in arts:
            blocks.extend(_build_article_blocks(a))

        blocks.append({"object": "block", "type": "divider", "divider": {}})

    # Errors section
    if errors:
        blocks.append({
            "object": "block",
            "type": "callout",
            "callout": {
                "rich_text": [{
                    "type": "text",
                    "text": {"content": "Fetch Errors:\n" + "\n".join(f"• {e}" for e in errors)},
                }],
                "icon": {"type": "emoji", "emoji": "\u26a0\ufe0f"},
                "color": "red_background",
            },
        })

    # Footer
    blocks.append({
        "object": "block",
        "type": "paragraph",
        "paragraph": {
            "rich_text": [{
                "type": "text",
                "text": {"content": "Curated by Claude AI · news-curator"},
                "annotations": {"color": "gray", "italic": True},
            }],
        },
    })

    return blocks


def upload_to_notion(curated: list[Article], errors: list[str], config: dict):
    """Create a Notion database page with the curated digest."""
    ncfg = config["notion"]
    token = ncfg["token"]
    database_id = ncfg["database_id"]
    today = datetime.now().strftime("%Y-%m-%d")
    title = f"{today} Daily Tech Digest"

    blocks = _build_notion_blocks(curated, errors)

    # Create page with first batch of blocks (max 100 per request)
    page_payload = {
        "parent": {"database_id": database_id},
        "icon": {"type": "emoji", "emoji": "\U0001f4f0"},
        "properties": {
            "이름": {"title": [{"text": {"content": title}}]},
            "작성일": {"date": {"start": today}},
        },
        "children": blocks[:100],
    }

    log.info("Creating Notion page: %s (%d blocks)", title, len(blocks))
    page = _notion_request("pages", token, page_payload)
    page_id = page["id"]
    page_url = page["url"]

    # Append remaining blocks in batches of 100
    remaining = blocks[100:]
    while remaining:
        batch = remaining[:100]
        remaining = remaining[100:]
        _notion_request(
            f"blocks/{page_id}/children", token,
            {"children": batch}, method="PATCH",
        )
        log.info("Appended %d extra blocks to page", len(batch))

    log.info("Notion page created: %s", page_url)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("=== News Curator started ===")
    config = load_config()
    conn = init_db()
    errors: list[str] = []

    # 1. Fetch & parse all feeds
    all_articles: list[Article] = []
    max_per = config["scoring"].get("max_articles_per_source", 15)

    for feed in config["feeds"]:
        name = feed["name"]
        xml = fetch_feed(feed)
        if xml is None:
            errors.append(f"{name}: fetch failed")
            continue

        parsed = parse_feed(xml, name)
        log.info("Parsed %d articles from %s", len(parsed), name)

        new = [a for a in parsed if not is_seen(conn, a.link)]
        log.info("New from %s: %d / %d", name, len(new), len(parsed))
        all_articles.extend(new[:max_per])

    # Filter out articles older than max_age_days
    max_age_days = config["scoring"].get("max_age_days", 3)
    cutoff_dt = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    before_filter = len(all_articles)
    filtered = []
    for a in all_articles:
        dt = parse_pub_date(a.pub_date)
        if dt is None:
            filtered.append(a)  # keep articles with unparseable dates
        elif dt >= cutoff_dt:
            filtered.append(a)
    all_articles = filtered
    if before_filter != len(all_articles):
        log.info("Filtered out %d old articles (> %d days)", before_filter - len(all_articles), max_age_days)

    if not all_articles:
        log.info("No new articles. Done.")
        cleanup_db(conn, config["db"].get("retention_days", 30))
        conn.close()
        return

    log.info("Total new articles for curation: %d", len(all_articles))

    # 2. LLM curation
    curated = curate_with_claude(all_articles, config)

    # Mark ALL fetched articles as seen (avoid re-processing)
    mark_seen(conn, all_articles)

    if not curated:
        log.info("No articles passed curation. Done.")
        cleanup_db(conn, config["db"].get("retention_days", 30))
        conn.close()
        return

    # 3. Upload to Notion
    upload_to_notion(curated, errors, config)

    # 4. Cleanup
    cleanup_db(conn, config["db"].get("retention_days", 30))
    conn.close()
    log.info("=== News Curator finished ===")


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    main()
