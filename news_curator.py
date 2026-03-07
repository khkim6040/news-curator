#!/usr/bin/env python3
"""Daily RSS News Curator — LLM-powered curation using Claude CLI."""

import argparse
import json
import logging
import os
import re
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from html import unescape
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
from xml.etree import ElementTree as ET

BASE_DIR = Path(__file__).parent
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Daily RSS News Curator")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="큐레이션만 수행하고 Notion 업로드/DB 기록을 건너뜀",
    )
    parser.add_argument(
        "--config", type=Path, default=BASE_DIR / "config.json",
        help="config.json 경로 (기본: %(default)s)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="DEBUG 레벨 로그 출력",
    )
    return parser.parse_args()


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
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z",
                 "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%d"):
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
    conn.execute("""
        CREATE TABLE IF NOT EXISTS run_history (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            run_at        TEXT,
            total_fetched INTEGER,
            total_new     INTEGER,
            total_curated INTEGER,
            error_count   INTEGER,
            source_stats  TEXT
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


def record_run(conn: sqlite3.Connection, total_fetched: int, total_new: int,
               total_curated: int, error_count: int, source_stats: dict):
    conn.execute(
        "INSERT INTO run_history (run_at, total_fetched, total_new, total_curated, error_count, source_stats)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (datetime.now().isoformat(), total_fetched, total_new, total_curated,
         error_count, json.dumps(source_stats, ensure_ascii=False)),
    )
    conn.commit()
    log.info("Run recorded: fetched=%d, new=%d, curated=%d, errors=%d",
             total_fetched, total_new, total_curated, error_count)


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

    return f"""당신은 10년 차 시니어 핀테크 백엔드 엔지니어이자 테크 리드입니다.
단순한 '신기술 소개'보다는 시스템 안정성, 대용량 트래픽 처리, 데이터 무결성(Data Integrity), 보안에 초점을 맞춰 기사를 평가합니다.

사용자 프로필:
{cur["persona"]}

관심 분야:
{interests}

## 채점 루브릭 (Fintech Backend Specialized)

9-10점 [필독 - 아키텍처/Deep Dive]:
- 금융 도메인(원장 설계, 트랜잭션, 정합성)과 직접 관련된 심층 기술 아티클.
- 대규모 트래픽 분산 처리, 장애 대응(Failover), 데이터베이스 락킹/격리 수준에 대한 실무적 경험 공유.
- 핀테크 보안 규제(망분리, 개인정보 암호화)와 관련된 기술적 해법.

7-8점 [추천 - Best Practice]:
- Java/Kotlin/Spring Boot, DB(MySQL, Redis), Kafka 등 백엔드 코어 기술의 성능 최적화 사례.
- MSA 전환, CI/CD 파이프라인 개선 등 생산성 향상 사례.
- AI/LLM의 실무 적용 사례 (개발 생산성, 코드 리뷰 자동화 등).

4-6점 [참고 - 일반 동향]:
- 클라우드 벤더(AWS/GCP)의 신규 기능 중 백엔드와 연관된 것.
- 일반적인 개발 방법론이나 소프트 스킬.

1-3점 [제외]:
- 프론트엔드(React, CSS 등), 모바일 UI/UX 전용 기사.
- 기술적 내용이 없는 단순 제품 홍보, 투자 유치, 경영진 인터뷰.
- 주가 변동, 암호화폐 시세 등 '투자 정보'.
- 기술적 깊이 없이 용어만 나열한 기사.

## 규칙

1. 각 기사를 위 루브릭에 따라 1-10점으로 평가
2. {min_score}점 이상인 기사만 선택
3. 최대 {max_articles}개까지 선택
4. 요약: 기사의 핵심 내용을 자연스러운 서술형 한국어 2-3문장으로 정리. 레이블이나 화살표 없이 읽기 좋게 작성. 미사여구 금지.
5. 한국어 태그 1-3개 부여
6. reason: "왜 내가 이걸 읽어야 하는가"를 구체적으로 작성. 현재 업무에 어떻게 적용 가능한지 한 줄로 서술.

반드시 아래 JSON 배열 형식으로만 응답하세요 (다른 텍스트 없이):
[
  {{
    "index": 0,
    "score": 8,
    "summary": "Kafka 클러스터를 3→5노드로 확장하면서 파티션 재배치 시 발생한 컨슈머 랙 급증 문제를 다룬다. 롤링 리밸런싱과 스로틀링 설정으로 무중단 확장에 성공했다.",
    "tags": ["태그1", "태그2"],
    "reason": "현재 결제 모듈 리팩토링에 참고할 수 있는 DB Deadlock 해결 패턴 포함"
  }}
]

{min_score}점 이상인 기사가 없으면 빈 배열 []을 반환하세요.

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
    except Exception as e:
        log.error("Claude CLI unexpected error: %s", e)
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
        # Find the first balanced [...] using bracket depth tracking
        selections = None
        start = text.find('[')
        if start >= 0:
            depth = 0
            for i in range(start, len(text)):
                if text[i] == '[':
                    depth += 1
                elif text[i] == ']':
                    depth -= 1
                    if depth == 0:
                        try:
                            selections = json.loads(text[start:i+1])
                        except json.JSONDecodeError as e:
                            log.error("Failed to parse Claude JSON: %s\nResponse: %.500s", e, text)
                        break
        if selections is None:
            log.error("No JSON array in Claude response: %.500s", text)
            return []

    # Map back to articles
    curated = []
    for sel in selections:
        if isinstance(sel, dict):
            idx = sel.get("index")
            score = sel.get("score", 0)
            summary = sel.get("summary", "")
            tags = sel.get("tags", [])
            reason = sel.get("reason", "")
        elif isinstance(sel, (int, float)):
            idx = int(sel)
            score, summary, tags, reason = 0, "", [], ""
        else:
            continue
        if idx is None or not (0 <= idx < len(articles)):
            continue
        a = articles[idx]
        a.score = score
        a.summary = summary
        a.tags = tags
        a.reason = reason
        curated.append(a)

    curated.sort(key=lambda a: a.score, reverse=True)
    log.info("Claude selected %d / %d articles", len(curated), len(articles))
    return curated


# ---------------------------------------------------------------------------
# Notion uploader
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


def _estimate_reading_time(description: str) -> str:
    """Estimate reading time based on description length."""
    # description is truncated to 500 chars; estimate full article from ratio
    # Average Korean reading speed: ~500 chars/min, English: ~200 words/min
    char_count = len(description)
    # Rough estimate: description is ~10-20% of full article
    estimated_full = char_count * 7
    minutes = max(1, estimated_full // 500)
    if minutes <= 3:
        return "~3분"
    elif minutes <= 7:
        return "~5분"
    elif minutes <= 12:
        return "~10분"
    else:
        return "10분+"


def _build_article_blocks(article: Article) -> list[dict]:
    """Build Notion blocks for a single article."""
    tags_text = " · ".join(article.tags) if article.tags else ""
    reading_time = _estimate_reading_time(article.description)

    # Callout block with article info
    rich_text = [
        {
            "type": "text",
            "text": {"content": article.title, "link": {"url": article.link}},
            "annotations": {"bold": True},
        },
        {
            "type": "text",
            "text": {"content": f"  {reading_time}  via {article.source}", "link": None},
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
                "icon": {"type": "emoji", "emoji": "\U0001f4cc"},
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

    # No articles message
    if not curated:
        blocks.append({
            "object": "block",
            "type": "callout",
            "callout": {
                "rich_text": [{
                    "type": "text",
                    "text": {"content": "오늘은 추천 기준을 충족하는 기사가 없습니다.\n퀄리티 높은 글만 엄선하고 있으니, 오늘은 편하게 쉬어가세요!"},
                }],
                "icon": {"type": "emoji", "emoji": "☕"},
                "color": "gray_background",
            },
        })
        blocks.append({"object": "block", "type": "divider", "divider": {}})

    # List articles (already sorted by score descending)
    for a in curated:
        blocks.extend(_build_article_blocks(a))

    if curated:
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

def _print_dry_run(curated: list[Article], errors: list[str]):
    """Print curation results to stdout for dry-run mode."""
    print(f"\n{'='*60}")
    print(f"  DRY-RUN 결과: {len(curated)}건 선별")
    print(f"{'='*60}\n")
    for i, a in enumerate(curated, 1):
        tags = " · ".join(a.tags) if a.tags else ""
        print(f"[{i}] ({a.score}점) {a.title}")
        print(f"    출처: {a.source}  |  {a.link}")
        if tags:
            print(f"    태그: {tags}")
        print(f"    요약: {a.summary}")
        if a.reason:
            print(f"    사유: {a.reason}")
        print()
    if errors:
        print(f"--- 피드 오류 ({len(errors)}건) ---")
        for e in errors:
            print(f"  • {e}")
        print()


def _print_run_summary(*, elapsed: float, total_feeds: int, total_fetched: int,
                       total_new: int, total_curated: int, errors: list[str],
                       source_stats: dict[str, dict], dry_run: bool):
    """Print structured run summary for launchd log analysis."""
    status = "DRY_RUN" if dry_run else ("OK" if not errors else "PARTIAL_FAIL")
    failed_feeds = [n for n, s in source_stats.items() if s.get("error")]
    top_sources = sorted(
        [(n, s["curated"]) for n, s in source_stats.items() if s.get("curated", 0) > 0],
        key=lambda x: x[1], reverse=True,
    )

    lines = [
        "",
        "=" * 60,
        "  RUN SUMMARY",
        "=" * 60,
        f"  status:          {status}",
        f"  elapsed:         {elapsed:.1f}s",
        f"  feeds:           {total_feeds} total, {total_feeds - len(failed_feeds)} ok, {len(failed_feeds)} failed",
        f"  articles:        {total_fetched} fetched -> {total_new} new -> {total_curated} curated",
    ]
    if top_sources:
        top_str = ", ".join(f"{n}({c})" for n, c in top_sources[:5])
        lines.append(f"  top sources:     {top_str}")
    if failed_feeds:
        lines.append(f"  failed feeds:    {', '.join(failed_feeds)}")
    if errors:
        lines.append(f"  errors ({len(errors)}):")
        for e in errors:
            lines.append(f"    - {e}")
    lines.append("=" * 60)
    lines.append("")

    summary = "\n".join(lines)
    log.info(summary)


def main():
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    t_start = time.monotonic()
    log.info("=== News Curator started%s ===", " (dry-run)" if args.dry_run else "")
    config = load_config(args.config)
    conn = init_db()
    errors: list[str] = []
    source_stats: dict[str, dict] = {}
    total_feeds = len(config["feeds"])

    # 1. Fetch all feeds in parallel
    all_articles: list[Article] = []
    max_per = config["scoring"].get("max_articles_per_source", 15)

    feed_results: dict[str, str | None] = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_feed = {
            executor.submit(fetch_feed, feed): feed for feed in config["feeds"]
        }
        for future in as_completed(future_to_feed):
            feed = future_to_feed[future]
            name = feed["name"]
            try:
                feed_results[name] = future.result()
            except Exception as e:
                log.error("Unexpected error fetching %s: %s", name, e)
                feed_results[name] = None

    log.info("Fetched %d feeds in parallel", len(feed_results))

    # Parse & dedup sequentially (SQLite is not thread-safe)
    total_fetched = 0
    for feed in config["feeds"]:
        name = feed["name"]
        xml = feed_results.get(name)
        if xml is None:
            errors.append(f"{name}: fetch failed")
            source_stats[name] = {"fetched": 0, "new": 0, "curated": 0, "error": True}
            continue

        parsed = parse_feed(xml, name)
        blocked_domains = config.get("blocked_domains", [])
        if blocked_domains:
            before = len(parsed)
            parsed = [a for a in parsed if urlparse(a.link).hostname not in blocked_domains]
            blocked_count = before - len(parsed)
            if blocked_count:
                log.info("Blocked %d articles from %s (domain blocklist)", blocked_count, name)
        log.info("Parsed %d articles from %s", len(parsed), name)

        new = [a for a in parsed if not is_seen(conn, a.link)]
        log.info("New from %s: %d / %d", name, len(new), len(parsed))
        all_articles.extend(new[:max_per])
        total_fetched += len(parsed)
        source_stats[name] = {"fetched": len(parsed), "new": len(new), "curated": 0}

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

    total_new = len(all_articles)

    if not all_articles:
        log.info("No new articles. Done.")
        if not args.dry_run:
            record_run(conn, total_fetched, 0, 0, len(errors), source_stats)
            cleanup_db(conn, config["db"].get("retention_days", 30))
        _print_run_summary(
            elapsed=time.monotonic() - t_start, total_feeds=total_feeds,
            total_fetched=total_fetched, total_new=0, total_curated=0,
            errors=errors, source_stats=source_stats, dry_run=args.dry_run,
        )
        conn.close()
        return

    log.info("Total new articles for curation: %d", len(all_articles))

    # 2. LLM curation
    curated = curate_with_claude(all_articles, config)

    # Update per-source curated counts
    for a in curated:
        if a.source in source_stats:
            source_stats[a.source]["curated"] += 1

    if args.dry_run:
        _print_dry_run(curated, errors)
        _print_run_summary(
            elapsed=time.monotonic() - t_start, total_feeds=total_feeds,
            total_fetched=total_fetched, total_new=total_new, total_curated=len(curated),
            errors=errors, source_stats=source_stats, dry_run=True,
        )
        conn.close()
        return

    # Mark ALL fetched articles as seen (avoid re-processing)
    mark_seen(conn, all_articles)

    # 3. Upload to Notion (even if no articles passed curation)
    upload_to_notion(curated, errors, config)

    # 4. Record run history & cleanup
    record_run(conn, total_fetched, total_new, len(curated), len(errors), source_stats)
    cleanup_db(conn, config["db"].get("retention_days", 30))
    _print_run_summary(
        elapsed=time.monotonic() - t_start, total_feeds=total_feeds,
        total_fetched=total_fetched, total_new=total_new, total_curated=len(curated),
        errors=errors, source_stats=source_stats, dry_run=False,
    )
    conn.close()
    log.info("=== News Curator finished ===")


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("Interrupted by user")
    except Exception as e:
        log.error("Unexpected error: %s", e, exc_info=True)
