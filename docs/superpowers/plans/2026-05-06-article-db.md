# 기사별 Notion DB 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 큐레이션된 기사를 기사 단위로 별도 Notion DB에 적재하고, 기존 다이제스트 데이터를 마이그레이션한다.

**Architecture:** 기존 `upload_to_notion()` 뒤에 opt-in 방식으로 기사 DB 적재를 추가한다. `config.json`에 `article_database_id`가 있을 때만 동작하며, 실패해도 다이제스트에 영향 없다. 마이그레이션은 별도 스크립트로 분리한다.

**Tech Stack:** Python 3.10+ stdlib only (`urllib`, `json`), Notion API (2022-06-28)

---

### Task 1: 기사 DB 적재 함수 테스트 작성

**Files:**
- Create: `tests/test_article_db.py`

- [ ] **Step 1: 테스트 파일 생성 — `_build_article_page_payload` 단위 테스트**

`_build_article_page_payload`는 `Article` + `database_id` + `curation_date`를 받아 Notion API 페이지 생성 payload를 반환하는 순수 함수다.

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from news_curator import Article, _build_article_page_payload


def test_build_article_page_payload_full():
    """모든 필드가 채워진 기사의 payload를 검증한다."""
    article = Article(
        title="Kafka 클러스터 무중단 확장기",
        link="https://example.com/kafka",
        description="Kafka 관련 기사",
        pub_date="2026-05-01T10:00:00Z",
        source="GeekNews",
        score=8,
        summary="Kafka 클러스터를 3→5노드로 확장하면서 발생한 문제를 다룬다.",
        tags=["Kafka", "인프라"],
        reason="결제 모듈 리팩토링에 참고할 수 있다",
    )
    payload = _build_article_page_payload(article, "db-id-123", "2026-05-02")

    props = payload["parent"]
    assert props["database_id"] == "db-id-123"

    p = payload["properties"]
    assert p["제목"]["title"][0]["text"]["content"] == "Kafka 클러스터 무중단 확장기"
    assert p["링크"]["url"] == "https://example.com/kafka"
    assert p["출처"]["select"]["name"] == "GeekNews"
    assert p["태그"]["multi_select"] == [{"name": "Kafka"}, {"name": "인프라"}]
    assert p["점수"]["number"] == 8
    assert p["요약"]["rich_text"][0]["text"]["content"] == article.summary
    assert p["읽어야 할 이유"]["rich_text"][0]["text"]["content"] == article.reason
    assert p["발행일"]["date"]["start"] == "2026-05-01"
    assert p["큐레이션일"]["date"]["start"] == "2026-05-02"


def test_build_article_page_payload_no_pub_date():
    """발행일이 파싱 불가능하면 발행일 property가 None이다."""
    article = Article(
        title="제목",
        link="https://example.com/a",
        description="",
        pub_date="invalid-date",
        source="InfoQ",
        score=7,
        summary="요약",
        tags=["태그"],
        reason="이유",
    )
    payload = _build_article_page_payload(article, "db-id", "2026-05-02")
    assert payload["properties"]["발행일"]["date"] is None


def test_build_article_page_payload_empty_tags():
    """태그가 비어있으면 multi_select가 빈 배열이다."""
    article = Article(
        title="제목",
        link="https://example.com/b",
        description="",
        pub_date="",
        source="HN",
        score=6,
        summary="요약",
        tags=[],
        reason="",
    )
    payload = _build_article_page_payload(article, "db-id", "2026-05-02")
    assert payload["properties"]["태그"]["multi_select"] == []
    assert payload["properties"]["읽어야 할 이유"]["rich_text"][0]["text"]["content"] == ""
```

- [ ] **Step 2: 테스트 실행 — 실패 확인**

Run: `cd /Users/gwanhokim/personal-projects/news-curator && python3 -m pytest tests/test_article_db.py -v`

Expected: FAIL — `_build_article_page_payload` 가 존재하지 않아 ImportError

- [ ] **Step 3: 커밋**

```bash
git add tests/test_article_db.py
git commit -m "test: 기사 DB payload 빌드 함수 단위 테스트 추가"
```

---

### Task 2: 기사 DB payload 빌드 함수 구현

**Files:**
- Modify: `news_curator.py` (새 함수 추가)

- [ ] **Step 1: `_build_article_page_payload` 함수 구현**

`news_curator.py`의 `_build_article_blocks` 함수 앞에 추가한다:

```python
def _build_article_page_payload(article: Article, database_id: str, curation_date: str) -> dict:
    """Build a Notion page-create payload for one article in the article DB."""
    pub_date_iso = None
    dt = parse_pub_date(article.pub_date)
    if dt:
        pub_date_iso = dt.strftime("%Y-%m-%d")

    return {
        "parent": {"database_id": database_id},
        "properties": {
            "제목": {"title": [{"text": {"content": article.title}}]},
            "링크": {"url": article.link},
            "출처": {"select": {"name": article.source}},
            "태그": {"multi_select": [{"name": t} for t in article.tags]},
            "점수": {"number": article.score},
            "요약": {"rich_text": [{"text": {"content": article.summary}}]},
            "읽어야 할 이유": {"rich_text": [{"text": {"content": article.reason}}]},
            "발행일": {"date": {"start": pub_date_iso} if pub_date_iso else None},
            "큐레이션일": {"date": {"start": curation_date}},
        },
    }
```

- [ ] **Step 2: 테스트 실행 — 통과 확인**

Run: `cd /Users/gwanhokim/personal-projects/news-curator && python3 -m pytest tests/test_article_db.py -v`

Expected: 3 passed

- [ ] **Step 3: 커밋**

```bash
git add news_curator.py
git commit -m "feat: 기사 DB 페이지 payload 빌드 함수 추가"
```

---

### Task 3: `upload_articles_to_db` 함수 구현 및 본체 연결

**Files:**
- Modify: `news_curator.py` (`upload_articles_to_db` 추가, `upload_to_notion` 수정)

- [ ] **Step 1: `upload_articles_to_db` 함수 구현**

`upload_to_notion` 함수 바로 아래에 추가한다:

```python
def upload_articles_to_db(curated: list[Article], config: dict):
    """Upload individual articles to the article Notion DB (opt-in)."""
    ncfg = config.get("notion", {})
    article_db_id = ncfg.get("article_database_id")
    if not article_db_id:
        return

    token = os.environ.get("NOTION_TOKEN") or ncfg.get("token")
    if not token:
        log.warning("article_database_id is set but NOTION_TOKEN is missing — skipping article DB upload")
        return

    today = datetime.now().strftime("%Y-%m-%d")
    uploaded = 0
    for article in curated:
        try:
            payload = _build_article_page_payload(article, article_db_id, today)
            _notion_request("pages", token, payload)
            uploaded += 1
        except Exception as e:
            log.warning("Failed to upload article '%s' to article DB: %s", article.title, e)

    log.info("Uploaded %d / %d articles to article DB", uploaded, len(curated))
```

- [ ] **Step 2: `upload_to_notion` 뒤에 호출 추가**

`news_curator.py`의 `main()` 함수에서 `upload_to_notion(curated, errors, config)` 호출 바로 뒤에 추가:

```python
    # 3-1. Upload individual articles to article DB (opt-in)
    try:
        upload_articles_to_db(curated, config)
    except Exception as e:
        log.warning("Article DB upload failed (non-fatal): %s", e)
```

- [ ] **Step 3: `config.example.json` 업데이트**

`notion` 섹션에 `article_database_id` 필드 추가:

```json
{
  "notion": {
    "token": "ntn_xxx",
    "database_id": "your-digest-database-id",
    "article_database_id": "your-article-database-id (optional)"
  }
}
```

- [ ] **Step 4: 기존 동작 보존 테스트 작성**

`tests/test_article_db.py`에 추가:

```python
from unittest.mock import patch, MagicMock
from news_curator import upload_articles_to_db


def test_upload_articles_skips_when_no_article_db_id():
    """article_database_id가 없으면 아무것도 하지 않는다."""
    config = {"notion": {"token": "fake-token"}}
    with patch("news_curator._notion_request") as mock_req:
        upload_articles_to_db([], config)
        mock_req.assert_not_called()


def test_upload_articles_calls_notion_api():
    """article_database_id가 있으면 기사 수만큼 Notion API를 호출한다."""
    article = Article(
        title="Test", link="https://example.com/test",
        description="", pub_date="2026-05-01", source="Test",
        score=8, summary="요약", tags=["태그"], reason="이유",
    )
    config = {
        "notion": {
            "token": "fake-token",
            "article_database_id": "art-db-id",
        },
    }
    with patch("news_curator._notion_request", return_value={"id": "page-1"}) as mock_req:
        upload_articles_to_db([article], config)
        assert mock_req.call_count == 1
        call_args = mock_req.call_args
        assert call_args[0][0] == "pages"
        payload = call_args[0][2]
        assert payload["parent"]["database_id"] == "art-db-id"


def test_upload_articles_continues_on_single_failure():
    """하나의 기사 업로드가 실패해도 나머지를 계속 처리한다."""
    articles = [
        Article(title="A", link="https://a.com", description="", pub_date="",
                source="S", score=7, summary="s", tags=[], reason=""),
        Article(title="B", link="https://b.com", description="", pub_date="",
                source="S", score=8, summary="s", tags=[], reason=""),
    ]
    config = {"notion": {"token": "t", "article_database_id": "db"}}

    side_effects = [Exception("API error"), {"id": "page-2"}]
    with patch("news_curator._notion_request", side_effect=side_effects):
        upload_articles_to_db(articles, config)  # should not raise
```

- [ ] **Step 5: 전체 테스트 실행**

Run: `cd /Users/gwanhokim/personal-projects/news-curator && python3 -m pytest tests/test_article_db.py -v`

Expected: 6 passed

- [ ] **Step 6: 커밋**

```bash
git add news_curator.py config.example.json tests/test_article_db.py
git commit -m "feat: 기사 DB 적재 함수 구현 및 본체 연결 (opt-in)"
```

---

### Task 4: 마이그레이션 스크립트 — callout 파싱 테스트

**Files:**
- Create: `tests/test_migrate.py`

- [ ] **Step 1: callout 블록 파싱 함수 테스트 작성**

`_build_article_blocks`가 생성하는 callout 구조를 기준으로 역파싱 테스트를 작성한다:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from migrate_to_article_db import parse_callout_block


def _make_callout(title, link, source, summary, tags_str="", reason="", reading_time="~5분"):
    """news_curator.py의 _build_article_blocks가 생성하는 callout 구조를 재현한다."""
    rich_text = [
        {
            "type": "text",
            "text": {"content": title, "link": {"url": link}},
            "annotations": {"bold": True},
        },
        {
            "type": "text",
            "text": {"content": f"\n{reading_time}  via {source}", "link": None},
            "annotations": {"color": "gray", "italic": True},
        },
        {"type": "text", "text": {"content": f"\n{summary}", "link": None}},
    ]
    if tags_str:
        rich_text.append(
            {"type": "text", "text": {"content": f"\n{tags_str}", "link": None},
             "annotations": {"color": "gray"}}
        )
    if reason:
        rich_text.append(
            {"type": "text", "text": {"content": f"\n💬 읽어야 할 이유: {reason}", "link": None},
             "annotations": {"italic": True, "color": "gray"}}
        )
    return {
        "type": "callout",
        "callout": {
            "rich_text": rich_text,
            "icon": {"type": "emoji", "emoji": "📌"},
            "color": "gray_background",
        },
    }


def test_parse_callout_full():
    """모든 필드가 있는 callout을 정확히 파싱한다."""
    block = _make_callout(
        title="Kafka 확장기",
        link="https://example.com/kafka",
        source="GeekNews",
        summary="Kafka 클러스터를 확장한 사례다.",
        tags_str="Kafka · 인프라",
        reason="결제 모듈에 참고할 수 있다",
    )
    result = parse_callout_block(block)
    assert result is not None
    assert result["title"] == "Kafka 확장기"
    assert result["link"] == "https://example.com/kafka"
    assert result["source"] == "GeekNews"
    assert result["summary"] == "Kafka 클러스터를 확장한 사례다."
    assert result["tags"] == ["Kafka", "인프라"]
    assert result["reason"] == "결제 모듈에 참고할 수 있다"


def test_parse_callout_no_tags_no_reason():
    """태그와 reason이 없는 callout도 파싱한다."""
    block = _make_callout(
        title="간단한 기사",
        link="https://example.com/simple",
        source="HN",
        summary="간단한 요약이다.",
    )
    result = parse_callout_block(block)
    assert result is not None
    assert result["title"] == "간단한 기사"
    assert result["tags"] == []
    assert result["reason"] == ""


def test_parse_callout_non_callout_block():
    """callout이 아닌 블록은 None을 반환한다."""
    block = {"type": "paragraph", "paragraph": {"rich_text": []}}
    assert parse_callout_block(block) is None
```

- [ ] **Step 2: 테스트 실행 — 실패 확인**

Run: `cd /Users/gwanhokim/personal-projects/news-curator && python3 -m pytest tests/test_migrate.py -v`

Expected: FAIL — `migrate_to_article_db` 모듈이 없어 ImportError

- [ ] **Step 3: 커밋**

```bash
git add tests/test_migrate.py
git commit -m "test: 마이그레이션 callout 파싱 단위 테스트 추가"
```

---

### Task 5: 마이그레이션 스크립트 구현

**Files:**
- Create: `migrate_to_article_db.py`

- [ ] **Step 1: `parse_callout_block` 함수 구현**

```python
#!/usr/bin/env python3
"""One-time migration: parse existing digest pages and load articles into the article DB."""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

BASE_DIR = Path(__file__).parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def _notion_request(endpoint: str, token: str, payload: dict, method: str = "POST") -> dict:
    """Make a request to the Notion API."""
    url = f"https://api.notion.com/v1/{endpoint}"
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Content-Type", "application/json")
    req.add_header("Notion-Version", "2022-06-28")
    with urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _notion_get(endpoint: str, token: str) -> dict:
    """Make a GET request to the Notion API."""
    url = f"https://api.notion.com/v1/{endpoint}"
    req = Request(url, method="GET")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Notion-Version", "2022-06-28")
    with urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


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
    tags = []
    reason = ""

    for segment in rich_text:
        text_obj = segment.get("text", {})
        content = text_obj.get("content", "")
        link_obj = text_obj.get("link")
        annotations = segment.get("annotations", {})

        # First bold segment with a link → title + link
        if annotations.get("bold") and link_obj and link_obj.get("url") and not title:
            title = content
            link = link_obj["url"]
            continue

        # "via {source}" line → source
        if annotations.get("italic") and annotations.get("color") == "gray":
            via_match = re.search(r"via\s+(.+)", content)
            reason_match = re.search(r"💬 읽어야 할 이유:\s*(.+)", content)
            if reason_match:
                reason = reason_match.group(1).strip()
            elif via_match:
                source = via_match.group(1).strip()
            continue

        # Gray non-italic text after summary → tags (· separated)
        if annotations.get("color") == "gray" and not annotations.get("italic") and not annotations.get("bold"):
            stripped = content.strip()
            if " · " in stripped or (stripped and not summary):
                # Could be tags
                if " · " in stripped:
                    tags = [t.strip() for t in stripped.split(" · ") if t.strip()]
                    continue

        # Plain text → summary
        if not annotations.get("bold") and not annotations.get("italic") and not annotations.get("color"):
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


def fetch_all_digest_pages(token: str, database_id: str) -> list[dict]:
    """Fetch all pages from the digest database with pagination."""
    pages = []
    has_more = True
    start_cursor = None

    while has_more:
        payload = {"page_size": 100}
        if start_cursor:
            payload["start_cursor"] = start_cursor

        result = _notion_request(f"databases/{database_id}/query", token, payload)
        pages.extend(result.get("results", []))
        has_more = result.get("has_more", False)
        start_cursor = result.get("next_cursor")
        log.info("Fetched %d pages so far…", len(pages))

    return pages


def fetch_page_blocks(token: str, page_id: str) -> list[dict]:
    """Fetch all child blocks of a page."""
    blocks = []
    has_more = True
    start_cursor = None

    while has_more:
        url = f"blocks/{page_id}/children?page_size=100"
        if start_cursor:
            url += f"&start_cursor={start_cursor}"

        result = _notion_get(url, token)
        blocks.extend(result.get("results", []))
        has_more = result.get("has_more", False)
        start_cursor = result.get("next_cursor")

    return blocks


def fetch_existing_links(token: str, article_db_id: str) -> set[str]:
    """Fetch all existing article links from the article DB for dedup."""
    links = set()
    has_more = True
    start_cursor = None

    while has_more:
        payload = {"page_size": 100}
        if start_cursor:
            payload["start_cursor"] = start_cursor

        result = _notion_request(f"databases/{article_db_id}/query", token, payload)
        for page in result.get("results", []):
            url = page.get("properties", {}).get("링크", {}).get("url")
            if url:
                links.add(url)
        has_more = result.get("has_more", False)
        start_cursor = result.get("next_cursor")

    return links


def build_article_payload(article: dict, article_db_id: str, curation_date: str) -> dict:
    """Build a Notion page-create payload from parsed article dict."""
    return {
        "parent": {"database_id": article_db_id},
        "properties": {
            "제목": {"title": [{"text": {"content": article["title"]}}]},
            "링크": {"url": article["link"]},
            "출처": {"select": {"name": article["source"]}} if article["source"] else {"select": None},
            "태그": {"multi_select": [{"name": t} for t in article["tags"]]},
            "점수": {"number": 0},
            "요약": {"rich_text": [{"text": {"content": article["summary"]}}]},
            "읽어야 할 이유": {"rich_text": [{"text": {"content": article["reason"]}}]},
            "발행일": {"date": None},
            "큐레이션일": {"date": {"start": curation_date} if curation_date else None},
        },
    }


def migrate(config_path: Path, dry_run: bool = False):
    """Run the full migration."""
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    ncfg = config.get("notion", {})
    token = os.environ.get("NOTION_TOKEN") or ncfg.get("token")
    if not token:
        log.error("NOTION_TOKEN is not set")
        sys.exit(1)

    digest_db_id = ncfg.get("database_id")
    article_db_id = ncfg.get("article_database_id")
    if not digest_db_id or not article_db_id:
        log.error("Both 'database_id' and 'article_database_id' must be set in config")
        sys.exit(1)

    # 1. Fetch all digest pages
    log.info("Fetching digest pages from database %s…", digest_db_id)
    pages = fetch_all_digest_pages(token, digest_db_id)
    log.info("Found %d digest pages", len(pages))

    # 2. Fetch existing links for dedup
    existing_links = set()
    if not dry_run:
        log.info("Fetching existing articles for dedup…")
        existing_links = fetch_existing_links(token, article_db_id)
        log.info("Found %d existing articles in article DB", len(existing_links))

    # 3. Parse each page
    total_articles = 0
    uploaded = 0
    skipped_dup = 0
    skipped_parse = 0

    for page in pages:
        page_id = page["id"]
        date_prop = page.get("properties", {}).get("작성일", {}).get("date")
        curation_date = date_prop.get("start", "") if date_prop else ""
        title_parts = page.get("properties", {}).get("이름", {}).get("title", [])
        page_title = title_parts[0]["text"]["content"] if title_parts else "(untitled)"

        log.info("Processing page: %s (%s)", page_title, curation_date)

        blocks = fetch_page_blocks(token, page_id)
        page_articles = 0

        for block in blocks:
            parsed = parse_callout_block(block)
            if parsed is None:
                continue

            total_articles += 1
            page_articles += 1

            if parsed["link"] in existing_links:
                skipped_dup += 1
                if dry_run:
                    print(f"  [DUP] {parsed['title']}")
                continue

            if dry_run:
                tags_str = " · ".join(parsed["tags"]) if parsed["tags"] else "(없음)"
                print(f"  [{total_articles}] {parsed['title']}")
                print(f"      출처: {parsed['source']}  태그: {tags_str}")
                print(f"      요약: {parsed['summary'][:80]}…" if len(parsed['summary']) > 80 else f"      요약: {parsed['summary']}")
                continue

            try:
                payload = build_article_payload(parsed, article_db_id, curation_date)
                _notion_request("pages", token, payload)
                existing_links.add(parsed["link"])
                uploaded += 1
            except Exception as e:
                log.warning("Failed to upload '%s': %s", parsed["title"], e)

        if page_articles == 0:
            skipped_parse += 1

    # Summary
    print(f"\n{'='*50}")
    print(f"  Migration {'(DRY RUN) ' if dry_run else ''}Summary")
    print(f"{'='*50}")
    print(f"  Pages processed:    {len(pages)}")
    print(f"  Articles found:     {total_articles}")
    if not dry_run:
        print(f"  Uploaded:           {uploaded}")
    print(f"  Skipped (dup):      {skipped_dup}")
    print(f"  Pages w/o articles: {skipped_parse}")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description="Migrate digest pages to article DB")
    parser.add_argument("--dry-run", action="store_true", help="파싱 결과만 출력, 실제 적재 안 함")
    parser.add_argument("--config", type=Path, default=BASE_DIR / "config.json", help="config.json 경로")
    args = parser.parse_args()
    migrate(args.config, args.dry_run)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 테스트 실행 — 통과 확인**

Run: `cd /Users/gwanhokim/personal-projects/news-curator && python3 -m pytest tests/test_migrate.py -v`

Expected: 3 passed

- [ ] **Step 3: 커밋**

```bash
git add migrate_to_article_db.py
git commit -m "feat: 다이제스트 → 기사 DB 마이그레이션 스크립트 구현"
```

---

### Task 6: config.example.json 및 문서 정리

**Files:**
- Modify: `config.example.json`

- [ ] **Step 1: config.example.json 업데이트**

`notion` 섹션에 `article_database_id` 필드를 추가한다:

현재:
```json
(notion 키가 없음 — config.example.json에는 notion 섹션이 없고, 실제 config.json에만 있음)
```

config.example.json 전체에 notion 섹션이 빠져 있으므로 추가한다:

```json
{
  "feeds": [ ... ],
  "curator": { ... },
  "scoring": { ... },
  "blocked_domains": [ ... ],
  "notion": {
    "token": "ntn_xxx",
    "database_id": "your-digest-database-id",
    "article_database_id": "your-article-database-id (optional, enables per-article DB)"
  },
  "db": { ... }
}
```

- [ ] **Step 2: 전체 테스트 실행**

Run: `cd /Users/gwanhokim/personal-projects/news-curator && python3 -m pytest tests/ -v`

Expected: 6 passed (test_article_db 3건 + test_migrate 3건)

- [ ] **Step 3: 커밋**

```bash
git add config.example.json
git commit -m "docs: config.example.json에 notion 섹션 및 article_database_id 추가"
```

---

### Task 7: 수동 E2E 검증

- [ ] **Step 1: Notion에 기사 DB 생성**

Notion에서 빈 데이터베이스를 수동으로 생성한다. 속성은 자동으로 생성되므로 빈 DB로 충분하다.
생성된 DB의 ID를 `config.json`의 `notion.article_database_id`에 넣는다.

- [ ] **Step 2: dry-run으로 마이그레이션 확인**

Run: `cd /Users/gwanhokim/personal-projects/news-curator && python3 migrate_to_article_db.py --dry-run`

Expected: 기존 다이제스트 페이지에서 기사가 파싱되어 출력됨

- [ ] **Step 3: 실제 마이그레이션 실행**

Run: `cd /Users/gwanhokim/personal-projects/news-curator && python3 migrate_to_article_db.py`

Expected: Notion 기사 DB에 row들이 생성됨

- [ ] **Step 4: 본체 dry-run 확인**

Run: `cd /Users/gwanhokim/personal-projects/news-curator && python3 news_curator.py --dry-run`

Expected: 기존 큐레이션 동작 정상, 기사 DB 적재는 dry-run이므로 스킵
