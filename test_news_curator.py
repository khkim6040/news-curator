"""Tests for news_curator.py"""

import json
import sqlite3
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from unittest.mock import MagicMock

from news_curator import (
    Article,
    strip_html,
    _extract_text_from_html,
    fetch_article_body,
    parse_pub_date,
    parse_feed,
    init_db,
    is_seen,
    mark_seen,
    cleanup_db,
    record_run,
    _estimate_reading_time,
    _compute_title,
    _build_prompt,
    _build_article_blocks,
    _build_notion_blocks,
)


# ---------------------------------------------------------------------------
# 1. strip_html
# ---------------------------------------------------------------------------

class TestStripHtml(unittest.TestCase):
    def test_removes_tags(self):
        self.assertEqual(strip_html("<p>hello</p>"), "hello")

    def test_nested_tags(self):
        self.assertEqual(strip_html("<div><b>bold</b> text</div>"), "bold text")

    def test_html_entities(self):
        self.assertEqual(strip_html("&amp; &lt; &gt;"), "& < >")

    def test_collapses_whitespace(self):
        self.assertEqual(strip_html("a  b\n\nc"), "a b c")

    def test_empty_string(self):
        self.assertEqual(strip_html(""), "")

    def test_no_tags(self):
        self.assertEqual(strip_html("plain text"), "plain text")

    def test_self_closing_tags(self):
        self.assertEqual(strip_html("line1<br/>line2"), "line1line2")

    def test_unicode_entities(self):
        self.assertEqual(strip_html("&#8220;quoted&#8221;"), "\u201cquoted\u201d")


# ---------------------------------------------------------------------------
# 2. parse_pub_date
# ---------------------------------------------------------------------------

class TestParsePubDate(unittest.TestCase):
    def test_rfc822(self):
        dt = parse_pub_date("Sat, 22 Feb 2026 10:30:00 +0000")
        self.assertIsNotNone(dt)
        self.assertEqual(dt.year, 2026)
        self.assertEqual(dt.month, 2)
        self.assertEqual(dt.day, 22)
        self.assertIsNotNone(dt.tzinfo)

    def test_iso8601_z(self):
        dt = parse_pub_date("2026-02-22T10:30:00Z")
        self.assertIsNotNone(dt)
        self.assertEqual(dt.year, 2026)
        self.assertEqual(dt.tzinfo, timezone.utc)

    def test_iso8601_offset(self):
        dt = parse_pub_date("2026-02-22T10:30:00+09:00")
        self.assertIsNotNone(dt)
        self.assertEqual(dt.hour, 10)

    def test_iso8601_milliseconds(self):
        dt = parse_pub_date("2026-02-22T10:30:00.123Z")
        self.assertIsNotNone(dt)

    def test_date_only(self):
        dt = parse_pub_date("2026-02-22")
        self.assertIsNotNone(dt)
        self.assertEqual(dt.day, 22)
        self.assertEqual(dt.tzinfo, timezone.utc)

    def test_empty_string(self):
        self.assertIsNone(parse_pub_date(""))

    def test_none_input(self):
        self.assertIsNone(parse_pub_date(None))

    def test_whitespace_only(self):
        self.assertIsNone(parse_pub_date("   "))

    def test_unparseable(self):
        self.assertIsNone(parse_pub_date("not a date"))

    def test_leading_trailing_whitespace(self):
        dt = parse_pub_date("  2026-02-22T10:30:00Z  ")
        self.assertIsNotNone(dt)

    def test_timezone_aware(self):
        """All successfully parsed dates should be timezone-aware."""
        dates = [
            "Sat, 22 Feb 2026 10:30:00 +0000",
            "2026-02-22T10:30:00Z",
            "2026-02-22",
        ]
        for d in dates:
            dt = parse_pub_date(d)
            self.assertIsNotNone(dt.tzinfo, f"Expected tz-aware for: {d}")


# ---------------------------------------------------------------------------
# 3. parse_feed (RSS 2.0 & Atom)
# ---------------------------------------------------------------------------

RSS_SAMPLE = """\
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
<channel>
  <title>Test Feed</title>
  <item>
    <title>Article One</title>
    <link>https://example.com/1</link>
    <description>&lt;p&gt;First article&lt;/p&gt;</description>
    <pubDate>Sat, 22 Feb 2026 10:00:00 +0000</pubDate>
    <category>Tech</category>
    <category>News</category>
  </item>
  <item>
    <title>Article Two</title>
    <link>https://example.com/2</link>
    <description>Second article</description>
    <pubDate>Sun, 23 Feb 2026 12:00:00 +0000</pubDate>
  </item>
  <item>
    <title></title>
    <link>https://example.com/no-title</link>
    <description>No title</description>
  </item>
  <item>
    <title>No Link</title>
    <link></link>
    <description>Missing link</description>
  </item>
</channel>
</rss>"""

ATOM_SAMPLE = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Atom Feed</title>
  <entry>
    <title>Atom Article</title>
    <link rel="alternate" href="https://example.com/atom/1"/>
    <content type="html">&lt;p&gt;Atom content&lt;/p&gt;</content>
    <published>2026-02-22T10:00:00Z</published>
    <category term="Backend"/>
  </entry>
  <entry>
    <title>Atom No Link</title>
    <summary>Summary only</summary>
    <updated>2026-02-22T11:00:00Z</updated>
  </entry>
  <entry>
    <title>Atom Fallback Link</title>
    <link href="https://example.com/atom/3"/>
    <summary>Fallback link</summary>
    <updated>2026-02-22T12:00:00Z</updated>
  </entry>
</feed>"""


class TestParseFeed(unittest.TestCase):
    def test_rss_parses_articles(self):
        articles = parse_feed(RSS_SAMPLE, "TestRSS")
        # 2 valid articles (no-title and no-link are skipped)
        self.assertEqual(len(articles), 2)

    def test_rss_fields(self):
        articles = parse_feed(RSS_SAMPLE, "TestRSS")
        a = articles[0]
        self.assertEqual(a.title, "Article One")
        self.assertEqual(a.link, "https://example.com/1")
        self.assertEqual(a.source, "TestRSS")
        self.assertIn("Tech", a.categories)
        self.assertIn("News", a.categories)

    def test_rss_strips_html_in_description(self):
        articles = parse_feed(RSS_SAMPLE, "TestRSS")
        self.assertEqual(articles[0].description, "First article")

    def test_atom_parses_articles(self):
        articles = parse_feed(ATOM_SAMPLE, "TestAtom")
        # entry 2 has no link → skipped, so 2 valid
        self.assertEqual(len(articles), 2)

    def test_atom_fields(self):
        articles = parse_feed(ATOM_SAMPLE, "TestAtom")
        a = articles[0]
        self.assertEqual(a.title, "Atom Article")
        self.assertEqual(a.link, "https://example.com/atom/1")
        self.assertEqual(a.source, "TestAtom")
        self.assertIn("Backend", a.categories)

    def test_atom_fallback_link(self):
        """Entry without rel='alternate' falls back to first <link>."""
        articles = parse_feed(ATOM_SAMPLE, "TestAtom")
        a = articles[1]
        self.assertEqual(a.link, "https://example.com/atom/3")

    def test_atom_content_strips_html(self):
        articles = parse_feed(ATOM_SAMPLE, "TestAtom")
        self.assertEqual(articles[0].description, "Atom content")

    def test_invalid_xml(self):
        self.assertEqual(parse_feed("not xml at all", "Bad"), [])

    def test_empty_feed(self):
        xml = '<?xml version="1.0"?><rss version="2.0"><channel></channel></rss>'
        self.assertEqual(parse_feed(xml, "Empty"), [])

    def test_description_truncated_to_500(self):
        long_desc = "x" * 1000
        xml = f"""\
<?xml version="1.0"?>
<rss version="2.0"><channel>
  <item>
    <title>Long</title>
    <link>https://example.com/long</link>
    <description>{long_desc}</description>
  </item>
</channel></rss>"""
        articles = parse_feed(xml, "Trunc")
        self.assertLessEqual(len(articles[0].description), 500)


# ---------------------------------------------------------------------------
# 4. SQLite DB functions (in-memory)
# ---------------------------------------------------------------------------

class TestDatabase(unittest.TestCase):
    def setUp(self):
        """Use in-memory SQLite for isolation."""
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS seen_articles (
                url   TEXT PRIMARY KEY,
                title TEXT,
                source TEXT,
                seen_at TEXT DEFAULT (datetime('now'))
            )
        """)
        self.conn.execute("""
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
        self.conn.commit()

    def tearDown(self):
        self.conn.close()

    def test_is_seen_false(self):
        self.assertFalse(is_seen(self.conn, "https://example.com/new"))

    def test_mark_and_is_seen(self):
        articles = [Article("T", "https://example.com/1", "d", "", "S")]
        mark_seen(self.conn, articles)
        self.assertTrue(is_seen(self.conn, "https://example.com/1"))

    def test_mark_seen_idempotent(self):
        articles = [Article("T", "https://example.com/1", "d", "", "S")]
        mark_seen(self.conn, articles)
        mark_seen(self.conn, articles)  # should not raise
        self.assertTrue(is_seen(self.conn, "https://example.com/1"))

    def test_mark_seen_multiple(self):
        articles = [
            Article("A", "https://example.com/a", "", "", "S"),
            Article("B", "https://example.com/b", "", "", "S"),
        ]
        mark_seen(self.conn, articles)
        self.assertTrue(is_seen(self.conn, "https://example.com/a"))
        self.assertTrue(is_seen(self.conn, "https://example.com/b"))

    def test_cleanup_removes_old(self):
        # Insert an article with old seen_at
        old_date = (datetime.now() - timedelta(days=60)).isoformat()
        self.conn.execute(
            "INSERT INTO seen_articles (url, title, source, seen_at) VALUES (?, ?, ?, ?)",
            ("https://old.com", "Old", "S", old_date),
        )
        self.conn.execute(
            "INSERT INTO seen_articles (url, title, source) VALUES (?, ?, ?)",
            ("https://new.com", "New", "S"),
        )
        self.conn.commit()
        cleanup_db(self.conn, days=30)
        self.assertFalse(is_seen(self.conn, "https://old.com"))
        self.assertTrue(is_seen(self.conn, "https://new.com"))

    def test_record_run(self):
        record_run(self.conn, 100, 50, 10, 2, {"FeedA": {"curated": 5}})
        row = self.conn.execute("SELECT * FROM run_history").fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row[2], 100)  # total_fetched
        self.assertEqual(row[3], 50)   # total_new
        self.assertEqual(row[4], 10)   # total_curated
        self.assertEqual(row[5], 2)    # error_count


# ---------------------------------------------------------------------------
# 5. _estimate_reading_time
# ---------------------------------------------------------------------------

class TestEstimateReadingTime(unittest.TestCase):
    def test_short(self):
        # 10 chars * 7 = 70, 70 // 500 = 0 → max(1,0) = 1 → ~3분
        self.assertEqual(_estimate_reading_time("x" * 10), "~3분")

    def test_medium(self):
        # 500 chars * 7 = 3500, 3500 // 500 = 7 → ~5분
        self.assertEqual(_estimate_reading_time("x" * 500), "~5분")

    def test_long(self):
        # 700 chars * 7 = 4900, 4900 // 500 = 9 → ~10분
        self.assertEqual(_estimate_reading_time("x" * 700), "~10분")

    def test_very_long(self):
        # 1000 chars * 7 = 7000, 7000 // 500 = 14 → 10분+
        self.assertEqual(_estimate_reading_time("x" * 1000), "10분+")

    def test_empty(self):
        # 0 * 7 = 0, max(1,0) = 1 → ~3분
        self.assertEqual(_estimate_reading_time(""), "~3분")


# ---------------------------------------------------------------------------
# 6. Notion page title logic (upload_to_notion 내부 로직)
# ---------------------------------------------------------------------------


class TestNotionPageTitle(unittest.TestCase):
    def _article(self, title="Test"):
        return Article(title=title, link="https://x.com", description="",
                       pub_date="", source="S")

    def test_no_articles(self):
        self.assertEqual(_compute_title([]), "☕")

    def test_single_article(self):
        self.assertEqual(_compute_title([self._article("My Title")]), "My Title")

    def test_multiple_articles(self):
        articles = [self._article("First"), self._article("Second")]
        self.assertEqual(_compute_title(articles), "First 외 1건")

    def test_long_title_truncated(self):
        long = "가" * 35  # 35 chars > 30
        title = _compute_title([self._article(long)])
        self.assertTrue(title.endswith("…"))
        self.assertLessEqual(len(title), 29)  # 28 + "…"

    def test_long_title_multiple(self):
        long = "A" * 35
        articles = [self._article(long), self._article("B")]
        title = _compute_title(articles)
        self.assertIn("외 1건", title)
        self.assertTrue(title.startswith("A" * 28 + "…"))


# ---------------------------------------------------------------------------
# 7. _build_article_blocks
# ---------------------------------------------------------------------------

class TestBuildArticleBlocks(unittest.TestCase):
    def _article(self, **kwargs):
        defaults = dict(title="Test", link="https://x.com", description="desc",
                        pub_date="", source="S", tags=[], reason="", score=8, summary="요약")
        defaults.update(kwargs)
        return Article(**{k: defaults[k] for k in
                         ["title", "link", "description", "pub_date", "source",
                          "score", "summary", "tags", "reason"]
                         if k in defaults},
                       categories=defaults.get("categories", []))

    def test_returns_callout_block(self):
        a = self._article()
        blocks = _build_article_blocks(a)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["type"], "callout")

    def test_title_is_linked(self):
        a = self._article(title="My Title", link="https://example.com")
        blocks = _build_article_blocks(a)
        rich = blocks[0]["callout"]["rich_text"]
        self.assertEqual(rich[0]["text"]["content"], "My Title")
        self.assertEqual(rich[0]["text"]["link"]["url"], "https://example.com")

    def test_tags_included(self):
        a = self._article(tags=["Kafka", "분산처리"])
        blocks = _build_article_blocks(a)
        rich = blocks[0]["callout"]["rich_text"]
        tag_texts = [r["text"]["content"] for r in rich]
        self.assertTrue(any("Kafka" in t and "분산처리" in t for t in tag_texts))

    def test_no_tags(self):
        a = self._article(tags=[])
        blocks = _build_article_blocks(a)
        rich = blocks[0]["callout"]["rich_text"]
        # Should have title, meta, summary but no tag line
        # When no tags, the tag rich_text element is not added
        tag_contents = [r["text"]["content"] for r in rich
                        if "·" in r["text"]["content"] and "via" not in r["text"]["content"]]
        self.assertEqual(len(tag_contents), 0)

    def test_reason_included(self):
        a = self._article(reason="실무에 적용 가능")
        blocks = _build_article_blocks(a)
        rich = blocks[0]["callout"]["rich_text"]
        reason_texts = [r["text"]["content"] for r in rich if "실무에 적용 가능" in r["text"]["content"]]
        self.assertEqual(len(reason_texts), 1)

    def test_no_reason(self):
        a = self._article(reason="")
        blocks = _build_article_blocks(a)
        rich = blocks[0]["callout"]["rich_text"]
        reason_texts = [r["text"]["content"] for r in rich if "💬" in r["text"]["content"]]
        self.assertEqual(len(reason_texts), 0)


# ---------------------------------------------------------------------------
# 8. _build_notion_blocks
# ---------------------------------------------------------------------------

class TestBuildNotionBlocks(unittest.TestCase):
    def _article(self, title="Test", score=8):
        return Article(title=title, link="https://x.com", description="desc",
                       pub_date="", source="S", score=score, summary="요약",
                       tags=["태그"], reason="이유")

    def test_with_articles(self):
        blocks = _build_notion_blocks([self._article()], [])
        types = [b["type"] for b in blocks]
        self.assertIn("paragraph", types)
        self.assertIn("divider", types)
        self.assertIn("callout", types)

    def test_no_articles_rest_day(self):
        blocks = _build_notion_blocks([], [])
        callouts = [b for b in blocks if b["type"] == "callout"]
        self.assertTrue(len(callouts) >= 1)
        rest_text = callouts[0]["callout"]["rich_text"][0]["text"]["content"]
        self.assertIn("추천 기준을 충족하는 기사가 없습니다", rest_text)

    def test_footer_present(self):
        blocks = _build_notion_blocks([], [])
        last = blocks[-1]
        self.assertEqual(last["type"], "paragraph")
        self.assertIn("Curated by Claude AI", last["paragraph"]["rich_text"][0]["text"]["content"])

    def test_errors_not_in_blocks(self):
        """Errors are logged, not rendered in Notion blocks."""
        blocks_with = _build_notion_blocks([], ["Feed X failed"])
        blocks_without = _build_notion_blocks([], [])
        # Block count should be the same (errors are not rendered)
        self.assertEqual(len(blocks_with), len(blocks_without))

    def test_header_shows_count(self):
        articles = [self._article("A"), self._article("B")]
        blocks = _build_notion_blocks(articles, [])
        header = blocks[0]["paragraph"]["rich_text"][0]["text"]["content"]
        self.assertIn("2건", header)


# ---------------------------------------------------------------------------
# 9. _extract_text_from_html
# ---------------------------------------------------------------------------

class TestExtractTextFromHtml(unittest.TestCase):
    def test_removes_script_and_style(self):
        html = "<html><script>alert(1)</script><style>.x{}</style><p>Hello</p></html>"
        self.assertEqual(_extract_text_from_html(html), "Hello")

    def test_removes_nav_header_footer_aside(self):
        html = "<nav>menu</nav><header>hdr</header><main>content</main><footer>ft</footer><aside>ad</aside>"
        self.assertEqual(_extract_text_from_html(html), "content")

    def test_extracts_article_tag(self):
        html = "<div>noise</div><article><p>Important text</p></article><div>more noise</div>"
        self.assertEqual(_extract_text_from_html(html), "Important text")

    def test_extracts_content_div(self):
        html = '<div>noise</div><div class="post-content"><p>Body here</p></div>'
        self.assertEqual(_extract_text_from_html(html), "Body here")

    def test_extracts_main_tag(self):
        html = "<div>noise</div><main><p>Main content</p></main>"
        self.assertEqual(_extract_text_from_html(html), "Main content")

    def test_removes_null_bytes(self):
        html = "<p>text\x00with\x00nulls</p>"
        result = _extract_text_from_html(html)
        self.assertNotIn("\x00", result)
        self.assertEqual(result, "textwithnulls")

    def test_plain_text_passthrough(self):
        self.assertEqual(_extract_text_from_html("just plain text"), "just plain text")

    def test_empty_string(self):
        self.assertEqual(_extract_text_from_html(""), "")


# ---------------------------------------------------------------------------
# 10. fetch_article_body
# ---------------------------------------------------------------------------

class TestFetchArticleBody(unittest.TestCase):
    def _article(self, link="https://example.com/article"):
        return Article(title="Test", link=link, description="", pub_date="", source="S")

    def test_rejects_file_scheme(self):
        """SSRF protection: file:// URLs should be rejected."""
        a = self._article("file:///etc/passwd")
        self.assertEqual(fetch_article_body(a), "")

    def test_rejects_ftp_scheme(self):
        a = self._article("ftp://example.com/file")
        self.assertEqual(fetch_article_body(a), "")

    def test_rejects_empty_scheme(self):
        a = self._article("//example.com/path")
        self.assertEqual(fetch_article_body(a), "")

    @patch("news_curator.urlopen")
    def test_skips_non_html_content(self, mock_urlopen):
        resp = MagicMock()
        resp.headers = {"Content-Type": "application/pdf"}
        resp.read.return_value = b"PDF data"
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp

        result = fetch_article_body(self._article())
        self.assertEqual(result, "")

    @patch("news_curator.urlopen")
    def test_fetches_and_extracts_html(self, mock_urlopen):
        html = b"<html><article><p>Article body text</p></article></html>"
        resp = MagicMock()
        resp.headers = {"Content-Type": "text/html; charset=utf-8"}
        resp.read.return_value = html
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp

        result = fetch_article_body(self._article())
        self.assertEqual(result, "Article body text")

    @patch("news_curator.urlopen")
    def test_returns_full_body(self, mock_urlopen):
        long_text = "A" * 5000
        html = f"<html><article><p>{long_text}</p></article></html>".encode()
        resp = MagicMock()
        resp.headers = {"Content-Type": "text/html"}
        resp.read.return_value = html
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp

        result = fetch_article_body(self._article())
        self.assertEqual(len(result), 5000)

    @patch("news_curator.urlopen", side_effect=Exception("Connection refused"))
    def test_returns_empty_on_error(self, mock_urlopen):
        result = fetch_article_body(self._article())
        self.assertEqual(result, "")


# ---------------------------------------------------------------------------
# 11. _build_prompt body vs description preference
# ---------------------------------------------------------------------------

class TestBuildPromptBodyPreference(unittest.TestCase):
    def _config(self):
        return {
            "curator": {
                "persona": "테스트",
                "interests": ["backend"],
                "max_articles": 10,
            },
            "scoring": {"min_score": 7},
        }

    def _article_section(self, prompt: str) -> str:
        """Extract the article listing section after the '---' separator."""
        return prompt.split("---")[-1]

    def test_uses_body_when_available(self):
        a = Article(title="Test", link="https://x.com", description="DESC_UNIQUE_XYZ",
                    pub_date="", source="S", body="BODY_UNIQUE_ABC")
        prompt = _build_prompt([a], self._config())
        section = self._article_section(prompt)
        self.assertIn("BODY_UNIQUE_ABC", section)
        self.assertNotIn("DESC_UNIQUE_XYZ", section)

    def test_falls_back_to_description(self):
        a = Article(title="Test", link="https://x.com", description="DESC_FALLBACK_123",
                    pub_date="", source="S", body="")
        prompt = _build_prompt([a], self._config())
        section = self._article_section(prompt)
        self.assertIn("DESC_FALLBACK_123", section)

    def test_no_body_no_description(self):
        a = Article(title="Test", link="https://x.com", description="",
                    pub_date="", source="S", body="")
        prompt = _build_prompt([a], self._config())
        section = self._article_section(prompt)
        self.assertNotIn("본문:", section)
        self.assertNotIn("요약:", section)


if __name__ == "__main__":
    unittest.main()
