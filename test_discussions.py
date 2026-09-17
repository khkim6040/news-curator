"""Tests for community discussion fetching (Reddit / HN comment threads)."""

import json
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

import news_curator
from news_curator import (
    Article,
    _fetch_hn_thread,
    _fetch_reddit_thread,
    _reddit_get,
    fetch_article_body,
    parse_feed,
)

RSS_WITH_COMMENTS = """\
<rss version="2.0"><channel>
  <item>
    <title>HN Article</title>
    <link>https://example.com/post</link>
    <comments>https://news.ycombinator.com/item?id=12345</comments>
    <description>desc</description>
  </item>
</channel></rss>
"""

REDDIT_SELF_POST = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>Are we heading towards X?</title>
    <link href="https://www.reddit.com/r/ExperiencedDevs/comments/abc/are_we/"/>
    <content type="html">&lt;div&gt;&lt;p&gt;Long self text about architecture.&lt;/p&gt;&lt;/div&gt; &amp;#32; submitted by &amp;#32; &lt;a href="https://www.reddit.com/user/op"&gt; /u/op &lt;/a&gt; &lt;span&gt;&lt;a href="https://www.reddit.com/r/ExperiencedDevs/comments/abc/are_we/"&gt;[link]&lt;/a&gt;&lt;/span&gt; &lt;span&gt;&lt;a href="https://www.reddit.com/r/ExperiencedDevs/comments/abc/are_we/"&gt;[comments]&lt;/a&gt;&lt;/span&gt;</content>
  </entry>
  <entry>
    <title>/u/alice on Are we heading towards X?</title>
    <link href="https://www.reddit.com/r/ExperiencedDevs/comments/abc/are_we/c1/"/>
    <content type="html">&lt;div&gt;&lt;p&gt;First comment.&lt;/p&gt;&lt;/div&gt;</content>
  </entry>
  <entry>
    <title>/u/bob on Are we heading towards X?</title>
    <link href="https://www.reddit.com/r/ExperiencedDevs/comments/abc/are_we/c2/"/>
    <content type="html">&lt;div&gt;&lt;p&gt;Second comment.&lt;/p&gt;&lt;/div&gt;</content>
  </entry>
</feed>
"""

REDDIT_LINK_POST = """\
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>git worktree gotchas</title>
    <link href="https://www.reddit.com/r/programming/comments/xyz/git/"/>
    <content type="html">&amp;#32; submitted by &amp;#32; &lt;a href="https://www.reddit.com/user/op"&gt; /u/op &lt;/a&gt; &lt;span&gt;&lt;a href="https://blog.example.com/worktree"&gt;[link]&lt;/a&gt;&lt;/span&gt; &lt;span&gt;&lt;a href="https://www.reddit.com/r/programming/comments/xyz/git/"&gt;[comments]&lt;/a&gt;&lt;/span&gt;</content>
  </entry>
  <entry>
    <title>/u/alice on git worktree gotchas</title>
    <link href="https://www.reddit.com/r/programming/comments/xyz/git/c1/"/>
    <content type="html">&lt;p&gt;Nice writeup.&lt;/p&gt;</content>
  </entry>
</feed>
"""


def _resp(body: bytes, headers: dict | None = None):
    resp = MagicMock()
    resp.headers = headers or {}
    resp.read.return_value = body
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


class TestParseFeedComments(unittest.TestCase):
    def test_rss_captures_comments_url(self):
        a = parse_feed(RSS_WITH_COMMENTS, "HN")[0]
        self.assertEqual(a.comments_url, "https://news.ycombinator.com/item?id=12345")

    def test_atom_defaults_empty(self):
        a = parse_feed(REDDIT_SELF_POST, "Reddit")[0]
        self.assertEqual(a.comments_url, "")


class TestRedditThread(unittest.TestCase):
    @patch("news_curator._reddit_get", return_value=REDDIT_SELF_POST)
    def test_self_post_includes_selftext_and_comments(self, _):
        body = _fetch_reddit_thread("https://www.reddit.com/r/ExperiencedDevs/comments/abc/are_we/", 15)
        self.assertIn("Long self text about architecture.", body)
        self.assertIn("- First comment.", body)
        self.assertIn("- Second comment.", body)
        self.assertNotIn("submitted by", body)
        self.assertNotIn("[link]", body)

    @patch("news_curator._fetch_html_text", return_value="External article body.")
    @patch("news_curator._reddit_get", return_value=REDDIT_LINK_POST)
    def test_link_post_fetches_external_article(self, _, mock_html):
        body = _fetch_reddit_thread("https://www.reddit.com/r/programming/comments/xyz/git/", 15)
        mock_html.assert_called_once_with("https://blog.example.com/worktree", 15)
        self.assertIn("External article body.", body)
        self.assertIn("- Nice writeup.", body)

    @patch("news_curator._fetch_html_text", side_effect=OSError("boom"))
    @patch("news_curator._reddit_get", return_value=REDDIT_LINK_POST)
    def test_external_fetch_failure_keeps_comments(self, _, __):
        body = _fetch_reddit_thread("https://www.reddit.com/r/programming/comments/xyz/git/", 15)
        self.assertIn("- Nice writeup.", body)

    @patch("news_curator._reddit_get", return_value="<feed xmlns='http://www.w3.org/2005/Atom'></feed>")
    def test_empty_feed(self, _):
        self.assertEqual(_fetch_reddit_thread("https://www.reddit.com/r/x/comments/1/a/", 15), "")


class TestRedditGet(unittest.TestCase):
    def setUp(self):
        news_curator._reddit_next_ok = 0.0

    @patch("news_curator.time.sleep")
    @patch("news_curator.time.monotonic", return_value=1000.0)
    @patch("news_curator.urlopen")
    def test_waits_for_ratelimit_reset_between_calls(self, mock_urlopen, _, mock_sleep):
        mock_urlopen.return_value = _resp(b"<feed/>", {"x-ratelimit-reset": "32"})
        _reddit_get("https://www.reddit.com/a.rss", 15)
        mock_sleep.assert_not_called()
        _reddit_get("https://www.reddit.com/b.rss", 15)
        mock_sleep.assert_called_once_with(33.0)

    @patch("news_curator.time.sleep")
    @patch("news_curator.time.monotonic", return_value=1000.0)
    @patch("news_curator.urlopen")
    def test_retries_once_on_429(self, mock_urlopen, _, mock_sleep):
        err = HTTPError("u", 429, "Too Many", {"x-ratelimit-reset": "5"}, None)
        mock_urlopen.side_effect = [err, _resp(b"<feed/>", {"x-ratelimit-reset": "60"})]
        self.assertEqual(_reddit_get("https://www.reddit.com/a.rss", 15), "<feed/>")
        mock_sleep.assert_called_once_with(6.0)

    @patch("news_curator.time.sleep")
    @patch("news_curator.time.monotonic", return_value=1000.0)
    @patch("news_curator.urlopen")
    def test_gives_up_after_second_429(self, mock_urlopen, _, __):
        err = HTTPError("u", 429, "Too Many", {"x-ratelimit-reset": "5"}, None)
        mock_urlopen.side_effect = [err, err]
        with self.assertRaises(HTTPError):
            _reddit_get("https://www.reddit.com/a.rss", 15)


class TestHnThread(unittest.TestCase):
    @patch("news_curator.urlopen")
    def test_formats_top_level_comments(self, mock_urlopen):
        payload = {"children": [
            {"text": "<p>Great <i>point</i></p>"},
            {"text": None},
            {"text": "Second"},
        ]}
        mock_urlopen.return_value = _resp(json.dumps(payload).encode())
        body = _fetch_hn_thread("https://news.ycombinator.com/item?id=12345", 15)
        self.assertIn("https://hn.algolia.com/api/v1/items/12345", mock_urlopen.call_args[0][0].full_url)
        self.assertIn("- Great point", body)
        self.assertIn("- Second", body)

    def test_non_numeric_id_returns_empty(self):
        self.assertEqual(_fetch_hn_thread("https://news.ycombinator.com/item?id=abc", 15), "")

    @patch("news_curator.urlopen", side_effect=OSError("down"))
    def test_failure_returns_empty(self, _):
        self.assertEqual(_fetch_hn_thread("https://news.ycombinator.com/item?id=1", 15), "")


class TestFetchArticleBodyRouting(unittest.TestCase):
    @patch("news_curator._fetch_reddit_thread", return_value="thread")
    def test_reddit_links_use_thread_fetcher(self, mock_thread):
        a = Article(title="t", link="https://www.reddit.com/r/x/comments/1/a/", description="", pub_date="", source="Reddit")
        self.assertEqual(fetch_article_body(a), "thread")
        mock_thread.assert_called_once()

    @patch("news_curator._fetch_hn_thread", return_value="댓글:\n- c")
    @patch("news_curator._fetch_html_text", return_value="article")
    def test_hn_comments_appended_to_body(self, _, __):
        a = Article(title="t", link="https://example.com/p", description="", pub_date="", source="HN",
                    comments_url="https://news.ycombinator.com/item?id=1")
        self.assertEqual(fetch_article_body(a), "article\n\n댓글:\n- c")

    @patch("news_curator._fetch_hn_thread")
    @patch("news_curator._fetch_html_text", return_value="article")
    def test_no_comments_url_skips_hn(self, _, mock_hn):
        a = Article(title="t", link="https://example.com/p", description="", pub_date="", source="S")
        self.assertEqual(fetch_article_body(a), "article")
        mock_hn.assert_not_called()


if __name__ == "__main__":
    unittest.main()
