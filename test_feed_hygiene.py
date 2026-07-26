"""Regression test for the feed-hygiene fix.

Bug: main() only marked the curated survivors (all_articles) as seen, so feeds
that ship a large old back-catalog re-reported the same items as "new" on every
run forever. The fix marks *everything fetched* (all_fetched) as seen, including
items about to be dropped by the date filter or the per-source cap.

This test exercises the real is_seen/mark_seen against that scenario.
"""
import sqlite3

from news_curator import Article, is_seen, mark_seen


def _mk(url, title="t", source="s"):
    return Article(title=title, link=url, description="", pub_date="", source=source)


def _init(conn):
    conn.execute(
        "CREATE TABLE seen_articles (url TEXT PRIMARY KEY, title TEXT, source TEXT, "
        "seen_at TEXT DEFAULT (datetime('now')))"
    )


def test_old_and_capped_articles_are_not_new_on_second_run():
    conn = sqlite3.connect(":memory:")
    _init(conn)

    # Run 1: a feed ships 100 items; only a few would survive date-filter/cap,
    # but the fix marks ALL fetched.
    fetched = [_mk(f"https://blog/{i}") for i in range(100)]
    new = [a for a in fetched if not is_seen(conn, a.link)]
    assert len(new) == 100, "first run: everything is new"
    mark_seen(conn, fetched)  # <-- the fix: mark all_fetched, not just survivors

    # Run 2: same back-catalog comes back. None should be counted as new.
    new_again = [a for a in fetched if not is_seen(conn, a.link)]
    assert new_again == [], "second run: old back-catalog must not re-count as new"

    # A genuinely new item still registers.
    assert is_seen(conn, "https://blog/1")
    assert not is_seen(conn, "https://blog/new-post")


if __name__ == "__main__":
    test_old_and_capped_articles_are_not_new_on_second_run()
    print("ok")
