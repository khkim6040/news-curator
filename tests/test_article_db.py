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
    assert p["큐레이션일"]["date"]["start"] == "2026-05-02"
    assert "발행일" not in p


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
