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
