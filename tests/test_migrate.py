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


def test_parse_callout_notion_api_format():
    """Notion API가 반환하는 실제 형식(color: default, 점수 세그먼트 포함)을 파싱한다."""
    block = {
        "type": "callout",
        "callout": {
            "rich_text": [
                {
                    "type": "text",
                    "text": {"content": "sql-tap - 터미널 UI 기반 실시간 SQL 트래픽 뷰어", "link": {"url": "https://news.hada.io/topic?id=26884"}},
                    "annotations": {"bold": True, "italic": False, "strikethrough": False, "underline": False, "code": False, "color": "default"},
                },
                {
                    "type": "text",
                    "text": {"content": "  [8/10]", "link": None},
                    "annotations": {"bold": True, "italic": False, "strikethrough": False, "underline": False, "code": False, "color": "red"},
                },
                {
                    "type": "text",
                    "text": {"content": "  via GeekNews", "link": None},
                    "annotations": {"bold": False, "italic": True, "strikethrough": False, "underline": False, "code": False, "color": "gray"},
                },
                {
                    "type": "text",
                    "text": {"content": "\n백엔드 인프라 · 데이터베이스 모니터링", "link": None},
                    "annotations": {"bold": False, "italic": False, "strikethrough": False, "underline": False, "code": False, "color": "gray"},
                },
                {
                    "type": "text",
                    "text": {"content": "\nsql-tap은 PostgreSQL, MySQL 등의 SQL 쿼리를 실시간으로 모니터링하는 솔루션이다.", "link": None},
                    "annotations": {"bold": False, "italic": False, "strikethrough": False, "underline": False, "code": False, "color": "default"},
                },
                {
                    "type": "text",
                    "text": {"content": "\n💬 대규모 트래픽 처리 환경에서 쿼리 성능 분석에 직접 활용 가능한 도구", "link": None},
                    "annotations": {"bold": False, "italic": True, "strikethrough": False, "underline": False, "code": False, "color": "gray"},
                },
            ],
            "icon": {"type": "emoji", "emoji": "📌"},
            "color": "gray_background",
        },
    }
    result = parse_callout_block(block)
    assert result is not None
    assert result["title"] == "sql-tap - 터미널 UI 기반 실시간 SQL 트래픽 뷰어"
    assert result["link"] == "https://news.hada.io/topic?id=26884"
    assert result["source"] == "GeekNews"
    assert result["tags"] == ["백엔드 인프라", "데이터베이스 모니터링"]
    assert result["summary"] == "sql-tap은 PostgreSQL, MySQL 등의 SQL 쿼리를 실시간으로 모니터링하는 솔루션이다."
    assert result["reason"] == "대규모 트래픽 처리 환경에서 쿼리 성능 분석에 직접 활용 가능한 도구"
