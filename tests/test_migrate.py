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
