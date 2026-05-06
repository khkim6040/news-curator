# 기사별 Notion DB 설계

## 목표

몇 달치 쌓인 큐레이션 기사를 기사 단위로 쿼리/필터 가능하게 만든다.
기존 일별 다이제스트 페이지는 그대로 유지하고, 별도 기사 전용 Notion DB를 추가한다.

## 핵심 결정 사항

- 기존 다이제스트 DB와 **별도 Notion DB**로 생성
- 기사 하나 = DB row 하나
- `config.json`에 `article_database_id`가 없으면 기사 DB 적재를 건너뜀 (opt-in)
- 마이그레이션은 별도 일회성 스크립트 (`migrate_to_article_db.py`)

## 새 Notion DB 스키마

| Property | Notion 타입 | 값 예시 | 비고 |
|----------|------------|--------|------|
| 제목 | title | "Kafka 클러스터 무중단 확장기" | |
| 링크 | url | https://... | 기사 원본 URL |
| 출처 | select | "GeekNews" | 피드 이름 |
| 태그 | multi-select | "Kafka", "인프라" | 한국어 태그 1-3개 |
| 점수 | number | 8 | 1-10 |
| 요약 | rich_text | "Kafka 클러스터를..." | 2-3문장 |
| 읽어야 할 이유 | rich_text | "현재 결제 모듈..." | reason 필드 |
| 발행일 | date | 2026-04-30 | 기사 원본 pub_date |
| 큐레이션일 | date | 2026-05-01 | 다이제스트 발행일 |

## 본체 변경 (`news_curator.py`)

### 변경 범위

- `upload_to_notion()` 끝에 기사 DB 적재 함수 호출 추가
- 새 함수 `upload_articles_to_db(curated, config)` 추가
- `config.json`의 `notion` 섹션에 `article_database_id` 필드 추가

### 동작 방식

1. 기존 `upload_to_notion()`이 다이제스트 페이지를 정상 생성
2. `config["notion"]`에 `article_database_id`가 있으면 기사 DB 적재 실행
3. 없으면 건너뜀 (기존 동작 그대로)
4. 기사 DB 적재가 실패해도 다이제스트 페이지에는 영향 없음 (try/except로 격리)

### config.json 변경

`notion` 섹션에 `article_database_id` 추가:

```json
{
  "notion": {
    "token": "...",
    "database_id": "...",
    "article_database_id": "새 기사 DB ID (optional)"
  }
}
```

## 마이그레이션 스크립트 (`migrate_to_article_db.py`)

### 목적

기존 다이제스트 페이지에 임베딩된 기사 데이터를 파싱하여 새 기사 DB에 적재한다.

### 동작

1. Notion API로 다이제스트 DB의 모든 페이지 목록 조회 (pagination 처리)
2. 각 페이지의 children blocks 조회
3. callout 블록에서 기사 데이터 파싱:
   - 첫 번째 bold+link 텍스트 → 제목 + 링크
   - "via {source}" → 출처
   - 요약 텍스트 → 요약
   - 태그 텍스트 (· 구분) → 태그
   - "읽어야 할 이유:" 뒤 텍스트 → reason
   - 페이지의 `작성일` property → 큐레이션일
4. 파싱한 기사를 새 기사 DB에 row로 생성
5. 중복 방지: 링크(URL) 기준으로 이미 존재하는 기사는 건너뜀

### CLI 옵션

- `--dry-run`: 파싱 결과만 출력, 실제 적재 안 함
- `--config`: config.json 경로 (기본: `config.json`)

### 제약 사항

- callout 블록 텍스트 파싱이므로 형식이 달라진 과거 페이지는 실패할 수 있음
- 점수는 현재 다이제스트에 노출하지 않으므로 마이그레이션 시 0으로 채움
- 발행일도 다이제스트 블록에 없으므로 큐레이션일로 대체

## 되돌리기 전략

| 되돌리기 대상 | 방법 |
|-------------|------|
| 새 기사 적재 중단 | `config.json`에서 `article_database_id` 제거 |
| 기사 DB 자체 삭제 | Notion에서 DB 삭제 (본체 코드 무관) |
| 코드 원복 | 기사 DB 관련 함수만 제거하면 끝 |
| 마이그레이션 원복 | Notion에서 기사 DB의 row 전체 삭제 또는 DB 삭제 |

기존 다이제스트 흐름에 대한 사이드이펙트 zero.
