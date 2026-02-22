# News Curator

RSS 피드에서 기술 뉴스를 수집하고, Claude CLI로 자동 큐레이션하여 Notion 데이터베이스에 매일 다이제스트 페이지를 생성합니다.

## 파이프라인

```
RSS/Atom 피드 수집 → 중복 제거(SQLite) → 오래된 글 필터링 → Claude CLI 큐레이션 → Notion 페이지 생성
```

- 피드별 최대 8개 기사 수집, 3일 이내 글만 필터링
- Claude가 페르소나/관심사 기반으로 1-10점 채점, 한국어 요약 생성
- Notion 페이지에 점수 등급별 그룹핑 (필독/추천/참고)

## 요구사항

- Python 3.10+
- [Claude CLI](https://docs.anthropic.com/en/docs/claude-code) (`npm install -g @anthropic-ai/claude-code`)
- Notion Internal Integration 토큰

외부 Python 패키지 불필요 (표준 라이브러리만 사용)

## 설정

1. `config.example.json`을 복사하여 `config.json` 생성:

```bash
cp config.example.json config.json
```

2. Notion 설정:
   - [Notion Integrations](https://www.notion.so/profile/integrations)에서 Internal Integration 생성 → 토큰 복사
   - Notion 데이터베이스 생성 (속성: `Name`(제목), `작성일`(날짜))
   - 데이터베이스 페이지 → `...` → "Add connections" → 생성한 Integration 연결
   - 공개 공유 원할 시 "Share to web" 활성화

3. `config.json`에 토큰과 데이터베이스 ID 입력

## 실행

```bash
python3 news_curator.py
```

cron으로 매일 자동 실행 예시:

```bash
0 8 * * * cd /path/to/news-curator && python3 news_curator.py
```

## 설정 항목 (`config.json`)

| 섹션 | 키 | 설명 |
|------|----|------|
| `feeds[]` | `name`, `url`, `headers` | RSS/Atom 피드 소스 |
| `curator` | `model` | Claude 모델 (기본: `claude-haiku-4-5-20251001`) |
| `curator` | `persona` | 큐레이터 페르소나 설명 |
| `curator` | `interests` | 관심 분야 목록 |
| `curator` | `max_articles` | 최종 선별 최대 기사 수 |
| `notion` | `token`, `database_id` | Notion API 인증 정보 |
| `scoring` | `min_score` | 최소 선별 점수 (1-10) |
| `scoring` | `max_articles_per_source` | 피드당 최대 수집 수 |
| `scoring` | `max_age_days` | 기사 최대 허용 일수 |
| `db` | `retention_days` | SQLite 중복 기록 보관 기간 |
