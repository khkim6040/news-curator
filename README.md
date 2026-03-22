# News Curator

RSS 피드에서 기술 뉴스를 모아 Claude가 읽고 골라주는 자동 큐레이터입니다. 매일 아침 Notion에 다이제스트가 올라옵니다.

## 왜 만들었나

정보의 바다에서 살아남기 위해 만들었습니다.  
기술 뉴스를 여러 소스에서 직접 훑는 건 시간이 너무 듭니다. 관심 분야에 맞는 글만 골라서 한국어 요약과 함께 한 페이지로 받아보고 싶었습니다.

## 동작 방식

```
RSS 피드 수집 → 중복 제거 → Claude CLI로 채점·요약 → Notion 페이지 생성
```

config.json에 피드 목록, 관심사, Notion 토큰을 넣고 실행하면 됩니다.

## 기술 스택

- Python 3.10+ (외부 패키지 없음, 표준 라이브러리만 사용)
- Claude CLI -- 기사 채점과 한국어 요약
- Notion API -- 다이제스트 페이지 생성
- SQLite -- 중복 기사 필터링

## 실행

```bash
cp config.example.json config.json  # 설정 파일 생성 후 토큰/피드 입력
python3 news_curator.py
```

cron으로 매일 돌리면 알아서 쌓입니다:

```bash
0 8 * * * cd /path/to/news-curator && python3 news_curator.py
```
