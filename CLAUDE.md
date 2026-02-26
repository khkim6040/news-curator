# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running

```bash
python3 news_curator.py
```

No dependencies beyond the Python 3.10+ standard library. Requires the `claude` CLI (`npm install -g @anthropic-ai/claude-code`) to be available on PATH for the LLM curation step.

## Architecture

Single-script application (`news_curator.py`) that runs as a daily batch job:

1. **Fetch** — Downloads RSS/Atom feeds defined in `config.json` via `urllib`
2. **Dedup** — Filters out previously seen articles using SQLite (`seen.db`)
3. **Date Filter** — Drops articles older than `max_age_days` (default 3); articles with unparseable dates are kept
4. **Curate** — Sends all new articles to Claude CLI (`claude -p`) in a single prompt; Claude returns a JSON array of scored/summarized selections
5. **Publish** — Creates a Notion database page with the curated digest via the Notion API

The Claude CLI is invoked as a subprocess with `--output-format text --max-turns 4`. The `CLAUDECODE` env var is explicitly stripped to avoid recursion. The prompt and all curation output are in Korean.

Notion output groups articles by score tier: 🔥필독 (8-10), ⭐추천 (6-7), 💡참고 (4-5). Blocks are batched in groups of 100 per Notion API limits.

## Key Files

- `news_curator.py` — Entire application logic (feed parsing, Claude CLI integration, Notion page builder)
- `config.json` — Feed URLs, curator persona/interests, Notion API credentials, scoring thresholds (gitignored)
- `config.example.json` — Template config with placeholder values
- `seen.db` — SQLite database for article deduplication (auto-created, auto-pruned after `retention_days`)
- `curator.log` — Runtime log file

## Config Structure (`config.json`)

- `feeds[]` — RSS/Atom sources with `name`, `url`, optional `headers`
- `curator` — LLM settings: `model`, `persona`, `interests[]`, `max_articles`
- `notion` — Notion API config: `token`, `database_id`
- `scoring` — `min_score` (1-10 threshold), `max_articles_per_source`, `max_age_days`
- `db` — `retention_days` for SQLite cleanup

## Important Notes

- Feed parsing handles both RSS 2.0 (`<item>`) and Atom (`<entry>`) formats with `xml.etree.ElementTree`
- Claude's JSON response is parsed with fallback regex extraction if markdown code fences are present
- ALL fetched articles (not just curated ones) are marked as seen in SQLite to prevent re-processing
- Notion page properties use Korean field names: `이름` (title), `작성일` (date)
