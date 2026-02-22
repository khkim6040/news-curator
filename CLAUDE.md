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
3. **Curate** — Sends all new articles to Claude CLI (`claude -p`) in a single prompt; Claude returns a JSON array of scored/summarized selections
4. **Publish** — Creates a Notion database page with the curated digest via the Notion API

The Claude CLI is invoked as a subprocess with `--output-format text --max-turns 4`. The prompt and all curation output are in Korean.

## Key Files

- `news_curator.py` — Entire application logic (feed parsing, Claude CLI integration, Notion page builder)
- `config.json` — Feed URLs, curator persona/interests, Notion API credentials, scoring thresholds
- `seen.db` — SQLite database for article deduplication (auto-created, auto-pruned after `retention_days`)
- `curator.log` — Runtime log file

## Config Structure (`config.json`)

- `feeds[]` — RSS/Atom sources with `name`, `url`, optional `headers`
- `curator` — LLM settings: `model`, `persona`, `interests[]`, `max_articles`
- `notion` — Notion API config: `token`, `database_id`
- `scoring` — `min_score` (1-10 threshold), `max_articles_per_source`
- `db` — `retention_days` for SQLite cleanup

## Important Notes

- `config.json` contains the Notion API token — never commit secrets to version control
- The `CLAUDECODE` env var is explicitly stripped when invoking the Claude subprocess to avoid recursion
- Feed parsing handles both RSS 2.0 (`<item>`) and Atom (`<entry>`) formats
- Claude's JSON response is parsed with fallback regex extraction if markdown code fences are present
