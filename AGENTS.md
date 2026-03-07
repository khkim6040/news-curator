<!-- Generated: 2026-03-07 | Updated: 2026-03-07 -->

# news-curator

## Purpose
Daily RSS/Atom news curator that fetches tech articles, scores them via Claude CLI with a fintech backend engineering rubric, and publishes a curated digest to a Notion database. Runs as a macOS launchd batch job.

## Key Files

| File | Description |
|------|-------------|
| `news_curator.py` | Entire application: feed fetching, XML parsing, Claude CLI curation, Notion page builder, SQLite dedup |
| `config.json` | Runtime config: feed URLs, curator persona/interests, Notion credentials, scoring thresholds (gitignored) |
| `config.example.json` | Template config with placeholder values and 3 example feeds |
| `seen.db` | SQLite database for article deduplication and run history (auto-created, auto-pruned) |
| `curator.log` | Runtime log file |
| `install_launchd.sh` | Helper script to install the macOS launchd plist |
| `com.gwanho.news-curator.plist` | macOS launchd job definition |
| `CLAUDE.md` | Project instructions for Claude Code |

## For AI Agents

### Working In This Directory
- Single-file architecture: all logic is in `news_curator.py` (~817 lines)
- No dependencies beyond Python 3.10+ stdlib — do not add external packages
- No test suite or linting configured
- `config.json` is gitignored and contains secrets (Notion token) — never commit it
- The scoring rubric in `_build_prompt()` is hardcoded with fintech criteria, separate from config

### Key Architecture Decisions
- Claude CLI is invoked as a subprocess (`claude -p`), not via API — `CLAUDECODE` env var is stripped to prevent recursive invocation
- ALL fetched articles (not just curated) are marked as seen in SQLite to prevent re-processing
- Feed fetching uses `ThreadPoolExecutor` (10 workers), but SQLite operations are sequential (not thread-safe)
- Notion blocks are batched in groups of 100 per API limits
- Notion page properties use Korean field names: `이름` (title), `작성일` (date)
- Claude's JSON response parsing has a fallback: strip markdown fences, then bracket-depth tracking to extract first balanced `[...]`

### Config Structure
- `feeds[]` — RSS/Atom sources with `name`, `url`, optional `headers`
- `curator` — LLM settings: `model`, `persona`, `interests[]`, `max_articles`
- `notion` — `token`, `database_id`
- `scoring` — `min_score` (1-10), `max_articles_per_source`, `max_age_days`
- `blocked_domains` — paywall/unwanted domain blocklist
- `db` — `retention_days` for SQLite cleanup

### Running
```bash
python3 news_curator.py              # normal run
python3 news_curator.py --dry-run    # curate without Notion upload or DB writes
python3 news_curator.py --verbose    # DEBUG-level logging
python3 news_curator.py --config /path/to/config.json
```

### Common Modification Points
- Adding a new feed: edit `config.json` `feeds[]` array
- Changing curation criteria: modify `_build_prompt()` rubric (lines 243-322)
- Changing Notion page layout: modify `_build_article_blocks()` and `_build_notion_blocks()`
- Adding a new CLI flag: update `parse_args()` and wire through `main()`

## Dependencies

### External
- `claude` CLI (`npm install -g @anthropic-ai/claude-code`) — required for LLM curation step
- Notion API (version `2022-06-28`) — for publishing digests
- Python 3.10+ stdlib only (`urllib`, `sqlite3`, `xml.etree`, `json`, `concurrent.futures`)

<!-- MANUAL: -->
