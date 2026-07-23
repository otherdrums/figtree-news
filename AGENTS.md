# Figtree-News

Source-aware news aggregator built on Figtree figments. Every article is decomposed into WHO/WHAT/WHERE/WHEN/WHY/HOW semantic figments (role tags) that are reused across narratives — enabling structured search, natural story emergence, and self-correcting LLM evaluation.

## Build / Test / Lint

```bash
# Install (from figtree-news/ root)
pip install -e .

# Run all tests
python3 -m pytest tests/ -v

# Run specific test files
python3 -m pytest tests/test_web.py -v
python3 -m pytest tests/test_lineage.py -v
python3 -m pytest tests/test_crawler.py -v
python3 -m pytest tests/test_config_export.py -v

# No separate lint step — tests use CPU-only (no GPU required)
```

## Architecture

```
figtree_news/
├── cli.py              # Typer CLI: serve, evaluate, decompose, cogitate, query, search, etc.
├── config.py           # SourceRegistry, SearxngConfig, LlmConfig
├── ingest.py           # Feed + article → figments with boundary + KV
├── crawler.py          # RSS + SearXNG fetch loop → upsert into store
├── searxng.py          # SearXNG client, results_to_articles()
├── lineage.py          # Entity-based clustering, narrative + derivative figments
├── trust.py            # Source credibility scoring (source cred + cross-source corroboration)
├── evaluate.py         # LLMClient for external Qwen3.6-35B eval + decomposition
├── decompose.py        # Sentence → role figments (WHO/WHAT/WHERE/WHEN/WHY/HOW) via LLM
├── cogitate.py         # Automated reasoning over narratives with LLM
├── correct.py          # Self-correcting evaluation loop (verify → correct → re-verify)
├── query.py            # Structured + natural language query over decomposed articles
├── search_index.py     # SQLite FTS5 full-text search index
├── summarize_news.py   # Article summary generation
├── export.py           # Export figments to other formats
├── pipeline.py         # Multi-phase pipeline orchestration
├── llm_config.py       # LLM configuration utilities
├── web/
│   ├── serve.py        # FastAPI app: routes, API, control panel, background loop
│   ├── templates/      # Jinja2 HTML templates
│   │   └── index.html  # Main SPA with control panel, narrative/article views
│   └── static/
│       └── style.css   # Dark theme CSS, 600px slide panel
```

## Key Commands

```bash
# Serve the web UI + background crawler
python -m figtree_news.cli serve --db demo/news.lance --sources demo/sources.json --host 0.0.0.0 --port 8000

# Evaluate narratives with external LLM
python -m figtree_news.cli evaluate --db demo/news.lance --limit 20

# Decompose articles into role figments
python -m figtree_news.cli decompose --db demo/news.lance --limit 50

# Run LLM cogitation over narratives
python -m figtree_news.cli cogitate --db demo/news.lance --narrative-id <id>

# Query articles
python -m figtree_news.cli query --db demo/news.lance "Who is mentioned?"

# Search via SearXNG
python -m figtree_news.cli search "AI regulation"
```

## Key Design Details

- **CLI entry point**: `figtree-news` → `figtree_news.cli:app`
- **Server**: `serve` starts FastAPI (port 8000) + background crawler + cogitation loop
- **External LLM**: Qwen3.6-35B at `http://192.168.10.222:8081/v1/chat/completions`
  - **Critical**: Must pass `chat_template_kwargs: {"enable_thinking": false}` — Qwen3.6 puts ALL output in `reasoning_content` field otherwise
  - `chat_json()` strips `` tags via `re.sub()` and handles empty `content` by reading `reasoning_content`
- **SearXNG**: Local instance at `http://192.168.10.202:8081` (on a different machine — do not attempt podman/docker)
- **Store**: LanceDB at `demo/news.lance` — ~3,400+ figments, 360+ articles
- **Entity extraction**: `_article_entities()` uses title (not body text) for clustering specificity
- **Jaccard threshold**: 0.30 for article clustering (2+ shared entities required)
- **Numpy arrays**: Use `if boundary is None:` not `if not boundary:` — numpy truth value ambiguous error
- **Test environment**: CPU-only (no GPU required for tests), 3GB VRAM GPU available for production
- **Data path**: `demo/` directory contains sample sources.json, news.lance store
- **Logs**: `logs/` directory (gitignored) for server/crawler output

## Source Configuration

Sources are defined in `demo/sources.json` with:
- **RSS feeds**: Array of feed URLs per source
- **Trust scores**: Per-source `base_trust` (0.0-1.0)
- **SearXNG block**: Global search queries, categories, time_range
- **Unknown domains**: Auto-registered by crawler with `base_trust=0.7`

## LLM Configuration

External LLM (Qwen3.6-35B) is configured via `demo/sources.json` → `llm` block or `LlmConfig`:
```json
{
  "llm": {
    "url": "http://192.168.10.222:8081/v1/chat/completions",
    "model": "qwen3.6-35b-a3b-q4",
    "timeout": 120
  }
}
```

The `chat_template_kwargs: {"enable_thinking": false}` is hardcoded in `evaluate.py` `chat()` to prevent Qwen3.6 from dumping output into `reasoning_content`.

## Common Pitfalls

1. **Qwen3.6 thinking**: If LLM returns empty `content`, check `reasoning_content` field — model puts all output there unless `enable_thinking: false` is passed
2. **Numpy truth value**: Never use `if not array:` on numpy arrays — use `if array is None:`
3. **Entity clustering**: Uses title, not body text — titles must share 2+ named entities with Jaccard ≥ 0.30
4. **SearXNG 403 errors**: Requires restart after enabling JSON format in settings.yml on the SearXNG instance
5. **Test isolation**: Tests use temporary directories (`tmp_path`) — store data is ephemeral
