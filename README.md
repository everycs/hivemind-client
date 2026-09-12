# makersite-hivemind

Python client for the Hivemind vault-search API.

## Install

```bash
pip install git+https://github.com/everycs/hivemind-client.git
```

## Usage

```python
from makersite_hivemind import HivemindClient

async with HivemindClient("http://hivemind.orion.makersite.org:8300") as client:
    # Compressed answer (1-3 sentences, cached 24h)
    answer = await client.ask("what are the OrientDB backup conventions?")

    # Full-text search results
    results = await client.search("orientdb backup", limit=5)

    # Search with auto-query correction
    results = await client.smart_search("database keeps crashing")

    # Retrieve a single document
    doc = await client.get_document("database/orientdb-overview.md")

    # Health check
    status = await client.health()
```

## Circuit breaker

The client has a built-in circuit breaker. After 3 consecutive failures, it stops
making HTTP calls for 60 seconds (configurable):

```python
client = HivemindClient(
    "http://hivemind:8300",
    failure_threshold=5,
    circuit_open_seconds=120.0,
)
```

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/ask` | Compressed answer via hybrid BM25+vector RAG |
| GET | `/search` | BM25 full-text search |
| GET | `/smart-search` | Search with LLM query correction |
| GET | `/get` | Retrieve single document by path |
| GET | `/health` | Health check |
