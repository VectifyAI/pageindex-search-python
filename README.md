# pageindex-search

Python client for [Page Search API](https://search.pageindex.ai) by PageIndex. Upload a PDF, ask up to 10
questions at once, and get back the pages that answer each one, ranked by relevance. Every page is read.

```bash
pip install pageindex-search
```

Create an API key in the [console](https://search.pageindex.ai/console/keys) and set it:

```bash
export PAGEINDEX_SEARCH_API_KEY=pis_live_...
```

## Quick start

```python
from pageindex_search import PageSearch

client = PageSearch()

doc = client.documents.upload("3M_2018_10K.pdf")
result = client.search(doc, ["What was capital expenditure in 2018?", "Who is the CEO?"])

for answer in result.results:
    print(answer.question)
    for page in answer.pages:
        print(f"  page {page.page}: {page.relevance}")

print(f"cost ${result.usage.cost_usd}, credit left ${result.usage.balance_usd}")
```

Upload a document once and search it as often as you like; uploading the same file again returns the same
document. Documents are kept until you delete them:

```python
for doc in client.documents.list():  # every document, newest first, a page at a time
    print(doc.id, doc.filename, doc.pages)

client.search("doc_60a07bcef3094b68806070f2", ["What was revenue in 2018?"])  # search one again by its id
client.documents.delete("doc_60a07bcef3094b68806070f2")
```

## Reference

| | |
| --- | --- |
| `PageSearch(api_key=None, base_url=None, timeout=120, max_retries=2)` | The key defaults to `PAGEINDEX_SEARCH_API_KEY`. |
| `documents.upload(file, wait=True)` | A path, bytes or an open file. Returns a `Document`, ready to search unless `wait=False`. |
| `search(document, questions, top_k=None, idempotency_key=None)` | Up to 10 questions. Each answer lists the pages likely to help (possibly none), or exactly `top_k`. A search retried after a timeout is not charged twice. |
| `documents.list(limit=20, after=None)` | Your documents, newest first. Loop over it for every document; `.data` and `.has_more` are one page. |
| `documents.get(id)`, `documents.delete(document)` | One document; delete one. |
| `usage(start=None, end=None)` | Uploads, searches and cost by day, your credit and stored pages. |

Before 0.2 these were `upload()`, `document()`, `documents()`, `iter_documents()` and `delete()` on the client;
they still work, with a warning, for now.

Errors raise `PageSearchError`, with the API's `status`, `code`, `message` and `request_id`.

See the [API reference](https://search.pageindex.ai/docs/api) for every field and the
[pricing](https://search.pageindex.ai/pricing).
