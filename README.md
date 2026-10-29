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
result = client.search(["What was capital expenditure in 2018?", "Who is the CEO?"], document=doc)

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

client.search(["What was revenue in 2018?"], document="doc_60a07bcef3094b68806070f2")  # search one again by its id
client.documents.delete("doc_60a07bcef3094b68806070f2")
```

With many documents, search across them: the documents likely to answer are found first, then their pages.

```python
result = client.search("What was capital expenditure in 2018?")  # every document of yours
for doc in result.results[0].documents:
    print(doc.filename, [(p.page, p.relevance) for p in doc.pages])
```

To find only which documents answer, use `client.documents.search`.

Group documents into collections, and search within one:

```python
reports = client.collections.create("Annual reports")
client.documents.upload("3M_2018_10K.pdf", collection=reports)  # or later: client.collections.add(reports, [doc])
result = client.search("What was capital expenditure in 2018?", collection=reports)
```

A document may be in many collections; deleting a collection leaves its documents.

Give documents your own metadata (strings, numbers and booleans) and filter by it:

```python
doc = client.documents.upload("3M_2018_10K.pdf", metadata={"ticker": "MMM", "year": 2018})
client.documents.update(doc, metadata={"audited": True})          # None removes a key
recent = client.documents.list(filter={"year": {"$gte": 2020}})
result = client.search("Who is the CEO?", filter={"ticker": {"$in": ["MMM", "AAPL"]}})
```

Filters take `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`, `$and` and `$or`.

## Reference

| | |
| --- | --- |
| `PageSearch(api_key=None, base_url=None, timeout=120, max_retries=2)` | The key defaults to `PAGEINDEX_SEARCH_API_KEY`. |
| `documents.upload(file, collection=None, metadata=None, wait=True)` | A path, bytes or an open file. Returns a `Document`, ready to search unless `wait=False`. |
| `search(questions, document=None, documents=None, collection=None, filter=None, max_documents=None, document_top_k=None, top_k=None, idempotency_key=None)` | Up to 10 questions. For each, the pages that help, under the documents they are in: of one document, or across documents (those named, a collection's, those meeting `filter`, or all), whose most relevant are opened: those likely to answer, at most `max_documents` (default 3), or exactly `document_top_k`. Each document lists the pages likely to help, or exactly `top_k`. A search retried after a timeout is not charged twice. |
| `documents.search(questions, document_ids=None, collection=None, filter=None, top_k=None, idempotency_key=None)` | Up to 10 questions of your documents (all of them, up to 1,000, those named, or a collection's). Each answer lists the documents likely to hold it, or exactly `top_k`. Each document is priced as a page. |
| `documents.list(limit=20, after=None, collection=None, filter=None)` | Your documents, newest first, or a collection's, or those whose metadata meets `filter`. Loop over it for every document; `.data` and `.has_more` are one page. |
| `documents.get(id)`, `documents.delete(document)` | One document; delete one. |
| `documents.update(document, metadata=)` | Set metadata keys; `None` removes one. |
| `collections.schema(collection)`, `documents.schema(collection=None, filter=None)` | The metadata keys the documents have: each key's types, how many documents have it, its least and greatest numbers and commonest strings. Counted as they are now; nothing is enforced. |
| `collections.create(name)`, `.list()`, `.get(id)`, `.rename(collection, name)`, `.delete(collection)` | Named groups of your documents. Deleting one leaves its documents. |
| `collections.add(collection, documents)`, `.remove(collection, document)` | Put documents in a collection, or take one out. `documents.upload(..., collection=)` puts a new one in. |
| `usage(start=None, end=None)` | Uploads, searches and cost by day, your credit and stored pages. |

In 0.4, `search(document, questions)` became `search(questions, document=document)`, and each answer lists
documents, each with its pages (`answer.pages` is the first document's). Before 0.2 these were `upload()`, `document()`, `documents()`, `iter_documents()` and `delete()` on the client;
they still work, with a warning, for now.

Errors raise `PageSearchError`, with the API's `status`, `code`, `message` and `request_id`.

See the [API reference](https://search.pageindex.ai/docs/api) for every field and the
[pricing](https://search.pageindex.ai/pricing).
