"""The client against a stand-in for the API: errors it raises. test_live.py runs it against the real one."""
import httpx
import pytest

from pageindex_search import PageSearch, PageSearchError


def client(handler) -> PageSearch:
    return PageSearch(api_key="pis_live_x", max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_failed_upload_raises():
    failed = {"object": "document", "id": "doc_1", "status": "failed", "filename": "a.pdf", "pages": None,
              "bytes": 4, "created_at": "2026-10-27T00:00:00Z", "error": "The PDF could not be read; it may be damaged."}
    pages = client(lambda request: httpx.Response(422, json=failed, headers={"x-request-id": "req_1"}))
    with pytest.raises(PageSearchError) as error:
        pages.documents.upload(b"%PDF-1.4")
    assert error.value.code == "document_failed" and error.value.status == 422 and error.value.request_id == "req_1"
    assert error.value.message == "The PDF could not be read; it may be damaged."


def test_api_error_raises():
    body = {"error": {"code": "document_not_found", "message": "No document doc_9.", "request_id": "req_9"}}
    pages = client(lambda request: httpx.Response(404, json=body))
    with pytest.raises(PageSearchError) as error:
        pages.documents.get("doc_9")
    assert (error.value.status, error.value.code, error.value.message, error.value.request_id) == \
        (404, "document_not_found", "No document doc_9.", "req_9")


def test_search_retries_with_one_idempotency_key():
    keys = []

    def handler(request):
        keys.append(request.headers.get("idempotency-key"))
        if len(keys) == 1:
            return httpx.Response(503, json={"error": {"code": "search_unavailable", "message": "busy"}})
        return httpx.Response(200, json={"object": "search", "id": "srch_1", "results": [],
                                         "failed_documents": [], "processing_documents": [], "usage": None,
                                         "seconds": 0.1})

    pages = PageSearch(api_key="pis_live_x", max_retries=1,
                       http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    pages.search("q", document="doc_1")
    assert len(keys) == 2 and keys[0] and keys[0] == keys[1]


def test_list_pages_on_when_iterated():
    docs = [{"object": "document", "id": f"doc_{i}", "status": "ready", "filename": "a.pdf", "pages": 1, "bytes": 4,
             "created_at": "2026-10-27T00:00:00Z"} for i in range(5)]
    seen = []

    def handler(request):
        after = request.url.params.get("after")
        seen.append(after)
        start = 0 if after is None else int(after.split("_")[1]) + 1
        page = docs[start:start + 2]
        return httpx.Response(200, json={"object": "list", "data": page, "has_more": start + 2 < len(docs)})

    pages = client(handler)
    first = pages.documents.list(limit=2)
    assert [d.id for d in first.data] == ["doc_0", "doc_1"] and first.has_more
    assert [d.id for d in pages.documents.list(limit=2)] == [f"doc_{i}" for i in range(5)]
    assert seen[1:] == [None, "doc_1", "doc_3"]


def test_old_names_still_work_and_warn():
    doc = {"object": "document", "id": "doc_1", "status": "ready", "filename": "a.pdf", "pages": 1, "bytes": 4,
           "created_at": "2026-10-27T00:00:00Z"}
    pages = client(lambda request: httpx.Response(200, json=doc if request.method == "GET" and "doc_1" in str(request.url)
                                                  else {"object": "list", "data": [doc], "has_more": False}))
    with pytest.warns(DeprecationWarning, match="documents.get"):
        assert pages.document("doc_1").id == "doc_1"
    with pytest.warns(DeprecationWarning, match="documents.list"):
        assert [d.id for d in pages.documents()] == ["doc_1"]
    with pytest.warns(DeprecationWarning, match="documents.list"):
        assert [d.id for d in pages.iter_documents()] == ["doc_1"]


def test_document_search():
    sent = []

    def handler(request):
        sent.append(request)
        return httpx.Response(200, json={
            "object": "document_search", "id": "srch_1",
            "results": [{"question": "Who is the CEO?", "documents": [
                {"document_id": "doc_2", "filename": "b.pdf", "pages": 40, "relevance": 0.9}]}],
            "failed_documents": [], "processing_documents": ["doc_3"],
            "usage": {"documents": 2, "questions": 1, "billable_documents": 2, "cost_usd": 0.001,
                      "balance_usd": 4.999},
            "seconds": 0.8})

    from pageindex_search import Document
    doc = Document(id="doc_1", status="ready")
    result = client(handler).documents.search("Who is the CEO?", document_ids=[doc, "doc_2"], top_k=1)
    import json
    assert sent[0].url.path == "/v1/documents/search" and sent[0].headers["idempotency-key"]
    assert json.loads(sent[0].content) == {"questions": ["Who is the CEO?"], "document_ids": ["doc_1", "doc_2"],
                                           "top_k": 1}
    assert result.results[0].documents[0].document_id == "doc_2" and result.results[0].documents[0].pages == 40
    assert result.processing_documents == ["doc_3"] and result.usage.billable_documents == 2


def test_collections():
    sent = []
    col = {"object": "collection", "id": "col_1", "name": "Reports", "documents": 2, "created_at": "2026-10-29T00:00:00Z"}
    doc = {"object": "document", "id": "doc_1", "status": "ready", "filename": "a.pdf", "pages": 1, "bytes": 4,
           "created_at": "2026-10-27T00:00:00Z"}

    def handler(request):
        sent.append(request)
        if request.url.path == "/v1/documents" and request.method == "POST":
            return httpx.Response(200, json=doc)
        if request.url.path == "/v1/documents":
            return httpx.Response(200, json={"object": "list", "data": [doc], "has_more": False})
        return httpx.Response(200, json=col)

    pages = client(handler)
    made = pages.collections.create("Reports")
    assert made.id == "col_1" and made.documents == 2
    pages.collections.add(made, ["doc_1", "doc_2"])
    pages.collections.remove(made, "doc_2")
    pages.documents.upload(b"%PDF-1.4", collection=made, wait=False)
    assert [d.id for d in pages.documents.list(collection="col_1")] == ["doc_1"]
    import json
    assert (sent[1].url.path, json.loads(sent[1].content)) == ("/v1/collections/col_1/documents",
                                                               {"document_ids": ["doc_1", "doc_2"]})
    assert (sent[2].method, sent[2].url.path) == ("DELETE", "/v1/collections/col_1/documents/doc_2")
    assert b'name="collection_id"' in sent[3].content and b"col_1" in sent[3].content
    assert sent[4].url.params["collection_id"] == "col_1"


def test_metadata_and_filters():
    sent = []
    doc = {"object": "document", "id": "doc_1", "status": "ready", "filename": "a.pdf", "pages": 1, "bytes": 4,
           "metadata": {"year": 2018}, "created_at": "2026-10-27T00:00:00Z"}

    def handler(request):
        sent.append(request)
        if request.url.path == "/v1/documents" and request.method == "GET":
            return httpx.Response(200, json={"object": "list", "data": [doc], "has_more": False})
        if request.url.path == "/v1/documents/search":
            return httpx.Response(200, json={"object": "document_search", "id": "s", "results": [],
                                             "failed_documents": [], "processing_documents": [], "usage": None})
        return httpx.Response(200, json=doc)

    import json
    pages = client(handler)
    assert pages.documents.upload(b"%PDF-1.4", metadata={"year": 2018}, wait=False).metadata == {"year": 2018}
    pages.documents.update("doc_1", metadata={"note": None})
    assert [d.metadata for d in pages.documents.list(filter={"year": {"$gte": 2015}})] == [{"year": 2018}]
    pages.documents.search("q", filter={"ticker": "MMM"})
    assert b'name="metadata"' in sent[0].content and b'{"year": 2018}' in sent[0].content
    assert (sent[1].method, json.loads(sent[1].content)) == ("PATCH", {"metadata": {"note": None}})
    assert json.loads(sent[2].url.params["filter"]) == {"year": {"$gte": 2015}}
    assert json.loads(sent[3].content)["filter"] == {"ticker": "MMM"}


def test_search_one_document_or_across():
    import json
    sent = []
    found = {"document_id": "doc_1", "filename": "a.pdf", "page_count": 3, "metadata": {"year": 2026},
             "relevance": None, "pages": [{"page": 2, "relevance": 0.9}], "failed_pages": []}

    def handler(request):
        sent.append(request)
        return httpx.Response(200, json={"object": "search", "id": "srch_1",
                                         "results": [{"question": "q", "documents": [found]}],
                                         "failed_documents": [], "processing_documents": [],
                                         "usage": {"documents": 0, "pages": 3, "questions": 1, "cost_usd": 0.0015,
                                                   "balance_usd": 4.9}, "seconds": 0.5})

    pages = client(handler)
    result = pages.search("q", document="doc_1", top_k=2)
    assert sent[0].url.path == "/v1/search"
    assert json.loads(sent[0].content) == {"questions": ["q"], "document_id": "doc_1", "top_k": 2}
    answer = result.results[0]
    assert answer.documents[0].metadata == {"year": 2026} and answer.documents[0].page_count == 3
    assert [(p.page, p.relevance) for p in answer.pages] == [(2, 0.9)] and result.usage.pages == 3
    pages.search(["q1", "q2"], collection="col_1", filter={"year": 2026}, max_documents=2)
    pages.search("q", document_top_k=3)
    assert json.loads(sent[1].content) == {"questions": ["q1", "q2"], "collection_id": "col_1",
                                           "filter": {"year": 2026}, "max_documents": 2}
    assert json.loads(sent[2].content) == {"questions": ["q"], "document_top_k": 3}
    # The call before 0.4 says how it reads now.
    with pytest.raises(TypeError, match="document="):
        pages.search("doc_1", ["q"])
