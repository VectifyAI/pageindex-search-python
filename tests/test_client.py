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
        return httpx.Response(200, json={"object": "search", "id": "srch_1", "document_id": "doc_1", "results": [],
                                         "failed_pages": [], "usage": None, "seconds": 0.1})

    pages = PageSearch(api_key="pis_live_x", max_retries=1,
                       http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    pages.search("doc_1", "q")
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
