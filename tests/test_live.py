"""Against the live API, with a real key and a real PDF; skipped without them.

    PAGEINDEX_SEARCH_API_KEY=pis_live_... PAGE_SEARCH_TEST_PDF=report.pdf pytest
"""
import os

import pytest

from pageindex_search import PageSearch, PageSearchError

PDF = os.environ.get("PAGE_SEARCH_TEST_PDF")
pytestmark = pytest.mark.skipif(not (PDF and os.environ.get("PAGEINDEX_SEARCH_API_KEY")), reason="needs a key and a PDF")


def test_upload_search_and_usage():
    with PageSearch() as client:
        doc = client.documents.upload(PDF)
        assert doc.status == "ready" and doc.pages
        assert client.documents.upload(PDF).id == doc.id  # the same file is the same document

        result = client.search(["What is this document about?", "Who wrote it?"], document=doc, top_k=3)
        assert [a.question for a in result.results] == ["What is this document about?", "Who wrote it?"]
        assert all(len(a.pages) == 3 for a in result.results)
        pages = result.results[0].pages
        assert all(1 <= p.page <= doc.pages and 0 <= p.relevance <= 1 for p in pages)
        assert pages == sorted(pages, key=lambda p: -p.relevance)
        assert result.usage.pages == doc.pages and result.usage.documents == 0

        assert any(d.id == doc.id for d in client.documents.list(limit=100).data)
        assert "balance_usd" in client.usage()


def test_errors():
    with PageSearch() as client:
        with pytest.raises(PageSearchError) as error:
            client.documents.get("doc_does_not_exist")
        assert error.value.status == 404 and error.value.code == "document_not_found"
    with pytest.raises(PageSearchError) as error:
        PageSearch(api_key="pis_live_wrong").documents.list()
    assert error.value.status == 401
