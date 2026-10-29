"""The Page Search API client: upload a PDF, then search its pages with up to 10 questions at once."""

from __future__ import annotations

import builtins
import json
import os
import time
import uuid
import warnings
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, BinaryIO, Iterator, Sequence, Union

import httpx

from ._version import __version__

DEFAULT_BASE_URL = "https://api.search.pageindex.ai/v1"
API_KEY_ENV = "PAGEINDEX_SEARCH_API_KEY"
# Retried, after a short wait that doubles: the server was busy or briefly unreachable.
RETRY_STATUSES = {429, 500, 502, 503, 504}

FileInput = Union[str, os.PathLike, bytes, BinaryIO]


class PageSearchError(Exception):
    """An error the API returned, or a document that could not be made."""

    def __init__(self, message: str, *, status: int | None = None, code: str | None = None,
                 request_id: str | None = None):
        super().__init__(message)
        self.message = message
        self.status = status
        self.code = code
        self.request_id = request_id

    def __str__(self) -> str:
        parts = [f"{self.code}: {self.message}" if self.code else self.message]
        if self.status:
            parts.append(f"(HTTP {self.status})")
        if self.request_id:
            parts.append(f"[{self.request_id}]")
        return " ".join(parts)


def _make(cls, data: dict[str, Any]):
    """A dataclass from a response, ignoring fields this version does not know."""
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class Document:
    id: str
    status: str  # "processing", "ready" or "failed"
    filename: str | None = None
    pages: int | None = None
    bytes: int | None = None
    created_at: str | None = None
    error: str | None = None
    duplicate: bool = False  # the same file was already uploaded; this is that document
    metadata: dict[str, Any] = field(default_factory=dict)  # your own keys: strings, numbers and booleans


@dataclass
class Collection:
    id: str
    name: str
    documents: int = 0  # how many documents it holds
    created_at: str | None = None


@dataclass
class Page:
    page: int  # numbered from 1
    relevance: float  # from 0 to 1: how likely the page helps answer the question


@dataclass
class Answer:
    question: str
    pages: list[Page]  # most relevant first


@dataclass
class SearchUsage:
    pages: int
    questions: int
    billable_pages: int
    cost_usd: float
    balance_usd: float


@dataclass
class SearchResult:
    id: str
    document_id: str
    results: list[Answer]  # one for each question, in the order asked
    failed_pages: list[int] = field(default_factory=list)  # pages that could not be read; not charged
    usage: SearchUsage | None = None
    seconds: float | None = None

    @classmethod
    def _from(cls, data: dict[str, Any]) -> "SearchResult":
        out = _make(cls, data)
        out.results = [Answer(a["question"], [_make(Page, p) for p in a["pages"]]) for a in data["results"]]
        out.usage = _make(SearchUsage, data["usage"]) if data.get("usage") else None
        return out


@dataclass
class DocumentMatch:
    document_id: str
    relevance: float  # from 0 to 1: how likely the document holds what answers the question
    filename: str | None = None
    pages: int | None = None


@dataclass
class DocumentAnswer:
    question: str
    documents: list[DocumentMatch]  # most relevant first


@dataclass
class DocumentSearchUsage:
    documents: int
    questions: int
    billable_documents: int  # each priced as a page
    cost_usd: float
    balance_usd: float


@dataclass
class DocumentSearchResult:
    id: str
    results: list[DocumentAnswer]  # one for each question, in the order asked
    failed_documents: list[str] = field(default_factory=list)  # could not be judged just now; not charged
    processing_documents: list[str] = field(default_factory=list)  # still being made; not searched
    usage: DocumentSearchUsage | None = None
    seconds: float | None = None

    @classmethod
    def _from(cls, data: dict[str, Any]) -> "DocumentSearchResult":
        out = _make(cls, data)
        out.results = [DocumentAnswer(a["question"], [_make(DocumentMatch, d) for d in a["documents"]])
                       for a in data["results"]]
        out.usage = _make(DocumentSearchUsage, data["usage"]) if data.get("usage") else None
        return out


class DocumentPage:
    """One page of your documents, newest first: data, and has_more. Iterating it goes on through the pages
    after it, so a for loop over client.documents.list() sees every document."""

    def __init__(self, documents: "Documents", data: list[Document], has_more: bool, limit: int,
                 collection: Collection | str | None = None, filter: dict[str, Any] | None = None):
        self._documents, self.data, self.has_more, self._limit = documents, data, has_more, limit
        self._collection, self._filter = collection, filter

    def next_page(self) -> DocumentPage | None:
        if not self.has_more or not self.data:
            return None
        return self._documents.list(limit=self._limit, after=self.data[-1].id, collection=self._collection,
                                    filter=self._filter)

    def __iter__(self) -> Iterator[Document]:
        page: DocumentPage | None = self
        while page is not None:
            yield from page.data
            page = page.next_page()


class CollectionPage:
    """One page of your collections, newest first: data, and has_more. Iterating it goes on through the pages
    after it, as DocumentPage does."""

    def __init__(self, collections: "Collections", data: list[Collection], has_more: bool, limit: int):
        self._collections, self.data, self.has_more, self._limit = collections, data, has_more, limit

    def next_page(self) -> CollectionPage | None:
        if not self.has_more or not self.data:
            return None
        return self._collections.list(limit=self._limit, after=self.data[-1].id)

    def __iter__(self) -> Iterator[Collection]:
        page: CollectionPage | None = self
        while page is not None:
            yield from page.data
            page = page.next_page()


class Documents:
    """Your documents: client.documents.upload(file), .get(id), .list(), .delete(document), and .search(questions)
    to find which of them answer a question."""

    def __init__(self, client: "PageSearch"):
        self._client = client

    def upload(self, file: FileInput, *, filename: str | None = None, collection: Collection | str | None = None,
               metadata: dict[str, Any] | None = None, wait: bool = True, timeout: float = 300.0) -> Document:
        """Upload a PDF: a path, bytes or an open binary file. Uploading the same file again returns its document.
        With collection, the document is put in that collection too; metadata is set on it, as update() does.

        With wait (the default), returns once the document is ready to search, or raises if it failed; most are
        ready in a second or two. Without it, returns at once; the document may still be processing.
        """
        if isinstance(file, (str, os.PathLike)):
            path = Path(file)
            with path.open("rb") as f:
                data = f.read()
            filename = filename or path.name
        elif isinstance(file, bytes):
            data = file
        else:
            data = file.read()
            filename = filename or Path(getattr(file, "name", "") or "document.pdf").name
        form: dict[str, str] = {}
        if collection is not None:
            form["collection_id"] = _id(collection)
        if metadata is not None:
            form["metadata"] = json.dumps(metadata)
        response = self._client._request("POST", "/documents", data=form or None,
                                         files={"file": (filename or "document.pdf", data, "application/pdf")})
        doc = _make(Document, response)
        return self.wait(doc, timeout=timeout) if wait else doc

    def wait(self, document: Document | str, *, timeout: float = 300.0, interval: float = 1.0) -> Document:
        """The document once it is ready; raises PageSearchError if it failed or is not ready within timeout."""
        doc = document if isinstance(document, Document) else self.get(document)
        deadline = time.monotonic() + timeout
        while doc.status == "processing":
            if time.monotonic() > deadline:
                raise PageSearchError(f"Document {doc.id} is still processing after {timeout:.0f} s.")
            time.sleep(interval)
            doc = self.get(doc.id)
        if doc.status == "failed":
            raise PageSearchError(doc.error or f"Document {doc.id} could not be processed.", code="document_failed")
        return doc

    def get(self, document_id: str) -> Document:
        return _make(Document, self._client._request("GET", f"/documents/{document_id}"))

    def update(self, document: Document | str, *, metadata: dict[str, Any]) -> Document:
        """Set metadata keys (strings, numbers or booleans; at most 16 keys); a key set to None is removed, and keys
        not named stay."""
        return _make(Document, self._client._request("PATCH", f"/documents/{_id(document)}",
                                                     json={"metadata": metadata}))

    def list(self, *, limit: int = 20, after: str | None = None, collection: Collection | str | None = None,
             filter: dict[str, Any] | None = None) -> DocumentPage:
        """Your documents, newest first, or those in a collection, or whose metadata meets filter (such as
        {"year": {"$gte": 2020}}); limit (1 to 100) a page; after: the last id of the page before.

            for doc in client.documents.list():          # every document, a page at a time
            page = client.documents.list(limit=10)       # one page: page.data, page.has_more
        """
        params: dict[str, Any] = {"limit": limit}
        if after:
            params["after"] = after
        if collection is not None:
            params["collection_id"] = _id(collection)
        if filter is not None:
            params["filter"] = json.dumps(filter)
        page = self._client._request("GET", "/documents", params=params)
        return DocumentPage(self, [_make(Document, d) for d in page["data"]], page["has_more"], limit,
                            collection, filter)

    def delete(self, document: Document | str) -> None:
        """Delete a document and its pages; searches of it then fail."""
        self._client._request("DELETE", f"/documents/{_id(document)}")

    def search(self, questions: str | Sequence[str], *, document_ids: Sequence[Document | str] | None = None,
               collection: Collection | str | None = None, filter: dict[str, Any] | None = None,
               top_k: int | None = None, idempotency_key: str | None = None) -> DocumentSearchResult:
        """Ask up to 10 questions of your documents. For each, the documents likely to hold the answer, most
        relevant first; then search those documents' pages with client.search().

        Every document of yours (up to 1,000), only those in document_ids, or only those in a collection and those
        whose metadata meets filter. Without
        top_k, each answer lists the documents likely to help, possibly none; with it, exactly top_k. Each document
        is priced as a page. Retries send the same idempotency_key, as client.search() does.
        """
        body: dict[str, Any] = {"questions": [questions] if isinstance(questions, str) else list(questions)}
        if document_ids is not None:
            body["document_ids"] = [_id(d) for d in document_ids]
        if collection is not None:
            body["collection_id"] = _id(collection)
        if filter is not None:
            body["filter"] = filter
        if top_k is not None:
            body["top_k"] = top_k
        headers = {"Idempotency-Key": idempotency_key or str(uuid.uuid4())}
        return DocumentSearchResult._from(self._client._request("POST", "/documents/search", json=body,
                                                                headers=headers))

    def __call__(self, *, limit: int = 20, after: str | None = None) -> builtins.list[Document]:
        _renamed("client.documents()", "client.documents.list().data")
        return self.list(limit=limit, after=after).data


class Collections:
    """Named groups of your documents, to search within: client.collections.create(name), .list(), .get(id),
    .rename(collection, name), .delete(collection), .add(collection, documents), .remove(collection, document).
    A document may be in many collections; deleting a collection leaves its documents."""

    def __init__(self, client: "PageSearch"):
        self._client = client

    def create(self, name: str) -> Collection:
        return _make(Collection, self._client._request("POST", "/collections", json={"name": name}))

    def list(self, *, limit: int = 20, after: str | None = None) -> CollectionPage:
        """Your collections, newest first, limit (1 to 100) a page. Loop over it for every collection."""
        params: dict[str, Any] = {"limit": limit}
        if after:
            params["after"] = after
        page = self._client._request("GET", "/collections", params=params)
        return CollectionPage(self, [_make(Collection, c) for c in page["data"]], page["has_more"], limit)

    def get(self, collection_id: str) -> Collection:
        return _make(Collection, self._client._request("GET", f"/collections/{collection_id}"))

    def rename(self, collection: Collection | str, name: str) -> Collection:
        return _make(Collection, self._client._request("PATCH", f"/collections/{_id(collection)}",
                                                       json={"name": name}))

    def delete(self, collection: Collection | str) -> None:
        """Delete the collection; its documents stay."""
        self._client._request("DELETE", f"/collections/{_id(collection)}")

    def add(self, collection: Collection | str, documents: Sequence[Document | str]) -> Collection:
        """Put documents in the collection (up to 1,000 at once); those already in it stay."""
        return _make(Collection, self._client._request("POST", f"/collections/{_id(collection)}/documents",
                                                       json={"document_ids": [_id(d) for d in documents]}))

    def remove(self, collection: Collection | str, document: Document | str) -> Collection:
        """Take a document out of the collection; the document stays."""
        return _make(Collection, self._client._request(
            "DELETE", f"/collections/{_id(collection)}/documents/{_id(document)}"))


def _renamed(old: str, new: str) -> None:
    warnings.warn(f"{old} is now {new}; the old name goes in a later version.", DeprecationWarning, stacklevel=3)


class PageSearch:
    """A client for the Page Search API.

        client = PageSearch()  # reads PAGEINDEX_SEARCH_API_KEY
        doc = client.documents.upload("report.pdf")
        result = client.search(doc, ["What was revenue in 2024?"])
        for answer in result.results:
            print(answer.question, [(p.page, p.relevance) for p in answer.pages])
    """

    def __init__(self, api_key: str | None = None, *, base_url: str | None = None, timeout: float = 120.0,
                 max_retries: int = 2, http_client: httpx.Client | None = None):
        api_key = api_key or os.environ.get(API_KEY_ENV)
        if not api_key:
            raise PageSearchError(f"No API key: pass api_key or set {API_KEY_ENV}. "
                                  "Create one at https://search.pageindex.ai/console/keys.")
        self.base_url = (base_url or os.environ.get("PAGEINDEX_SEARCH_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.max_retries = max_retries
        self._http = http_client or httpx.Client(timeout=timeout)
        self._headers = {"Authorization": f"Bearer {api_key}", "User-Agent": f"pageindex-search-python/{__version__}"}
        self.documents = Documents(self)
        self.collections = Collections(self)

    # The names before 0.2, kept for a while.

    def upload(self, file: FileInput, **kwargs: Any) -> Document:
        _renamed("client.upload()", "client.documents.upload()")
        return self.documents.upload(file, **kwargs)

    def wait(self, document: Document | str, **kwargs: Any) -> Document:
        _renamed("client.wait()", "client.documents.wait()")
        return self.documents.wait(document, **kwargs)

    def document(self, document_id: str) -> Document:
        _renamed("client.document()", "client.documents.get()")
        return self.documents.get(document_id)

    def iter_documents(self) -> Iterator[Document]:
        _renamed("client.iter_documents()", "client.documents.list()")
        return iter(self.documents.list(limit=100))

    def delete(self, document: Document | str) -> None:
        _renamed("client.delete()", "client.documents.delete()")
        self.documents.delete(document)

    # Search

    def search(self, document: Document | str, questions: str | Sequence[str], *,
               top_k: int | None = None, idempotency_key: str | None = None) -> SearchResult:
        """Ask up to 10 questions about a document. For each, the pages that help answer it, most relevant first.

        Without top_k, each answer lists the pages likely to help, possibly none. With it, exactly top_k pages.

        Retries send the same idempotency_key (a new one by default), so a search retried after a timeout is not
        run or charged twice; pass your own to make your own retries safe too.
        """
        body: dict[str, Any] = {"questions": [questions] if isinstance(questions, str) else list(questions)}
        if top_k is not None:
            body["top_k"] = top_k
        headers = {"Idempotency-Key": idempotency_key or str(uuid.uuid4())}
        return SearchResult._from(self._request("POST", f"/documents/{_id(document)}/search", json=body, headers=headers))

    # Account

    def usage(self, *, start: str | None = None, end: str | None = None) -> dict[str, Any]:
        """Uploads, searches and cost by day (UTC, dates as YYYY-MM-DD; the last 30 days by default), your
        credit balance and the pages you have stored."""
        params = {k: v for k, v in {"start": start, "end": end}.items() if v}
        return self._request("GET", "/usage", params=params)

    # Plumbing

    def _request(self, method: str, path: str, headers: dict[str, str] | None = None, **kwargs: Any) -> dict[str, Any]:
        headers = self._headers | (headers or {})  # the same on every attempt
        for attempt in range(self.max_retries + 1):
            try:
                response = self._http.request(method, self.base_url + path, headers=headers, **kwargs)
            except httpx.TransportError as error:
                if attempt == self.max_retries:
                    raise PageSearchError(f"Could not reach the API: {error}") from error
            else:
                if response.status_code not in RETRY_STATUSES or attempt == self.max_retries:
                    return _json(response)
            time.sleep(0.5 * 2 ** attempt)
        raise AssertionError("unreachable")

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "PageSearch":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _id(item: Document | Collection | str) -> str:
    return item if isinstance(item, str) else item.id


def _json(response: httpx.Response) -> dict[str, Any]:
    try:
        data = response.json()
    except ValueError:
        data = None
    if response.is_success and isinstance(data, dict):
        return data
    request_id = response.headers.get("x-request-id")
    if isinstance(data, dict) and data.get("object") == "document" and data.get("status") == "failed":
        # An upload whose file could not be read answers with the document, failed, and why.
        raise PageSearchError(data.get("error") or f"Document {data.get('id')} could not be read.",
                              status=response.status_code, code="document_failed", request_id=request_id)
    error = data.get("error") if isinstance(data, dict) else None
    error = error if isinstance(error, dict) else {}
    raise PageSearchError(error.get("message") or response.text[:200] or response.reason_phrase,
                          status=response.status_code, code=error.get("code"),
                          request_id=error.get("request_id") or request_id)
