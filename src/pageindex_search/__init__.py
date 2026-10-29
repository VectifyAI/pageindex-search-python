"""Page Search API by PageIndex: upload a PDF, ask up to 10 questions, get the pages that answer each one."""

from ._client import (
    Answer, Collection, FoundDocument, CollectionPage, Collections, Document, DocumentAnswer, DocumentMatch, DocumentPage,
    Documents, DocumentSearchResult, DocumentSearchUsage, Page, PageSearch, PageSearchError, SearchResult,
    SearchUsage,
)
from ._version import __version__

__all__ = ["PageSearch", "PageSearchError", "Document", "DocumentPage", "Documents", "SearchResult", "Answer", "Page",
           "SearchUsage", "FoundDocument", "Collection", "CollectionPage", "Collections", "DocumentSearchResult", "DocumentAnswer",
           "DocumentMatch", "DocumentSearchUsage", "__version__"]
