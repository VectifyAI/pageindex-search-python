"""Page Search API by PageIndex: upload a PDF, ask up to 10 questions, get the pages that answer each one."""

from ._client import (
    Answer, Document, DocumentPage, Documents, Page, PageSearch, PageSearchError, SearchResult, SearchUsage,
)
from ._version import __version__

__all__ = ["PageSearch", "PageSearchError", "Document", "DocumentPage", "Documents", "SearchResult", "Answer", "Page", "SearchUsage", "__version__"]
