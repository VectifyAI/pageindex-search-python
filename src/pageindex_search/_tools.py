"""Tools an LLM agent can call: list documents by their metadata, find which documents answer, search the pages of the ones it picks (with their
text), read pages, with pictures of those best seen, and zoom in on a part of a page too small to read.

    tools = client.tools(collection=filings)
    response = anthropic.messages.create(..., tools=tools.definitions)
    for block in response.content:
        if block.type == "tool_use":
            content = tools.run(block.name, block.input).anthropic()  # the tool_result content
"""
from __future__ import annotations

import base64
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._client import Collection, PageSearch, Schema

MAX_PAGES_READ = 20  # pages one read_pages call may read
MAX_IMAGES = 5  # pictures one read_pages call may give: each is some thousands of tokens
MAX_QUESTIONS = 10  # questions one search may ask
MAX_LISTED = 100  # documents one list_documents call may list
CORNERS = ("x0", "y0", "x1", "y1")
BOX = {"type": "object", "properties": {k: {"type": "number", "minimum": 0, "maximum": 1} for k in CORNERS},
       "required": list(CORNERS), "additionalProperties": False,
       "description": "The part to see, as fractions of the page: x from 0 at its left edge to 1 at its right, y from "
                      "0 at its top to 1 at its bottom; x0, y0 the top left corner, x1, y1 the bottom right. "
                      '{"x0": 0, "y0": 0.3, "x1": 1, "y1": 0.6} is the band from 30% to 60% of the way down, full '
                      "width."}

# The notes on a page whose read is "image", in search_pages and in read_pages.
PICTURED = "A picture: its text may be incomplete or wrong. See it with read_pages before answering from it."
SEE_PICTURE = ("A picture: its text may be incomplete or wrong. Read what you need from the picture, zooming in "
               "with zoom_page where the print is small.")

QUESTIONS = {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": MAX_QUESTIONS,
             "description": "The questions, as a person would ask them: several at once cost much less than one "
                            "at a time, so ask together the questions you have of the same documents."}

FILTERS = ('Conditions: {"key": value} equals; {"key": {"$ne"|"$in"|"$nin"|"$gt"|"$gte"|"$lt"|"$lte": ...}}; '
           '{"$and": [...]} and {"$or": [...]} combine filters. Values match exactly, type included: "2018" is not '
           '2018.')


def describe(schema: Schema) -> str:
    """The metadata keys, their types and values, as a line each, for the agent to write filters from."""
    if not schema.keys:
        return "The documents have no metadata, so there is nothing to filter on."
    lines = [f"Metadata keys of the {schema.documents} documents:"]
    for key in schema.keys:
        line = f"- {key.key}: {' or '.join(key.types)}, on {key.documents} documents"
        if key.min is not None:
            line += f"; {_number(key.min)} to {_number(key.max)}"
        if key.values:
            shown = ", ".join(f"{json.dumps(v['value'])} ({v['documents']})" for v in key.values)
            more = sum(v["documents"] for v in key.values) < key.documents
            line += f"; values: {shown}" + (", and others" if more else "")
        lines.append(line)
    return "\n".join(lines)


def _top_k(what: str) -> dict[str, Any]:
    return {"anyOf": [{"type": "string", "enum": ["auto"]}, {"type": "integer", "minimum": 1, "maximum": 50}],
            "description": f"\"auto\" (the default, and usually best): the {what} likely to help. A number: exactly "
                           "that many, most relevant first, only when auto did not find enough."}


def _asked(questions: list[str] | str) -> list[str]:
    """The questions; one given alone, as models sometimes do, is a list of one."""
    return [questions] if isinstance(questions, str) else list(questions)


def _auto(top_k: int | str | None) -> int | None:
    return None if top_k in (None, "auto") else int(top_k)


def _number(n: float | None) -> str:
    return str(int(n)) if n is not None and float(n).is_integer() else str(n)


@dataclass
class PageImage:
    """A page as PNG bytes, or, with box, part of it: x0, y0, x1, y1."""
    document_id: str
    page: int
    png: bytes
    box: dict[str, float] | None = None

    @property
    def label(self) -> str:
        """What the picture is, to put before it: "Page 7 of doc_...:"."""
        part = "" if self.box is None else ", " + ", ".join(f"{k} {_number(v)}" for k, v in self.box.items())
        return f"Page {self.page} of {self.document_id}{part}:"

    @property
    def base64(self) -> str:
        return base64.b64encode(self.png).decode()

    @property
    def data_url(self) -> str:
        """For OpenAI: data:image/png;base64,..."""
        return "data:image/png;base64," + self.base64


@dataclass
class ToolResult:
    """What a tool gives back: JSON text, and from read_pages, pictures of pages."""
    text: str
    images: list[PageImage] = field(default_factory=list)

    def anthropic(self) -> str | list[dict[str, Any]]:
        """The tool_result content for Claude: the text, then each picture after its page number."""
        if not self.images:
            return self.text
        content: list[dict[str, Any]] = [{"type": "text", "text": self.text}]
        for image in self.images:
            content.append({"type": "text", "text": image.label})
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                        "data": image.base64}})
        return content


class Tools:
    """list_documents, search_documents, search_pages, read_pages, view_pages and zoom_page over a scope: a collection, documents whose
    metadata meets a filter, or all of yours. The metadata schema is read once, when made; make new ones when the documents change."""

    def __init__(self, client: PageSearch, *, collection: Collection | str | None = None,
                 filter: dict[str, Any] | None = None):
        from ._client import _id
        self._client = client
        self._collection = _id(collection) if collection is not None else None
        self._filter = filter
        if self._collection is not None and filter is None:
            self.schema = client.collections.schema(self._collection)
        else:
            self.schema = client.documents.schema(collection=self._collection, filter=filter)
        self.definitions = [self._list_documents(), self._search_documents(), self._search_pages(), self._read_pages(),
                            self._view_pages(), self._zoom_page()]

    @property
    def openai(self) -> list[dict[str, Any]]:
        """The definitions in OpenAI's function format."""
        return [{"type": "function", "function": {"name": d["name"], "description": d["description"],
                                                  "parameters": d["input_schema"]}} for d in self.definitions]

    def openai_agents(self) -> list[Any]:
        """The tools as OpenAI Agents SDK FunctionTools, for Agent(tools=...): pictures of pages go to the model as
        images. Needs openai-agents (pip install "pageindex-search[agents]")."""
        import asyncio

        try:
            from agents import FunctionTool, ToolOutputImage, ToolOutputText
        except ImportError:
            raise ImportError('tools.openai_agents() needs openai-agents: pip install "pageindex-search[agents]"') from None

        def tool(definition: dict[str, Any]) -> Any:
            async def invoke(ctx: Any, args: str) -> Any:
                out = await asyncio.to_thread(self.run, definition["name"], json.loads(args or "{}"))
                if not out.images:
                    return out.text
                parts: list[Any] = [ToolOutputText(text=out.text)]
                for image in out.images:
                    parts += [ToolOutputText(text=image.label), ToolOutputImage(image_url=image.data_url, detail="high")]
                return parts
            # filter is any object, which a strict schema cannot hold.
            return FunctionTool(name=definition["name"], description=definition["description"],
                                params_json_schema=definition["input_schema"], on_invoke_tool=invoke,
                                strict_json_schema=False)

        return [tool(d) for d in self.definitions]

    def _list_documents(self) -> dict[str, Any]:
        return {
            "name": "list_documents",
            "description": (
                f"List the documents, of {self.schema.documents}, whose metadata meets a filter, newest first, each "
                "with its id, file name, page count, upload time and metadata, and how many match in all. Free. When the question says which documents it "
                "is about in terms the metadata holds, find them here and search their pages with search_pages; use "
                "search_documents, which is priced, only when a filter cannot pick them out: none match, too many "
                "do, or the question says only what it asks.\n\n" + describe(self.schema) + "\n\n" + FILTERS
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "filter": {"type": "object", "description": "Only documents whose metadata meets this; without "
                                                                "it, all."},
                    "limit": {"type": "integer", "minimum": 1, "maximum": MAX_LISTED,
                              "description": "At most this many documents (20 by default)."},
                    "offset": {"type": "integer", "minimum": 0,
                               "description": "Skip this many first, for the next page: the offset a list gave back."},
                },
            },
        }

    def _search_documents(self) -> dict[str, Any]:
        return {
            "name": "search_documents",
            "description": (
                f"Find which of {self.schema.documents} documents likely answer each question, most relevant first, "
                "with each one's relevance (0 to 1), page count and metadata; always at least one. Priced by the "
                "documents it looks at: when a metadata filter alone picks the documents, use list_documents, which "
                "is free, and when it narrows them, pass it here. Then call search_pages with the documents worth "
                "reading. Searching a document's pages is priced by its pages, so search the most relevant documents "
                "first, and others only when the question needs them or those did not answer. Use top_k \"auto\": "
                "it returns the documents likely to help, which is best."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "questions": QUESTIONS,
                    "filter": {"type": "object", "description": "Only documents whose metadata meets this, written "
                                                                "as for list_documents."},
                    "top_k": _top_k("documents"),
                },
                "required": ["questions"],
            },
        }

    def _search_pages(self) -> dict[str, Any]:
        return {
            "name": "search_pages",
            "description": (
                "Find the pages of one document that help answer each question, most relevant first, each with its "
                "relevance (0 to 1) and text. Pages are numbered from 1. It is priced by the document's "
                "pages: search the documents search_documents ranks most relevant first, calling it for each at the "
                "same time, and others only when needed; to ask again in other words, ask only "
                "of the documents still missing the answer. A page whose read is "
                "\"image\" is a picture, such as a scan or a chart: its text may be incomplete or empty, so look at "
                "it with read_pages. Use top_k \"auto\": it returns the pages "
                "likely to help, which is best, and a number brings back many pages of text. The most relevant pages "
                "usually hold the answer: once they do, stop searching. If they do not, more_pages lists the next "
                "most relevant: read those with read_pages, which is free, before searching again."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "questions": QUESTIONS,
                    "document_id": {"type": "string", "description": "The document to search, from search_documents."},
                    "top_k": _top_k("pages"),
                },
                "required": ["questions", "document_id"],
            },
        }

    def _read_pages(self) -> dict[str, Any]:
        return {
            "name": "read_pages",
            "description": (
                "Read pages of a document, numbered from 1: their text, with a picture of any page whose read is "
                "\"image\" (a scan or a chart, whose text may be incomplete). Use it for pages search_pages found "
                "that are pictures, or for the pages around one, such as where a table goes on. At most "
                f"{MAX_PAGES_READ} pages at a time, and at most {MAX_IMAGES} of them pictures: in a scanned document "
                f"every page is a picture, so read {MAX_IMAGES} at a time. If a picture's small print or figures are hard "
                "to read, zoom in with zoom_page."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "document_id": {"type": "string"},
                    "pages": {"type": "array", "items": {"type": "integer", "minimum": 1}, "minItems": 1,
                              "maxItems": MAX_PAGES_READ},
                },
                "required": ["document_id", "pages"],
            },
        }

    def _view_pages(self) -> dict[str, Any]:
        return {
            "name": "view_pages",
            "description": (
                f"See pages of a document as pictures, with their text: up to {MAX_IMAGES} at a time, numbered from "
                "1. Use it where how a page looks matters, such as a table's layout or a chart on a page of text. If "
                "small print or figures are hard to read, zoom in with zoom_page before answering."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "document_id": {"type": "string"},
                    "pages": {"type": "array", "items": {"type": "integer", "minimum": 1}, "minItems": 1,
                              "maxItems": MAX_IMAGES},
                },
                "required": ["document_id", "pages"],
            },
        }

    def _zoom_page(self) -> dict[str, Any]:
        return {
            "name": "zoom_page",
            "description": (
                "See part of a page, sharper: for small print, a dense table's figures or a footnote that are hard to "
                "read in a picture from read_pages or view_pages. Look at the whole page first, then zoom on the part "
                "you need, a band no taller than about a third of the page. Free, as reading pages is."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "document_id": {"type": "string"},
                    "page": {"type": "integer", "minimum": 1},
                    "box": BOX,
                },
                "required": ["document_id", "page", "box"],
            },
        }

    def run(self, name: str, input: dict[str, Any]) -> ToolResult:
        """Call a tool the agent asked for; what to send back as its result. An error comes back as
        {"error": ...}, for the agent to correct."""
        from ._client import PageSearchError
        if "question" in input and "questions" not in input:  # one question, as a model may send it
            input = {**input, "questions": input["question"]}
            del input["question"]
        try:
            if name == "list_documents":
                return ToolResult(_text(self.list_documents(**input)))
            if name == "search_documents":
                return ToolResult(_text(self.search_documents(**input)))
            if name == "search_pages":
                return ToolResult(_text(self.search_pages(**input)))
            if name == "read_pages":
                return self.read_pages(**input)
            if name == "view_pages":
                return self.view_pages(**input)
            if name == "zoom_page":
                return self.zoom_page(**input)
            out = {"error": f"No tool {name}."}
        except PageSearchError as error:
            out = {"error": error.message}
        except (TypeError, ValueError) as error:
            out = {"error": f"Bad input for {name}: {error}"}
        return ToolResult(_text(out))

    def list_documents(self, filter: dict[str, Any] | None = None, limit: int = 20, offset: int = 0) -> dict[str, Any]:
        if not 1 <= int(limit) <= MAX_LISTED:
            raise ValueError(f"limit is 1 to {MAX_LISTED}")
        if int(offset) < 0:
            raise ValueError("offset is 0 or more")
        page = self._client.documents.list(limit=int(limit), offset=int(offset), collection=self._collection,
                                           filter=self._scope(filter))
        out: dict[str, Any] = {"total": page.total, "documents": [
            {"document_id": d.id, "filename": d.filename, "pages": d.pages, "status": d.status,
             "created_at": d.created_at, "metadata": d.metadata} for d in page.data]}
        if page.has_more and page.data:
            last = page.offset + len(page.data)
            out["more"] = (f"Documents {page.offset + 1} to {last} of {page.total}. Next page: offset {last}, with "
                           "the same filter; or narrow the filter.")
        return out

    def search_documents(self, questions: list[str] | str, filter: dict[str, Any] | None = None,
                         top_k: int | str | None = None) -> dict[str, Any]:
        scope = self._scope(filter)
        found = self._client.documents.search(_asked(questions), collection=self._collection, filter=scope,
                                              top_k=_auto(top_k))
        return {"results": [{"question": a.question,
                             "documents": [{"document_id": d.document_id, "filename": d.filename, "pages": d.pages,
                                            "relevance": d.relevance, "metadata": d.metadata} for d in a.documents]}
                            for a in found.results]}

    def _scope(self, filter: dict[str, Any] | None) -> dict[str, Any] | None:
        """The filter the agent gave, within the one the tools were made with."""
        return self._filter if filter is None else filter if self._filter is None else {"$and": [self._filter, filter]}

    def search_pages(self, questions: list[str] | str, document_id: str, top_k: int | str | None = None) -> dict[str, Any]:
        found = self._client.search(_asked(questions), document=document_id, top_k=_auto(top_k), include_text=True)
        first = found.results[0].documents[0]
        results = []
        for a in found.results:
            d = a.documents[0]
            out: dict[str, Any] = {"question": a.question, "pages": [
                {"page": p.page, "relevance": p.relevance, "read": p.read, "text": p.text,
                 **({"note": PICTURED} if p.read == "image" else {})} for p in d.pages]}
            if d.more_pages:
                out["more_pages"] = [{"page": p.page, "relevance": p.relevance, "read": p.read} for p in d.more_pages]
            results.append(out)
        return {"document_id": first.document_id, "filename": first.filename, "results": results}

    def read_pages(self, document_id: str, pages: list[int]) -> ToolResult:
        return self._pages(document_id, pages, MAX_PAGES_READ, lambda p: p.read == "image")

    def view_pages(self, document_id: str, pages: list[int]) -> ToolResult:
        return self._pages(document_id, pages, MAX_IMAGES, lambda p: True)

    def zoom_page(self, document_id: str, page: int, box: dict[str, float]) -> ToolResult:
        if not isinstance(box, dict) or set(box) != set(CORNERS):
            raise ValueError('box is {"x0": ..., "y0": ..., "x1": ..., "y1": ...}, fractions of the page')
        box = {k: float(box[k]) for k in CORNERS}
        png = self._client.documents.page_image(document_id, page, box=[box[k] for k in CORNERS])
        return ToolResult(_text({"document_id": document_id, "page": page, "box": box}),
                          [PageImage(document_id, page, png, box)])

    def _pages(self, document_id: str, pages: list[int], most: int, pictured_if) -> ToolResult:
        """The pages' text, and pictures of those pictured_if picks."""
        named = list(dict.fromkeys(pages))
        if not 1 <= len(named) <= most:
            return ToolResult(_text({"error": f"Name 1 to {most} pages."}))
        texts = self._client.documents.pages(document_id, named)
        pictured = [p.page for p in texts if pictured_if(p)]
        if len(pictured) > MAX_IMAGES:
            return ToolResult(_text({"error": f"Pages {', '.join(map(str, pictured))} are pictures; read at most "
                                              f"{MAX_IMAGES} of them at a time."}))
        with ThreadPoolExecutor(MAX_IMAGES) as pool:
            pngs = list(pool.map(lambda page: self._client.documents.page_image(document_id, page), pictured))
        out = {"document_id": document_id,
               "pages": [{"page": p.page, "read": p.read, "text": p.text,
                          **({"note": SEE_PICTURE} if p.read == "image" else {})}
                         for p in texts]}
        return ToolResult(_text(out), [PageImage(document_id, page, png) for page, png in zip(pictured, pngs)])


def _text(out: dict[str, Any]) -> str:
    return json.dumps(out, ensure_ascii=False)
