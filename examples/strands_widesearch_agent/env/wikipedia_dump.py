"""Wikipedia from local Kiwix ZIM dumps: article pages and the site's search and read APIs, as page text.

Dumps.fetch(lang, url) returns what the live site would return for `url` as page text (or JSON text, for the API
endpoints), computed from the dump:

  /wiki/<title>, /<zh variant>/<title>, /w/index.php?title=<title>   article text (redirects followed)
  /wiki/Special:Search?search=Q, /w/index.php?search=Q               "go" search: the article if Q is an exact
                                                                     title, else a results page (fulltext=1 always
                                                                     gives the results page)
  /w/api.php?action=query&list=search&srsearch=Q                     search results (JSON)
  /w/api.php?action=opensearch&search=Q                              title suggestions (JSON)
  /w/api.php?action=query&prop=extracts|revisions&titles=T           page text (JSON); revisions only without
                                                                     rvstart / rvend / rvdir / rvlimit > 1
  /w/api.php?action=parse&page=T[&prop=sections][&section=N]         page text, section list, or one section (JSON)
  /w/rest.php/v1/search/page?q=Q, /w/rest.php/v1/search/title?q=Q    search / title suggestions (JSON)
  /api/rest_v1/page/{html,mobile-html,summary}/T                     page text; summary is the lead (JSON)

Differences from the live site: search uses the ZIM's full-text (Xapian) and title indexes, so ranking differs from
CirrusSearch; CirrusSearch operators (insource:, intitle:, ...) are reduced to their terms; snippets are the start
of the article lead, not query-highlighted; page text is the dump's rendered HTML as text (no wikitext), and page
ids are not available. Anything else on these hosts (revision history, curid, other API modules) returns
"Error: HTTP 404.", as a missing page does.
"""
import json
import re
import threading
from urllib.parse import parse_qs, unquote, urlparse

from bs4 import BeautifulSoup

ZH_VARIANTS = ("zh", "zh-cn", "zh-tw", "zh-hk", "zh-mo", "zh-sg", "zh-my", "zh-hans", "zh-hant")
NOT_FOUND = "Error: HTTP 404."
CIRRUS_OP = re.compile(
    r"-?\b(?:insource|intitle|incategory|hastemplate|linksto|prefix|inlanguage|deepcat|"
    r"morelike|articletopic|subpageof|filetype|boost-templates):\s*",
    re.I,
)
SNIPPET_CHARS = 300


def _qs(query):
    return parse_qs(query, keep_blank_values=True)


def _q(query, key, default=""):
    return (_qs(query).get(key) or [default])[0]


def _norm_title(title):
    return unquote(title).strip().replace(" ", "_")


def _plain_query(q):
    """CirrusSearch syntax -> plain terms for the Xapian index: operators dropped, /regex/ and quotes kept as words."""
    q = CIRRUS_OP.sub(" ", q)
    q = re.sub(r"/([^/]+)/", r"\1", q)
    return re.sub(r"\s+", " ", q).strip()


class Dumps:
    def __init__(self, paths, html_text):
        """paths: {lang: path to the ZIM}; html_text: the HTML -> text function of the agent's live pages."""
        self.paths, self.html_text = paths, html_text
        self._local = threading.local()

    # ---------------------------------------------------------------- readers
    def _archive(self, lang):
        d = self._local.__dict__.setdefault("archives", {})  # one reader per thread and language
        if lang not in d:
            from libzim.reader import Archive

            d[lang] = Archive(str(self.paths[lang]))
        return d[lang]

    def _item(self, lang, title):
        a = self._archive(lang)
        for t in dict.fromkeys([title[:1].upper() + title[1:], title]):  # first letter upper-cased, then as given
            try:
                return a.get_entry_by_path(t).get_item()  # follows redirects
            except KeyError:
                continue
        return None

    def _page(self, lang, title):
        """(title, html) of an article, or None."""
        item = self._item(lang, _norm_title(title)) if title else None
        if item is None or "html" not in item.mimetype:
            return None
        return item.title, bytes(item.content).decode("utf-8", "replace")

    def _lead(self, html):
        soup = BeautifulSoup(html, "lxml")
        for p in soup.find_all("p"):
            t = re.sub(r"\s+", " ", p.get_text(" ")).strip()
            if len(t) > 40:
                return t
        return ""

    def _search(self, lang, q, limit):
        """(estimated total, [(title, snippet)]) from the full-text index."""
        from libzim.search import Query, Searcher

        a = self._archive(lang)
        s = self._local.__dict__.setdefault("searchers", {})
        if lang not in s:
            s[lang] = Searcher(a)
        q = _plain_query(q)
        if not q:
            return 0, []
        res = s[lang].search(Query().set_query(q))
        out = []
        for path in res.getResults(0, limit):
            try:
                item = a.get_entry_by_path(path).get_item()
            except KeyError:
                continue
            out.append((item.title, self._lead(bytes(item.content).decode("utf-8", "replace"))[:SNIPPET_CHARS]))
        return res.getEstimatedMatches(), out

    def _suggest(self, lang, q, limit):
        from libzim.suggestion import SuggestionSearcher

        a = self._archive(lang)
        s = self._local.__dict__.setdefault("suggesters", {})
        if lang not in s:
            s[lang] = SuggestionSearcher(a)
        return [a.get_entry_by_path(p).title for p in s[lang].suggest(q).getResults(0, limit)]

    # ---------------------------------------------------------------- endpoints
    def _article(self, lang, title):
        page = self._page(lang, title)
        return self.html_text(page[1]) if page else NOT_FOUND

    def _search_page(self, lang, q, fulltext, limit, host):
        if not fulltext and self._page(lang, q):  # MediaWiki "go": exact title -> the article
            return self._article(lang, q)
        total, hits = self._search(lang, q, limit)
        if not hits:
            return f"Search results\n\nThere were no results matching the query: {q}"
        lines = [f"Search results\n\nResults 1 – {len(hits)} of about {total} for: {q}\n"]
        for i, (t, snip) in enumerate(hits, 1):
            lines.append(f"{i}. {t}\nURL: https://{host}/wiki/{t.replace(' ', '_')}\n{snip}\n")
        return "\n".join(lines)

    def _api(self, lang, query):
        action = _q(query, "action")
        if action == "opensearch":
            q, n = _q(query, "search"), int(_q(query, "limit", "10") or 10)
            titles = self._suggest(lang, q, min(n, 50))
            return json.dumps(
                [
                    q,
                    titles,
                    [""] * len(titles),
                    [f"https://{lang}.wikipedia.org/wiki/{t.replace(' ', '_')}" for t in titles],
                ],
                ensure_ascii=False,
                indent=1,
            )
        if action == "query" and _q(query, "list") == "search":
            q, n = _q(query, "srsearch"), int(_q(query, "srlimit", "10") or 10)
            total, hits = self._search(lang, q, min(n, 50))
            return json.dumps(
                {
                    "batchcomplete": "",
                    "query": {
                        "searchinfo": {"totalhits": total},
                        "search": [{"ns": 0, "title": t, "snippet": s} for t, s in hits],
                    },
                },
                ensure_ascii=False,
                indent=1,
            )
        props = set(_q(query, "prop").split("|"))
        if action == "query" and props & {"extracts", "revisions"}:
            if (
                any(k in _qs(query) for k in ("rvstart", "rvend", "rvdir", "rvcontinue"))
                or int(_q(query, "rvlimit", "1") or 1) > 1
            ):
                return NOT_FOUND  # revision history is not in the dump
            pages = {}
            for i, t in enumerate(_q(query, "titles").split("|")):
                page = self._page(lang, t)
                if page is None:
                    pages[str(-1 - i)] = {"ns": 0, "title": t, "missing": ""}
                    continue
                text = self.html_text(page[1])
                entry = {"ns": 0, "title": page[0]}
                if "extracts" in props:
                    entry["extract"] = self._lead(page[1]) if "exintro" in _qs(query) else text
                if "revisions" in props:
                    entry["revisions"] = [{"slots": {"main": {"contentmodel": "text", "*": text}}}]
                pages[f"dump{i}"] = entry
            return json.dumps({"batchcomplete": "", "query": {"pages": pages}}, ensure_ascii=False, indent=1)
        if action == "parse":
            page = self._page(lang, _q(query, "page"))
            if page is None:
                return json.dumps({"error": {"code": "missingtitle", "info": "The page you specified doesn't exist."}})
            title, html = page
            lead, secs = self._walk(html)
            if _q(query, "prop") == "sections":
                return json.dumps(
                    {
                        "parse": {
                            "title": title,
                            "sections": [
                                {"toclevel": lv - 1, "level": str(lv), "line": h, "number": str(i), "index": str(i)}
                                for i, (lv, h, _) in enumerate(secs, 1)
                            ],
                        }
                    },
                    ensure_ascii=False,
                    indent=1,
                )
            if "section" in _qs(query):
                i = int(_q(query, "section") or 0)
                body = lead if i == 0 else (secs[i - 1][2] if 0 < i <= len(secs) else None)
                if body is None:
                    return json.dumps({"error": {"code": "nosuchsection", "info": f"There is no section {i}."}})
                return json.dumps({"parse": {"title": title, "text": {"*": body}}}, ensure_ascii=False, indent=1)
            return json.dumps(
                {"parse": {"title": title, "text": {"*": self.html_text(html)}}}, ensure_ascii=False, indent=1
            )
        return NOT_FOUND

    BLOCKS = ("p", "li", "td", "th", "dd", "dt")

    def _walk(self, html):
        """Lead text and [(level, heading, text)] for h2-h4 sections, in one pass over the document."""
        soup = BeautifulSoup(html, "lxml")
        lead, secs = [], []
        for el in soup.find_all(("h2", "h3", "h4") + self.BLOCKS):
            if el.name in ("h2", "h3", "h4"):
                secs.append((int(el.name[1]), el.get_text(" ").strip(), []))
                continue
            if el.find_parent(self.BLOCKS):
                continue  # nested block: counted with its parent
            t = re.sub(r"\s+", " ", el.get_text(" ")).strip()
            if t:
                (secs[-1][2] if secs else lead).append(t)
        return "\n".join(lead), [(lv, h, "\n".join(body)) for lv, h, body in secs]

    def fetch(self, lang, url):
        u = urlparse(url)
        host, path, query = (u.hostname or "").lower(), u.path, u.query
        seg = path.split("/")
        if path.startswith("/wiki/"):
            title = unquote(path[len("/wiki/") :])
            if title.replace(" ", "_").lower() == "special:search":
                q = _q(query, "search")
                return (
                    self._search_page(lang, q, "fulltext" in _qs(query), int(_q(query, "limit", "20") or 20), host)
                    if q
                    else NOT_FOUND
                )
            return self._article(lang, title)
        if lang == "zh" and len(seg) > 2 and seg[1] in ZH_VARIANTS:
            return self._article(lang, "/".join(seg[2:]))
        if path in ("/w/index.php", "/index.php"):
            if _q(query, "search"):
                return self._search_page(
                    lang, _q(query, "search"), "fulltext" in _qs(query), int(_q(query, "limit", "20") or 20), host
                )
            if (
                _q(query, "title")
                and not _q(query, "oldid")
                and not _q(query, "curid")
                and _q(query, "action", "view") in ("view", "render")
            ):
                return self._article(lang, _q(query, "title"))
            return NOT_FOUND
        if path == "/w/api.php":
            return self._api(lang, query)
        if path.startswith("/w/rest.php/v1/search/"):
            q, n = _q(query, "q"), int(_q(query, "limit", "10") or 10)
            if path.endswith("/search/title"):
                return json.dumps(
                    {"pages": [{"key": t.replace(" ", "_"), "title": t} for t in self._suggest(lang, q, min(n, 100))]},
                    ensure_ascii=False,
                    indent=1,
                )
            if path.endswith("/search/page"):
                _, hits = self._search(lang, q, min(n, 100))
                return json.dumps(
                    {"pages": [{"key": t.replace(" ", "_"), "title": t, "excerpt": s} for t, s in hits]},
                    ensure_ascii=False,
                    indent=1,
                )
            return NOT_FOUND
        m = re.match(r"^/api/rest_v1/page/(html|mobile-html|summary)/([^/?]+)", path)
        if m:
            page = self._page(lang, m.group(2))
            if page is None:
                return NOT_FOUND
            if m.group(1) == "summary":
                return json.dumps({"title": page[0], "extract": self._lead(page[1])}, ensure_ascii=False, indent=1)
            return self.html_text(page[1])
        return NOT_FOUND
