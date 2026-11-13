"""Bounded transport for the pinned official database functions."""
import json
from urllib.parse import urljoin, urlsplit

import requests

from ..common.online import _limit

BASES = {
    "query_uniprot": "https://rest.uniprot.org",
    "query_ensembl": "https://rest.ensembl.org",
    "query_interpro": "https://www.ebi.ac.uk/interpro/api",
}
MAX_REPLY_BYTES = 48_000


def endpoint_url(name, endpoint):
    base = BASES[name]
    if not isinstance(endpoint, str) or not 1 <= len(endpoint) <= 4000:
        raise ValueError("endpoint must be a nonempty API path or HTTPS URL")
    if "://" not in endpoint:
        if endpoint.startswith("//"):
            raise ValueError("Protocol-relative endpoints are not supported")
        endpoint = base.rstrip("/") + "/" + endpoint.lstrip("/")
    parsed, expected = urlsplit(endpoint), urlsplit(base)
    if (
        parsed.scheme != "https"
        or parsed.hostname != expected.hostname
        or parsed.username
        or parsed.password
        or parsed.port not in (None, 443)
        or parsed.fragment
    ):
        raise ValueError("Endpoint must use the tool's official HTTPS API host")
    if expected.path and not (parsed.path == expected.path or parsed.path.startswith(expected.path + "/")):
        raise ValueError("Endpoint must stay under the official API path")
    if any(part in {".", ".."} for part in parsed.path.split("/")):
        raise ValueError("Endpoint path traversal is not supported")
    return endpoint


def query_rest_api(endpoint, method="GET", params=None, headers=None, json_data=None, description=None):
    """Keep the official result envelope; attach explicit transport status."""
    source = next(
        (name for name, base in BASES.items() if urlsplit(endpoint).hostname == urlsplit(base).hostname), None
    )
    info = {"endpoint": endpoint, "method": method, "description": description}
    try:
        if source is None or method != "GET" or json_data is not None:
            raise ValueError("Only direct GET queries to the selected official APIs are supported")
        endpoint = endpoint_url(source, endpoint)
        # Redirects are HTTP navigation, not retries. Never follow a different host.
        for redirect in range(4):
            _limit(urlsplit(endpoint).hostname, 0.35)
            with requests.get(
                endpoint,
                params=params,
                headers=headers or {"Accept": "application/json"},
                timeout=(5, 30),
                stream=True,
                allow_redirects=False,
            ) as response:
                status = response.status_code
                info.update(endpoint=response.url, http_status=status)
                if status in (301, 302, 303, 307, 308):
                    if redirect == 3:
                        return {
                            "success": False,
                            "status": "backend_error",
                            "error": "Too many API redirects",
                            "query_info": info,
                        }
                    endpoint = endpoint_url(source, urljoin(response.url, response.headers.get("Location", "")))
                    params = None
                    continue
                category = (
                    "not_found"
                    if status == 404
                    else "backend_error"
                    if status >= 500 or status in (401, 403, 408, 429)
                    else "query_error"
                    if status >= 400
                    else "ok"
                )
                # Preserve transport failure classification even for a large error page.
                chunks, size = [], 0
                for chunk in response.iter_content(8192):
                    size += len(chunk)
                    if size > MAX_REPLY_BYTES:
                        return {
                            "success": False,
                            "status": category if category != "ok" else "result_too_large",
                            "error": "API response exceeds 48 KB; narrow or page the query",
                            "query_info": info,
                        }
                    chunks.append(chunk)
                body = b"".join(chunks)
                if category != "ok":
                    return {
                        "success": False,
                        "status": category,
                        "error": body.decode(errors="replace")[:1000],
                        "query_info": info,
                    }
                try:
                    result = json.loads(body)
                except (ValueError, UnicodeDecodeError):
                    result = {"raw_text": body.decode(errors="replace")}
                return {"success": True, "status": "ok", "query_info": info, "result": result}
    except (ValueError, TypeError) as exc:
        return {"success": False, "status": "invalid_arguments", "error": str(exc), "query_info": info}
    except (requests.RequestException, OSError) as exc:
        return {"success": False, "status": "backend_error", "error": str(exc), "query_info": info}
