"""seqqa's tools: generic sequence primitives, Biomni's molecular-biology and database functions, and the fixed
local references.

No tool accepts a question, answer, file path, or executable code. Biomni's functions are kept verbatim except
that their NCBI and HTTP requests go through the bounded clients defined below.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import pickle
import re
import threading
from collections import Counter
from functools import lru_cache
from typing import Any
from urllib.parse import urljoin, urlsplit

import jsonschema
import requests
from Bio import Entrez, Restriction, SeqIO
from Bio.Data import CodonTable, IUPACData
from Bio.Seq import Seq
from Bio.SeqUtils import MeltingTemp as mt
from bs4 import BeautifulSoup

from ..common.online import NCBI_HOST, NCBI_KEY_INTERVAL, _limit
from ..common.online import ncbi_keys as _keys
from ..common.paths import DATA

# ------------------------------------------------------ bounded REST transport
# Bounded transport for the pinned official database functions.

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


# --------------------------------------------------- Biomni database functions
# A verbatim subset of Biomni's database.py: the REST query functions seqqa's tools call, with the
# natural-language path disabled.


def _query_llm_for_api(*args, **kwargs):
    raise RuntimeError("Natural-language delegation is disabled")


def _format_query_results(result, options=None):
    """A general-purpose formatter for query function results to reduce output size.

    Parameters
    ----------
    result (dict): The original API response dictionary
    options (dict, optional): Formatting options including:
        - max_items (int): Maximum number of items to include in lists (default: 5)
        - max_depth (int): Maximum depth to traverse in nested dictionaries (default: 2)
        - include_keys (list): Only include these top-level keys (overrides exclude_keys)
        - exclude_keys (list): Exclude these keys from the output
        - summarize_lists (bool): Whether to summarize long lists (default: True)
        - truncate_strings (int): Maximum length for string values (default: 100)

    Returns
    -------
    dict: A condensed version of the input results

    """

    def _format_value(value, depth, options):
        """Recursively format a value based on its type and formatting options.

        Parameters
        ----------
        value: The value to format
        depth (int): Current recursion depth
        options (dict): Formatting options

        Returns
        -------
        Formatted value

        """
        # Base case: reached max depth
        if depth >= options["max_depth"] and (isinstance(value, dict | list)):
            if isinstance(value, dict):
                return {
                    "_summary": f"Nested dictionary with {len(value)} keys",
                    "_keys": list(value.keys())[: options["max_items"]],
                }
            else:  # list
                return _summarize_list(value, options)

        # Process based on type
        if isinstance(value, dict):
            return _format_dict(value, depth, options)
        elif isinstance(value, list):
            return _format_list(value, depth, options)
        elif isinstance(value, str) and len(value) > options["truncate_strings"]:
            return value[: options["truncate_strings"]] + "... (truncated)"
        else:
            return value

    def _format_dict(d, depth, options):
        """Format a dictionary according to options."""
        result = {}

        # Filter keys based on include/exclude options
        keys_to_process = d.keys()
        if depth == 0 and options["include_keys"]:  # Only apply at top level
            keys_to_process = [k for k in keys_to_process if k in options["include_keys"]]
        elif depth == 0 and options["exclude_keys"]:  # Only apply at top level
            keys_to_process = [k for k in keys_to_process if k not in options["exclude_keys"]]

        # Process each key
        for key in keys_to_process:
            result[key] = _format_value(d[key], depth + 1, options)

        return result

    def _format_list(lst, depth, options):
        """Format a list according to options."""
        if options["summarize_lists"] and len(lst) > options["max_items"]:
            return _summarize_list(lst, options)

        result = []
        for i, item in enumerate(lst):
            if i >= options["max_items"]:
                remaining = len(lst) - options["max_items"]
                result.append(f"... {remaining} more items (omitted)")
                break
            result.append(_format_value(item, depth + 1, options))

        return result

    def _summarize_list(lst, options):
        """Create a summary for a list."""
        if not lst:
            return []

        # Sample a few items
        sample = lst[: min(3, len(lst))]
        sample_formatted = [_format_value(item, options["max_depth"], options) for item in sample]

        # For homogeneous lists, provide type info
        if len(lst) > 0:
            item_type = type(lst[0]).__name__
            homogeneous = all(isinstance(item, type(lst[0])) for item in lst)
            type_info = f"all {item_type}" if homogeneous else "mixed types"
        else:
            type_info = "empty"

        return {
            "_summary": f"List with {len(lst)} items ({type_info})",
            "_sample": sample_formatted,
        }

    if options is None:
        options = {}

    # Default options
    default_options = {
        "max_items": 5,
        "max_depth": 20,
        "include_keys": None,
        "exclude_keys": ["raw_response", "debug_info", "request_details"],
        "summarize_lists": True,
        "truncate_strings": 100,
    }

    # Merge provided options with defaults
    for key, value in default_options.items():
        if key not in options:
            options[key] = value

    # Filter and format the result
    formatted = _format_value(result, 0, options)
    return formatted


def query_uniprot(
    prompt=None,
    endpoint=None,
    max_results=5,
):
    """Query the UniProt REST API using either natural language or a direct endpoint.

    Parameters
    ----------
    prompt (str, required): Natural language query about proteins (e.g., "Find information about human insulin")
    endpoint (str, optional): Full or partial UniProt API endpoint URL to query directly
                            (e.g., "https://rest.uniprot.org/uniprotkb/P01308")
    max_results (int): Maximum number of results to return

    Returns
    -------
    dict: Dictionary containing the query information and the UniProt API results

    Examples
    --------
    - Natural language: query_uniprot(prompt="Find information about human insulin protein")
    - Direct endpoint: query_uniprot(endpoint="https://rest.uniprot.org/uniprotkb/P01308")

    """
    # Base URL for UniProt API
    base_url = "https://rest.uniprot.org"

    # Ensure we have either a prompt or an endpoint
    if prompt is None and endpoint is None:
        return {"error": "Either a prompt or an endpoint must be provided"}

    # If using prompt, parse with Claude
    if prompt:
        # Load UniProt schema
        schema_path = os.path.join(os.path.dirname(__file__), "schema_db", "uniprot.pkl")
        with open(schema_path, "rb") as f:
            uniprot_schema = pickle.load(f)

        # Create system prompt template
        system_template = """
        You are a protein biology expert specialized in using the UniProt REST API.

        Based on the user's natural language request, determine the appropriate UniProt REST API endpoint and parameters.

        UNIPROT REST API SCHEMA:
        {schema}

        Your response should be a JSON object with the following fields:
        1. "full_url": The complete URL to query (including base URL, dataset, endpoint type, and parameters)
        2. "description": A brief description of what the query is doing

        SPECIAL NOTES:
        - Base URL is "https://rest.uniprot.org"
        - Search in reviewed (Swiss-Prot) entries first before using non-reviewed (TrEMBL) entries
        - Assume organism is human unless otherwise specified. Human taxonomy ID is 9606
        - Use gene_exact: for exact gene name searches
        - Use specific query fields like accession:, gene:, organism_id: in search queries
        - Use quotes for terms with spaces: organism_name:"Homo sapiens"

        Return ONLY the JSON object with no additional text.
        """

        # Query Claude to generate the API call
        llm_result = _query_llm_for_api(
            prompt=prompt,
            schema=uniprot_schema,
            system_template=system_template,
        )

        if not llm_result["success"]:
            return llm_result

        # Get the full URL from Claude's response
        query_info = llm_result["data"]
        endpoint = query_info.get("full_url", "")
        description = query_info.get("description", "")

        if not endpoint:
            return {
                "error": "Failed to generate a valid endpoint from the prompt",
                "llm_response": llm_result.get("raw_response", "No response"),
            }
    else:
        # Use provided endpoint directly
        if endpoint.startswith("/"):
            endpoint = f"{base_url}{endpoint}"
        elif not endpoint.startswith("http"):
            endpoint = f"{base_url}/{endpoint.lstrip('/')}"
        description = "Direct query to provided endpoint"

    # Use the common REST API helper function
    api_result = query_rest_api(endpoint=endpoint, method="GET", description=description)

    return api_result


def query_interpro(
    prompt=None,
    endpoint=None,
    max_results=3,
):
    """Query the InterPro REST API using natural language or a direct endpoint.

    Parameters
    ----------
    prompt (str, required): Natural language query about protein domains or families
    endpoint (str, optional): Direct endpoint path or full URL (e.g., "/entry/interpro/IPR023411"
                             or "https://www.ebi.ac.uk/interpro/api/entry/interpro/IPR023411")
    max_results (int): Maximum number of results to return per page

    Returns
    -------
    dict: Dictionary containing both the query information and the InterPro API results

    Examples
    --------
    - Natural language: query_interpro("Find information about kinase domains in InterPro")
    - Direct endpoint: query_interpro(endpoint="/entry/interpro/IPR023411")

    """
    # Base URL for InterPro API
    base_url = "https://www.ebi.ac.uk/interpro/api"

    # Default parameters
    format = "json"

    # Ensure we have either a prompt or an endpoint
    if prompt is None and endpoint is None:
        return {"error": "Either a prompt or an endpoint must be provided"}

    # If using prompt, parse with Claude
    if prompt:
        # Load InterPro schema
        schema_path = os.path.join(os.path.dirname(__file__), "schema_db", "interpro.pkl")
        with open(schema_path, "rb") as f:
            interpro_schema = pickle.load(f)

        # Create system prompt template
        system_template = """
        You are a protein domain expert specialized in using the InterPro REST API.

        Based on the user's natural language request, determine the appropriate InterPro REST API endpoint.

        INTERPRO REST API SCHEMA:
        {schema}

        Your response should be a JSON object with the following fields:
        1. "full_url": The complete URL to query (including the base URL "https://www.ebi.ac.uk/interpro/api")
        2. "description": A brief description of what the query is doing

        SPECIAL NOTES:
        - Path components for data types: entry, protein, structure, set, taxonomy, proteome
        - Common sources: interpro, pfam, cdd, uniprot, pdb
        - Protein subtypes can be "reviewed" or "unreviewed"
        - For specific entries, use lowercase accessions (e.g., "ipr000001" instead of "IPR000001")
        - Endpoints can be hierarchical like "/entry/interpro/protein/uniprot/P04637"

        Return ONLY the JSON object with no additional text.
        """

        # Query Claude to generate the API call
        llm_result = _query_llm_for_api(
            prompt=prompt,
            schema=interpro_schema,
            system_template=system_template,
        )

        if not llm_result["success"]:
            return llm_result

        # Extract the full URL from Claude's response
        query_info = llm_result["data"]
        endpoint = query_info.get("full_url", "")
        description = query_info.get("description", "")

        if not endpoint:
            return {
                "error": "Failed to generate a valid endpoint from the prompt",
                "llm_response": llm_result.get("raw_response", "No response"),
            }
    else:
        # Process provided endpoint
        # If it's just a path, add the base URL
        if endpoint.startswith("/"):
            endpoint = f"{base_url}{endpoint}"
        elif not endpoint.startswith("http"):
            endpoint = f"{base_url}/{endpoint.lstrip('/')}"

        description = "Direct query to provided endpoint"

    # Add pagination parameters
    params = {"page": 1, "page_size": max_results}

    # Add format parameter if not json
    if format and format != "json":
        params["format"] = format

    # Make the API request
    api_result = query_rest_api(endpoint=endpoint, method="GET", params=params, description=description)

    return api_result


def query_ensembl(
    prompt=None,
    endpoint=None,
    verbose=True,
):
    """Query the Ensembl REST API using natural language or a direct endpoint.

    Parameters
    ----------
    prompt (str, required): Natural language query about genomic data
    endpoint (str, optional): Direct API endpoint to query (e.g., "lookup/symbol/human/BRCA2") or full URL
    verbose (bool): Whether to return detailed results

    Returns
    -------
    dict: Dictionary containing the query results or error information

    Examples
    --------
    - Natural language: query_ensembl("Get information about the human BRCA2 gene")
    - Direct endpoint: query_ensembl(endpoint="lookup/symbol/homo_sapiens/BRCA2")

    """
    # Base URL for Ensembl API
    base_url = "https://rest.ensembl.org"

    # Ensure we have either a prompt or an endpoint
    if not prompt and not endpoint:
        return {"error": "Either a prompt or an endpoint must be provided"}

    # If using prompt, parse with Claude
    if prompt:
        # Load Ensembl schema
        schema_path = os.path.join(os.path.dirname(__file__), "schema_db", "ensembl.pkl")
        with open(schema_path, "rb") as f:
            ensembl_schema = pickle.load(f)

        # Create system prompt template
        system_template = """
        You are a genomics and bioinformatics expert specialized in using the Ensembl REST API.

        Based on the user's natural language request, determine the appropriate Ensembl REST API endpoint and parameters.

        ENSEMBL REST API SCHEMA:
        {schema}

        Your response should be a JSON object with the following fields:
        1. "endpoint": The API endpoint to query (e.g., "lookup/symbol/homo_sapiens/BRCA2")
        2. "params": An object containing query parameters specific to the endpoint
        3. "description": A brief description of what the query is doing

        SPECIAL NOTES:
        - Chromosome region queries have a maximum length of 4900000 bp inclusive, so bp of start and end should be 4900000 bp apart. If the user's query exceeds this limit, Ensembl will return an error.
        - For symbol lookups, the format is "lookup/symbol/[species]/[symbol]"
        - To find the coordinates of a band on a chromosome, use /info/assembly/homo_sapiens/[chromosome] with parameters "band":1
        - To find the overlapping genes of a genomic region, use /overlap/region/homo_sapiens/[chromosome]:[start]-[end]
        - For sequence queries, specify the sequence type in parameters (genomic, cdna, cds, protein)
        - For converting rsID to hg38 genomic coordinates, use the "GET id/variation/[species]/[rsid]" endpoint
        - Many endpoints support "content-type" parameter for format specification (application/json, text/xml)

        Return ONLY the JSON object with no additional text.
        """

        # Query Claude to generate the API call
        llm_result = _query_llm_for_api(
            prompt=prompt,
            schema=ensembl_schema,
            system_template=system_template,
        )

        if not llm_result["success"]:
            return llm_result

        # Get the endpoint and parameters from Claude's response
        query_info = llm_result["data"]
        endpoint = query_info.get("endpoint", "")
        params = query_info.get("params", {})
        description = query_info.get("description", "")

        if not endpoint:
            return {
                "error": "Failed to generate a valid endpoint from the prompt",
                "llm_response": llm_result.get("raw_response", "No response"),
            }
    else:
        # Process provided endpoint
        if endpoint.startswith("http"):
            # If a full URL is provided, extract the endpoint part
            if endpoint.startswith(base_url):
                endpoint = endpoint[len(base_url) :].lstrip("/")

        params = {}
        description = "Direct query to Ensembl API"

    # Remove leading slash if present
    if endpoint.startswith("/"):
        endpoint = endpoint[1:]

    # Prepare headers for JSON response
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    # Construct the URL
    url = f"{base_url}/{endpoint}"

    # Execute the Ensembl API request using the helper function
    api_result = query_rest_api(
        endpoint=url,
        method="GET",
        params=params,
        headers=headers,
        description=description,
    )

    # Format the results if successful
    if not verbose and "success" in api_result and api_result["success"] and "result" in api_result:
        return _format_query_results(api_result["result"])

    return api_result


# ------------------------------------------ Biomni molecular-biology functions


def _ncbi_gene_coding_sequence(gene_name: str, organism: str, email: str = None) -> list[dict[str, str]]:
    """Retrieves the coding sequence(s) of a specified gene from NCBI _ENTREZ.

    Args:
        gene_name: Name of the gene
        organism: Name of the organism
        email: Email address for NCBI Entrez (recommended)

    Returns:
        List of dictionaries containing:
            - refseq_id: RefSeq ID of the gene
            - sequence: Coding sequence of the gene

    """
    if email:
        _ENTREZ.email = email

    def search_gene() -> str:
        """Search for gene ID in NCBI database."""
        query = f"{organism}[Organism] AND {gene_name}[Gene]"
        with _ENTREZ.esearch(db="gene", term=query, retmax=5) as handle:
            record = _ENTREZ.read(handle)

        if not record["IdList"]:
            raise ValueError(f"No records found for gene '{gene_name}' in organism '{organism}'")

        return record["IdList"][0]

    def get_refseq_id(gene_id: str) -> str:
        """Get RefSeq ID from gene record."""
        with _ENTREZ.efetch(db="gene", id=gene_id, rettype="gb", retmode="xml") as handle:
            gene_record = _ENTREZ.read(handle)

        try:
            locus = gene_record[0]["Entrezgene_locus"][0]
            accession = locus["Gene-commentary_accession"]
            version = locus["Gene-commentary_version"]
            return f"{accession}.{version}"
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Unable to process gene record: {e}") from None

    def get_coding_sequences(refseq_id: str) -> list[dict[str, str]]:
        """Extract coding sequences from RefSeq record."""
        with _ENTREZ.efetch(db="nucleotide", id=refseq_id, rettype="gbwithparts", retmode="text") as handle:
            seq_record = SeqIO.read(handle, "genbank")

        sequences = []
        for feature in seq_record.features:
            if feature.type == "CDS" and feature.qualifiers.get("gene", ["N/A"])[0] == gene_name:
                cds_seq = feature.location.extract(seq_record).seq
                sequences.append({"refseq_id": refseq_id, "sequence": str(cds_seq)})

        return sequences

    # Main execution flow
    try:
        gene_id = search_gene()
        refseq_id = get_refseq_id(gene_id)
        sequences = get_coding_sequences(refseq_id)

        explanation = (
            "Output fields for each coding sequence:\n"
            "- refseq_id: RefSeq identifier for the gene sequence\n"
            "  * Format: NM_XXXXXX.X for mRNA or NC_XXXXXX.X for genomic DNA\n"
            "  * Example: NM_000546.5 for human TP53\n"
            "- sequence: The actual coding sequence of the gene\n"
            "  * Contains only exons (introns removed)\n"
            "  * Starts with ATG (start codon)\n"
            "  * Ends with a stop codon (TAA, TAG, or TGA)"
        )

        return {"explanation": explanation, "sequences": sequences}

    except Exception as e:
        # Log error if needed
        raise RuntimeError(f"Failed to retrieve coding sequences: {str(e)}") from e


def _ncbi_plasmid_sequence(identifier: str, is_addgene: bool = None) -> dict[str, Any] | None:
    """Unified function to retrieve plasmid sequences from either Addgene or NCBI.
    If is_addgene is True or identifier is numeric, uses Addgene.
    Otherwise searches NCBI using the plasmid name.

    Args:
        identifier (str): Either an Addgene ID or plasmid name
        is_addgene (bool, optional): Force Addgene lookup if True, force NCBI if False.
            If None, attempts to auto-detect based on identifier format.

    Returns:
        Optional[Dict[str, Any]]: The plasmid sequence and metadata if found, None otherwise

    """

    def _get_sequence_from_addgene(plasmid_id: str) -> dict[str, Any] | None:
        """Helper function to get sequence from Addgene."""
        ADDGENE_BASE_URL = "https://www.addgene.org"
        ADDGENE_PLASMID_SEQUENCES_PATH = "/{plasmid_id}/sequences/"

        url = f"{ADDGENE_BASE_URL}{ADDGENE_PLASMID_SEQUENCES_PATH.format(plasmid_id=plasmid_id)}"
        response = _bounded_get(url)

        if response.status_code != 200:
            print(f"Failed to retrieve Addgene plasmid {plasmid_id}")
            return None

        soup = BeautifulSoup(response.content, "html.parser")
        textarea = soup.find("textarea", {"class": "copy-from"})

        if textarea:
            sequence_text = textarea.text.strip()
            lines = sequence_text.split("\n")
            if lines[0].strip() == "> Addgene NGS Result":
                sequence = "".join(lines[1:]).replace(" ", "")
                return {
                    "explanation": (
                        "Output fields:\n"
                        "- source: Source database (Addgene)\n"
                        "- identifier: Addgene plasmid ID\n"
                        "- sequence: Complete plasmid sequence"
                    ),
                    "source": "Addgene",
                    "identifier": plasmid_id,
                    "sequence": sequence,
                }

        print(f"No sequence found for Addgene plasmid {plasmid_id}")
        return None

    def _get_sequence_from_ncbi(plasmid_name: str) -> dict[str, Any] | None:
        """Helper function to get sequence from NCBI."""
        # First get the accession number
        query = f"Cloning vector {plasmid_name}"
        with _ENTREZ.esearch(db="nuccore", term=query, retmax=1, sort="relevance") as handle:
            record = _ENTREZ.read(handle)

        if not record["IdList"]:
            print(f"No results found for {plasmid_name} in NCBI.")
            return None

        plasmid_id = record["IdList"][0]

        # Get the sequence using the accession
        with _ENTREZ.efetch(db="nuccore", id=plasmid_id, rettype="fasta", retmode="text") as handle:
            try:
                sequence = SeqIO.read(handle, "fasta")
                return {
                    "explanation": (
                        "Output fields:\n"
                        "- source: Source database (NCBI)\n"
                        "- identifier: NCBI accession number\n"
                        "- sequence: Complete plasmid sequence"
                    ),
                    "source": "NCBI",
                    "identifier": plasmid_id,
                    "sequence": str(sequence.seq),
                }
            except Exception as e:
                print(f"Error retrieving sequence for {plasmid_name}: {str(e)}")
                return None

    # Auto-detect if is_addgene wasn't specified
    if is_addgene is None:
        is_addgene = identifier.isdigit()

    if is_addgene:
        return _get_sequence_from_addgene(identifier)
    else:
        return _get_sequence_from_ncbi(identifier)


def align_sequences(long_seq: str, short_seqs: str | list[str]) -> list[dict]:
    """Align short sequences (primers) to a longer sequence, allowing for one mismatch.
    Checks both forward and reverse complement strands.

    Args:
        long_seq (str): Target DNA sequence
        short_seqs (Union[str, List[str]]): Single primer or list of primers

    Returns:
        List[Dict]: List of alignment results for each short sequence, including:
            - explanation: Explanation of the output fields
            - sequence: the short sequence that was aligned
            - alignments: list of dictionaries containing:
                - position: 0-based start position in the target sequence
                - strand: '+' for forward strand, '-' for reverse complement
                - mismatches: list of tuples (position, expected_base, found_base)
                  for any mismatches

    """
    # Standardize input
    long_seq = long_seq.upper()
    short_seqs = [short_seqs.upper()] if isinstance(short_seqs, str) else [seq.upper() for seq in short_seqs]

    def reverse_complement(seq):
        """Helper function to get reverse complement of a sequence."""
        complement = {"A": "T", "T": "A", "C": "G", "G": "C"}
        return "".join(complement.get(base, base) for base in reversed(seq))

    results = []

    for short_seq in short_seqs:
        alignments = []
        seq_len = len(short_seq)

        # Check both forward and reverse complement orientations
        sequences_to_check = [
            (short_seq, "+"),  # Forward strand
            (reverse_complement(short_seq), "-"),  # Reverse complement
        ]

        for seq_to_align, strand in sequences_to_check:
            # Check each possible position in the long sequence
            for i in range(len(long_seq) - seq_len + 1):
                window = long_seq[i : i + seq_len]
                mismatches = []

                # Compare sequences base by base
                for j in range(seq_len):
                    if window[j] != seq_to_align[j]:
                        mismatches.append((j, seq_to_align[j], window[j]))

                # If we have 0 or 1 mismatches, record the alignment
                if len(mismatches) <= 1:
                    alignments.append({"position": i, "strand": strand, "mismatches": mismatches})

        results.append({"sequence": short_seq, "alignments": alignments})

    return {
        "explanation": (
            "Output fields:\n"
            "- sequences: List of alignment results, where each contains:\n"
            "  * sequence: The short sequence that was aligned\n"
            "  * alignments: List of all alignment positions found, where each contains:\n"
            "    - position: 0-based start position in the target sequence\n"
            "    - strand: '+' for forward strand, '-' for reverse complement\n"
            "    - mismatches: List of mismatches, each containing:\n"
            "      * position: Position in the short sequence where mismatch occurs\n"
            "      * expected: Base expected from short sequence\n"
            "      * found: Base found in target sequence"
        ),
        "sequences": results,
    }


def find_sequence_mutations(query_sequence, reference_sequence, query_start=1):
    """Compare query sequence against reference sequence to identify mutations.

    Args:
        query_sequence (str): The sequence being analyzed
        reference_sequence (str): The reference sequence to compare against
        query_start (int): The start position of the query sequence

    Returns:
        list: List of mutations in format RefAA_Position_QueryAA

    Example:
        If query_sequence = "ACGT" and reference_sequence = "AGGT",
        the output would be ["A1C", "G2T"] indicating mutations at positions 1 and 2.

    """
    if not all([query_sequence, reference_sequence, query_start]):
        return {
            "explanation": (
                "Output fields:\n"
                "- mutations: List of mutations found, where each mutation is formatted as:\n"
                "  * RefAA_Position_QueryAA\n"
                "  * RefAA: Original amino acid/base in reference sequence\n"
                "  * Position: Position where mutation occurs (1-based)\n"
                "  * QueryAA: New amino acid/base in query sequence\n"
                "  * Example: 'A123T' means position 123 changed from A to T\n"
                "- success: Boolean indicating if comparison was successful"
            ),
            "mutations": [],
            "success": False,
        }

    mutations = []
    for i, (query_aa, ref_aa) in enumerate(zip(query_sequence, reference_sequence, strict=False)):
        if ref_aa not in (query_aa, "-") and query_aa != "-":  # Ignore gaps
            position = query_start + i
            mutations.append(f"{ref_aa}{position}{query_aa}")

    return {
        "explanation": (
            "Output fields:\n"
            "- mutations: List of mutations found, where each mutation is formatted as:\n"
            "  * RefAA_Position_QueryAA\n"
            "  * RefAA: Original amino acid/base in reference sequence\n"
            "  * Position: Position where mutation occurs (1-based)\n"
            "  * QueryAA: New amino acid/base in query sequence\n"
            "  * Example: 'A123T' means position 123 changed from A to T\n"
            "- success: Boolean indicating if comparison was successful"
        ),
        "mutations": mutations,
        "success": True,
    }


def design_primer(
    sequence: str,
    start_pos: int,
    primer_length: int = 20,
    min_gc: float = 0.4,
    max_gc: float = 0.6,
    min_tm: float = 55.0,
    max_tm: float = 65.0,
    search_window: int = 100,
) -> dict[str, Any] | None:
    """Design a single primer within the given sequence window.

    Args:
        sequence (str): Target DNA sequence
        start_pos (int): Starting position for primer search
        primer_length (int): Length of the primer to design (default: 20)
        min_gc (float): Minimum GC content (default: 0.4)
        max_gc (float): Maximum GC content (default: 0.6)
        min_tm (float): Minimum melting temperature in °C (default: 55.0)
        max_tm (float): Maximum melting temperature in °C (default: 65.0)
        search_window (int): Size of window to search for primers (default: 100)

    Returns:
        Optional[Dict[str, Any]]: Dictionary with primer information or None if no suitable primer found

    """
    # Extract candidate region for primer design
    primer_region_start = start_pos
    primer_region_end = min(start_pos + search_window, len(sequence))
    primer_region = sequence[primer_region_start:primer_region_end]

    # Handle case where region is too small
    if len(primer_region) < primer_length:
        return None

    # Generate candidate primers
    best_primer = None
    best_score = float("inf")  # Lower is better

    for j in range(len(primer_region) - primer_length + 1):
        candidate = primer_region[j : j + primer_length]

        # Calculate GC content
        gc_content = (candidate.count("G") + candidate.count("C")) / primer_length
        # Skip if GC content is outside acceptable range
        if gc_content < min_gc or gc_content > max_gc:
            continue

        # Calculate melting temperature
        tm = mt.Tm_Wallace(candidate)

        # Skip if Tm is outside acceptable range
        if tm < min_tm or tm > max_tm:
            continue

        # Score the candidate (distance from ideal GC and Tm)
        ideal_gc = (min_gc + max_gc) / 2
        ideal_tm = (min_tm + max_tm) / 2
        gc_penalty = abs(gc_content - ideal_gc) * 100
        tm_penalty = abs(tm - ideal_tm)

        score = gc_penalty + tm_penalty

        if score < best_score:
            best_score = score
            best_primer = {
                "sequence": candidate,
                "position": primer_region_start + j,
                "gc": gc_content,
                "tm": tm,
                "score": score,
            }

    return best_primer


def design_verification_primers(
    plasmid_sequence: str,
    target_region: tuple[int, int],
    existing_primers: list[dict[str, str]] | None = None,
    is_circular: bool = True,
    coverage_length: int = 800,
    primer_length: int = 20,
    min_gc: float = 0.4,
    max_gc: float = 0.6,
    min_tm: float = 55.0,
    max_tm: float = 65.0,
) -> dict[str, Any]:
    """Design Sanger sequencing primers to verify a specific region in a plasmid.

    First tries to use primers from an existing primer pool. If they cannot fully
    cover the region, designs additional primers as needed.

    Args:
        plasmid_sequence (str): The complete plasmid sequence
        target_region (Tuple[int, int]): Start and end positions to verify (0-based indexing)
        existing_primers (Optional[List[Dict[str, str]]]): List of existing primers with
                                                         their sequences and optional names.
                                                         If None, uses common lab primers.
        is_circular (bool): Whether the plasmid is circular (default: True)
        coverage_length (int): Typical read length for each primer (default: 800bp)
        primer_length (int): Length of newly designed primers (default: 20)
        min_gc (float): Minimum GC content for new primers (default: 0.4)
        max_gc (float): Maximum GC content for new primers (default: 0.6)
        min_tm (float): Minimum melting temperature in °C (default: 55.0)
        max_tm (float): Maximum melting temperature in °C (default: 65.0)

    Returns:
        Dict: Dictionary containing:
            - target_region: The region to be verified
            - recommended_primers: List of primers to use (from existing and/or newly designed)
            - coverage_map: How the primers cover the target region

    """
    # Use default primers if none are provided
    if existing_primers is None:
        # Common laboratory primers
        existing_primers = [
            {"name": "T7", "sequence": "TAATACGACTCACTATAGGG"},
            {"name": "T7_Term", "sequence": "GCTAGTTATTGCTCAGCGG"},
            {"name": "T3", "sequence": "ATTAACCCTCACTAAAGGGA"},
            {"name": "SP6", "sequence": "GATTTAGGTGACACTATAG"},
            {"name": "U6", "sequence": "GACTATCATATGCTTACCGT"},
            {"name": "BGHR", "sequence": "TAGAAGGCACAGTCGAGG"},
            {"name": "M13F", "sequence": "GTAAAACGACGGCCAG"},
            {"name": "M13R", "sequence": "CAGGAAACAGCTATGAC"},
            {"name": "M13-40FOR", "sequence": "GTTTTCCCAGTCACGAC"},
            {"name": "M13-48REV", "sequence": "CGGATAACAATTTCACACAG"},
            {"name": "CMV-Forward", "sequence": "CGCAAATGGGCGGTAGGCGTG"},
            {"name": "5GEX", "sequence": "GGGCTGGCAAGCCACGTTTGGTG"},
            {"name": "3GEX", "sequence": "CCGGGAGCTGCATGTGTCAGAGG"},
            {"name": "pFastBacF", "sequence": "GGATTATTCATACCGTCCCA"},
            {"name": "pFastBacR", "sequence": "CAAATGTGGTATGGCTGATT"},
            {"name": "pBAD_Forward", "sequence": "ATGCCATAGCATTTTTATCC"},
            {"name": "pBAD_Reverse", "sequence": "GATTTAATCTGTATCAGG"},
            {"name": "EGFP-C-For", "sequence": "CATGGTCCTGCTGGAGTTCGTG"},
            {"name": "EGFP-C-REV", "sequence": "GTTCAGGGGGAGGTGTG"},
            {"name": "EGFP-N", "sequence": "CGTCGCCGTCCAGCTCGACCA"},
            {"name": "SV40pA-R", "sequence": "GAAATTTGTGATGCTATTGC"},
            {"name": "ATTB1", "sequence": "GTTTGTACAAAAAAGCAGGC"},
            {"name": "ATTB2", "sequence": "CCACTTTGTACAAGAAAGCTGGGT"},
            {"name": "ATTL1", "sequence": "CGCGTTAACGCTAGCATGGATCTC"},
            {"name": "ATTL2", "sequence": "CATCAGAGATTTTGAGACAC"},
            {"name": "ITS1", "sequence": "TCCGTAGGTGAACCTGCGG"},
            {"name": "ITS4", "sequence": "TCCTCCGCTTATTGATATGC"},
            {"name": "PJET1-2F", "sequence": "CGACTCACTATAGGGAGAGCGGC"},
            {"name": "PJET1-2R", "sequence": "AAGAACATCGATTTTCCATGGCAG"},
        ]

    # Helper functions for region coverage calculations
    def is_position_covered(pos, covered_regions):
        """Check if a position is covered by any region."""
        return any(region["start"] <= pos <= region["end"] for region in covered_regions)

    def is_region_fully_covered(covered_regions, start, end):
        """Check if the target region is fully covered."""
        # Merge overlapping regions first
        merged = merge_overlapping_regions(covered_regions)

        # Check if merged regions cover the entire target
        return all(any(r["start"] <= pos <= r["end"] for r in merged) for pos in range(start, end + 1))

    def merge_overlapping_regions(regions):
        """Merge overlapping regions in the list."""
        if not regions:
            return []

        # Sort by start position
        sorted_regions = sorted(regions, key=lambda x: x["start"])

        # Merge overlapping
        merged = [sorted_regions[0]]
        for region in sorted_regions[1:]:
            prev = merged[-1]
            # If regions overlap or are adjacent
            if region["start"] <= prev["end"] + 1:
                # Extend previous region
                prev["end"] = max(prev["end"], region["end"])
            else:
                # Add as new region
                merged.append(region)

        return merged

    # Clean and prepare the plasmid sequence
    plasmid_sequence = plasmid_sequence.upper()

    # Handle target region
    start, end = target_region
    region_length = end - start + 1
    if region_length <= 0:
        raise ValueError("Target region end must be greater than start")

    # Prepare result containers
    recommended_primers = []
    coverage_map = []

    # If plasmid is circular, we may need to handle wrapping
    if is_circular and end >= len(plasmid_sequence):
        # Adjust for circular plasmid wrapping
        effective_sequence = plasmid_sequence + plasmid_sequence[: end - len(plasmid_sequence) + coverage_length]
    else:
        effective_sequence = plasmid_sequence

    # Step 1: Try to use existing primers if provided
    used_existing_primers = False

    if existing_primers:
        # Create a structured representation of existing primers
        primer_pool = []
        for i, primer in enumerate(existing_primers):
            if isinstance(primer, str):
                primer_info = {"name": f"Existing_{i + 1}", "sequence": primer}
            else:
                primer_info = primer.copy()
                if "name" not in primer_info:
                    primer_info["name"] = f"Existing_{i + 1}"

            primer_pool.append(primer_info)

        # Find alignments for all existing primers
        alignment_results = align_sequences(effective_sequence, [p["sequence"] for p in primer_pool])

        # Create list of all potential primer matches with their coverage
        potential_primers = []

        for i, result in enumerate(alignment_results.get("sequences", [])):
            primer_info = primer_pool[i]
            primer_seq = primer_info["sequence"]

            for alignment in result.get("alignments", []):
                position = alignment["position"]
                strand = alignment["strand"]

                # Calculate coverage range based on primer position and orientation
                if strand == "+":
                    coverage_start = position
                    coverage_end = position + coverage_length
                else:  # Reverse primer
                    coverage_end = position + len(primer_seq)
                    coverage_start = coverage_end - coverage_length

                # Check if this primer covers any part of the target region
                if coverage_start <= end and coverage_end >= start:
                    # Calculate the coverage within the target region
                    covered_start = max(start, coverage_start)
                    covered_end = min(end, coverage_end)

                    # Add to potential primers list
                    potential_primers.append(
                        {
                            "name": primer_info["name"],
                            "sequence": primer_seq,
                            "position": position,
                            "strand": strand,
                            "source": "existing",
                            "covers": [covered_start, covered_end],
                            "coverage_length": covered_end - covered_start + 1,
                        }
                    )

        # Sort potential primers by coverage length (descending)
        potential_primers.sort(key=lambda x: x["coverage_length"], reverse=True)

        # Select primers using a greedy approach to minimize the number used
        # while achieving maximum coverage
        if potential_primers:
            # Initialize covered regions
            covered_regions = []
            selected_primers = []

            # Keep selecting primers until we've covered everything or run out of primers
            while potential_primers and not is_region_fully_covered(covered_regions, start, end):
                # Take the first remaining primer (with most coverage)
                best_primer = potential_primers.pop(0)

                # Check if this primer adds any new coverage
                new_covered_start, new_covered_end = best_primer["covers"]
                adds_new_coverage = False

                for s in range(new_covered_start, new_covered_end + 1):
                    if not is_position_covered(s, covered_regions):
                        adds_new_coverage = True
                        break

                if adds_new_coverage:
                    # Add primer to selected list
                    selected_primers.append(best_primer)

                    # Update covered regions
                    covered_regions.append({"start": new_covered_start, "end": new_covered_end})

                    # Merge overlapping regions
                    covered_regions = merge_overlapping_regions(covered_regions)

            # Update recommended primers and coverage map
            for primer in selected_primers:
                recommended_primers.append(primer)
                coverage_map.append(
                    {
                        "primer": primer["name"],
                        "start": primer["covers"][0],
                        "end": primer["covers"][1],
                        "length": primer["covers"][1] - primer["covers"][0] + 1,
                    }
                )

                used_existing_primers = True

    # Step 2: Identify uncovered regions for new primer design
    if not used_existing_primers:
        # No existing primers used, the entire region is uncovered
        uncovered_regions = [{"start": start, "end": end}]
    else:
        # Sort and merge coverage map
        covered_regions = [{"start": cm["start"], "end": cm["end"]} for cm in coverage_map]
        covered_regions = merge_overlapping_regions(covered_regions)

        # Find gaps
        uncovered_regions = []
        current_pos = start

        for region in covered_regions:
            if current_pos < region["start"]:
                uncovered_regions.append({"start": current_pos, "end": region["start"] - 1})
            current_pos = max(current_pos, region["end"] + 1)

        if current_pos <= end:
            uncovered_regions.append({"start": current_pos, "end": end})

    # Step 3: Design new primers for uncovered regions
    for region in uncovered_regions:
        gap_start = region["start"]
        gap_end = region["end"]
        gap_length = gap_end - gap_start + 1

        # Determine how many primers we need for this gap
        num_primers_needed = max(
            1,
            (gap_length // (coverage_length // 2)) + (1 if gap_length % (coverage_length // 2) > 0 else 0),
        )

        # Calculate positions for even distribution of primers
        positions = []
        if num_primers_needed == 1:
            positions.append(max(0, gap_start - 100))  # Start primer a bit before the gap
        else:
            interval = gap_length / (num_primers_needed - 0.5)  # Distribute primers evenly
            for i in range(num_primers_needed):
                pos = max(0, int(gap_start - 100 + i * interval))
                positions.append(min(pos, len(effective_sequence) - primer_length))

        # Design primers for each position
        for pos in positions:
            # Design a new primer
            new_primer = design_primer(
                effective_sequence,
                pos,
                primer_length=primer_length,
                min_gc=min_gc,
                max_gc=max_gc,
                min_tm=min_tm,
                max_tm=max_tm,
            )

            # If we found a suitable primer
            if new_primer:
                primer_name = f"New_primer_{len(recommended_primers) + 1}"

                # Calculate coverage
                coverage_start = new_primer["position"]
                coverage_end = coverage_start + coverage_length

                # Adjust to target region boundaries
                covered_start = max(start, coverage_start)
                covered_end = min(end, coverage_end)

                # Add to recommended primers
                recommended_primers.append(
                    {
                        "name": primer_name,
                        "sequence": new_primer["sequence"],
                        "position": new_primer["position"],
                        "strand": "+",  # Forward strand primer
                        "source": "newly_designed",
                        "gc": new_primer["gc"],
                        "tm": new_primer["tm"],
                        "covers": [covered_start, covered_end],
                    }
                )

                # Add to coverage map
                coverage_map.append(
                    {
                        "primer": primer_name,
                        "start": covered_start,
                        "end": covered_end,
                        "length": covered_end - covered_start + 1,
                    }
                )

    # Recalculate coverage to account for new primers
    coverage_map.sort(key=lambda x: x["start"])

    # Check if the target region is fully covered
    covered_regions = [{"start": cm["start"], "end": cm["end"]} for cm in coverage_map]
    is_fully_covered = is_region_fully_covered(covered_regions, start, end)

    # Prepare the final result
    result = {
        "target_region": {"start": start, "end": end, "length": region_length},
        "recommended_primers": recommended_primers,
        "coverage_map": coverage_map,
        "is_fully_covered": is_fully_covered,
    }

    if not is_fully_covered:
        result["warning"] = "The target region may not be fully covered. Consider manually reviewing the coverage map."

    return result


# --------------------------------------------------------- sequence primitives
# Fixed sequence operations. No code, expressions, files or task-aware solvers.

MAX_SEQUENCE = 100_000
MAX_SEARCH_WORK = 20_000_000
IUPAC = IUPACData.ambiguous_dna_values
ENZYMES = {str(enzyme): enzyme for enzyme in Restriction.AllEnzymes}


def sequence_counts(sequence):
    """A symbol histogram, without derived ratios or combined base counts."""
    sequence = sequence.upper()
    return {"length": len(sequence), "counts": dict(sorted(Counter(sequence).items()))}


def sequence_reverse_complement(sequence):
    return {"sequence": str(Seq(sequence.upper()).reverse_complement()), "length": len(sequence)}


def sequence_slice(sequence, start, end, circular=False):
    size = len(sequence)
    if circular:
        if not 0 <= start < size or not start <= end <= start + size:
            raise ValueError("Circular slice requires 0 <= start < length and start <= end <= start+length")
    elif not 0 <= start <= end <= size:
        raise ValueError("Linear slice requires 0 <= start <= end <= length")
    result = (sequence * 2 if circular else sequence)[start:end].upper()
    return {"sequence": result, "length": len(result), "start": start, "end": end}


def _regex(pattern):
    return "".join("[" + IUPAC[base] + "]" for base in pattern)


def _starts(target, pattern, max_mismatches):
    """Overlapping matches, including a C-level candidate filter for one mismatch."""
    if not max_mismatches:
        yield from ((match.start(), []) for match in re.finditer("(?=" + _regex(pattern) + ")", target))
        return
    # With <=1 mismatch, at least one half of a pattern of length >=2 is exact.
    # This avoids a Python O(target_length * pattern_length) scan of every window.
    width = len(pattern)
    if width == 1:
        candidates = range(len(target))
    else:
        middle = width // 2
        candidates = set()
        for offset, half in ((0, pattern[:middle]), (middle, pattern[middle:])):
            for match in re.finditer("(?=" + _regex(half) + ")", target):
                start = match.start() - offset
                if 0 <= start <= len(target) - width:
                    candidates.add(start)
        candidates = sorted(candidates)
    for start in candidates:
        mismatches = []
        for offset, symbol in enumerate(pattern):
            if target[start + offset] not in IUPAC[symbol]:
                mismatches.append(offset)
                if len(mismatches) > max_mismatches:
                    break
        if len(mismatches) <= max_mismatches:
            yield start, mismatches


def sequence_find(sequence, patterns, max_mismatches=0, circular=False, offset=0, limit=200):
    """Search the supplied strand only; do not infer frames, cuts or primer pairs."""
    sequence = sequence.upper()
    patterns = [pattern.upper() for pattern in patterns]
    size = len(sequence)
    if size * sum(map(len, patterns)) > MAX_SEARCH_WORK:
        raise ValueError("Search exceeds the work limit; use fewer patterns or a shorter sequence")
    if any(len(pattern) > size for pattern in patterns):
        raise ValueError("Each pattern must fit within one molecule")
    counts, page = [], []
    total = 0
    for index, pattern in enumerate(patterns):
        target = sequence + sequence[: len(pattern) - 1] if circular else sequence
        count = 0
        for start, mismatches in _starts(target, pattern, max_mismatches):
            if start >= size:
                continue
            if offset <= total < offset + limit:
                page.append(
                    {
                        "pattern_index": index,
                        "start": start,
                        "end": start + len(pattern),
                        "mismatches": len(mismatches),
                        "mismatch_offsets": mismatches,
                    }
                )
            count += 1
            total += 1
        counts.append(count)
    return {
        "sequence_length": size,
        "matches": page,
        "counts_by_pattern": counts,
        "total_matches": total,
        "offset": offset,
        "next_offset": offset + len(page) if offset + len(page) < total else None,
        "coordinate_system": (
            "zero-based start, exclusive end on the supplied strand; " "end may exceed length for circular matches"
        ),
        "order": "pattern_index, then start; overlapping matches included",
    }


def sequence_translate(sequence, frame=0, table_id=1):
    sequence = sequence.upper()
    if "T" in sequence and "U" in sequence:
        raise ValueError("Use DNA or RNA, not mixed T and U")
    if table_id not in CodonTable.ambiguous_dna_by_id:
        raise ValueError("Unknown genetic-code table_id")
    if frame > len(sequence):
        raise ValueError("Frame starts beyond the sequence")
    count = (len(sequence) - frame) // 3
    end = frame + count * 3
    protein = str(Seq(sequence[frame:end].replace("U", "T")).translate(table=table_id))
    return {
        "sequence": protein,
        "length": len(protein),
        "frame": frame,
        "table_id": table_id,
        "translated_start": frame,
        "translated_end": end,
        "trailing_bases": len(sequence) - end,
    }


def restriction_enzyme_properties(enzymes):
    rows = {}
    for name in enzymes:
        enzyme = ENZYMES.get(name)
        if enzyme is None:
            rows[name] = {"status": "not_found"}
            continue
        pairs = [
            {"top": top, "bottom": len(enzyme.site) + bottom}
            for top, bottom in ((enzyme.fst5, enzyme.fst3), (enzyme.scd5, enzyme.scd3))
            if top is not None and bottom is not None
        ]
        rows[name] = {
            "status": "ok" if pairs else "undefined_cuts",
            "recognition_sequence": enzyme.site,
            "cut_pairs": pairs,
            "palindromic": enzyme.is_palindromic(),
        }
    return {
        "enzymes": rows,
        "coordinate_system": (
            "Cut offsets are zero-based boundaries relative to the recognition sequence's start, "
            "as written 5' to 3'. Both top and bottom offsets use that same left-to-right axis. "
            "Offsets may fall outside the motif. Each pair is one double-strand cleavage. "
            "An empty cut_pairs list means coordinates are unknown, not that the enzyme never cuts."
        ),
    }


PRIMITIVE_FUNCTIONS = {
    function.__name__: function
    for function in (
        sequence_counts,
        sequence_reverse_complement,
        sequence_slice,
        sequence_find,
        sequence_translate,
        restriction_enzyme_properties,
    )
}


def primitive_specs(spec):
    sequence = {"type": "string", "minLength": 1, "maxLength": MAX_SEQUENCE, "pattern": "^[A-Za-z*.-]+$"}
    dna = {**sequence, "pattern": "^[ACGTRYSWKMBDHVNacgtryswkmbdhvn]+$"}
    integer = {"type": "integer", "minimum": 0, "maximum": MAX_SEQUENCE * 2}
    return [
        spec(
            "sequence_counts",
            "Return length and a case-insensitive count for each symbol present "
            "in a DNA, RNA or protein sequence. Absent symbols have count zero. Gaps and '*' are "
            "counted as symbols. Returns no percentages.",
            {"sequence": sequence},
            ["sequence"],
        ),
        spec(
            "sequence_reverse_complement",
            "Return the reverse complement of a DNA sequence or " "IUPAC DNA motif, with ambiguous symbols preserved.",
            {"sequence": dna},
            ["sequence"],
        ),
        spec(
            "sequence_slice",
            "Extract exactly the supplied [start,end) interval from a sequence. "
            "Coordinates are zero-based, end-exclusive. For linear sequences require "
            "0 <= start <= end <= length. circular=true allows 0 <= start < length and "
            "start <= end <= start+length, so end can cross the origin. No strand conversion.",
            {"sequence": sequence, "start": integer, "end": integer, "circular": {"type": "boolean"}},
            ["sequence", "start", "end"],
        ),
        spec(
            "sequence_find",
            "Search the supplied DNA strand for explicit IUPAC DNA patterns. "
            "Returns overlapping matches as zero-based [start,end), mismatch offsets and counts "
            "per pattern. No automatic reverse-complement search. Defaults: exact matching, "
            "linear molecule, offset 0, limit 200. max_mismatches can be 0 or 1; an IUPAC symbol "
            "matches any of its allowed bases. circular=true includes origin-spanning matches "
            "with end > sequence length. Patterns must fit one molecule. Results are paginated "
            "in pattern-index/start order; use next_offset. Work limit: length*sum(pattern lengths) <= 20000000.",
            {
                "sequence": {**sequence, "pattern": "^[ACGTacgt]+$"},
                "patterns": {"type": "array", "minItems": 1, "maxItems": 100, "items": {**dna, "maxLength": 1000}},
                "max_mismatches": {"type": "integer", "enum": [0, 1]},
                "circular": {"type": "boolean"},
                "offset": {"type": "integer", "minimum": 0, "maximum": 10_000_000},
                "limit": {"type": "integer", "minimum": 1, "maximum": 200},
            },
            ["sequence", "patterns"],
        ),
        spec(
            "sequence_translate",
            "Translate all complete codons from the supplied DNA/RNA "
            "sequence, beginning at frame 0, 1 or 2 (default 0), using an NCBI genetic-code "
            "table_id (default 1). Keep stop codons as '*'; omit and report trailing incomplete "
            "codon bases. Does not find starts, stop at stops, select ORFs or reverse-complement.",
            {
                "sequence": {**dna, "pattern": "^[ACGTURYSWKMBDHVNacgturyswkmbdhvn]+$"},
                "frame": {"type": "integer", "enum": [0, 1, 2]},
                "table_id": {"type": "integer", "minimum": 1, "maximum": 33},
            },
            ["sequence"],
        ),
        spec(
            "restriction_enzyme_properties",
            "Look up named restriction enzymes in the pinned "
            "catalog. Returns IUPAC recognition sequence, palindromicity and cut pairs. Each "
            "pair has top/bottom zero-based boundary offsets measured from the recognition "
            "sequence start on the same left-to-right axis. Does not accept a target sequence "
            "or search for sites. Unknown names return not_found; known enzymes without cut "
            "coordinates return undefined_cuts.",
            {
                "enzymes": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 100,
                    "items": {"type": "string", "minLength": 1, "maxLength": 100},
                }
            },
            ["enzymes"],
        ),
    ]


# ---------------------------------------------------------------- Biomni tools
# Related official tools: Biomni molecular-biology and database functions, behind the bounded transport.

ONLINE_NAMES = set(BASES)
LOCAL_NAMES = {
    "find_sequence_mutations",
    "design_primer",
    "design_verification_primers",
}
EXTRA_NAMES = ONLINE_NAMES | LOCAL_NAMES


EXTRA_FUNCTIONS = {
    "find_sequence_mutations": find_sequence_mutations,
    "design_primer": design_primer,
    "design_verification_primers": design_verification_primers,
    "query_uniprot": query_uniprot,
    "query_ensembl": query_ensembl,
    "query_interpro": query_interpro,
}


def extra_specs(spec):
    dna = {"type": "string", "pattern": "^[ACGTacgt]+$", "minLength": 1, "maxLength": 10000}
    sequence = {"type": "string", "pattern": "^[A-Za-z*.-]+$", "minLength": 1, "maxLength": 100000}
    boolean = {"type": "boolean"}
    primer_options = {
        "primer_length": {"type": "integer", "minimum": 4, "maximum": 100},
        "min_gc": {"type": "number", "minimum": 0, "maximum": 1},
        "max_gc": {"type": "number", "minimum": 0, "maximum": 1},
        "min_tm": {"type": "number", "minimum": 0, "maximum": 150},
        "max_tm": {"type": "number", "minimum": 0, "maximum": 150},
    }
    tools = [
        spec(
            "find_sequence_mutations",
            "Compare equal-length, already aligned reference and query sequences for substitutions. "
            "Returns reference-base/position/query-base strings; ignores gap columns. "
            "query_start is one-based and defaults to 1. Does not align sequences or call indels.",
            {
                "query_sequence": sequence,
                "reference_sequence": sequence,
                "query_start": {"type": "integer", "minimum": 1, "maximum": 1000000000},
            },
            ["query_sequence", "reference_sequence"],
        ),
        spec(
            "design_primer",
            "Design a forward primer within a DNA window, using GC constraints and Wallace melting "
            "temperature. start_pos is zero-based. Defaults: length 20, GC 0.4–0.6, Tm 55–65 C, "
            "search_window 100. Returns the selected primer or no match.",
            {
                "sequence": dna,
                "start_pos": {"type": "integer", "minimum": 0},
                "search_window": {"type": "integer", "minimum": 1, "maximum": 1000},
                **primer_options,
            },
            ["sequence", "start_pos"],
        ),
        spec(
            "design_verification_primers",
            "Choose existing or design new Sanger sequencing primers to cover a plasmid region. "
            "target_region is [start,end], both zero-based and inclusive; an unwrapped end may "
            "cross the origin for a circular plasmid. Defaults: circular true, coverage 800, "
            "primer length 20, GC 0.4–0.6, Tm 55–65 C; common primers used when omitted.",
            {
                "plasmid_sequence": dna,
                "target_region": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 0},
                    "minItems": 2,
                    "maxItems": 2,
                },
                "existing_primers": {
                    "type": ["array", "null"],
                    "maxItems": 30,
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "maxLength": 100},
                            "sequence": {**dna, "maxLength": 100},
                        },
                        "required": ["sequence"],
                        "additionalProperties": False,
                    },
                },
                "is_circular": boolean,
                "coverage_length": {"type": "integer", "minimum": 100, "maximum": 2000},
                **primer_options,
            },
            ["plasmid_sequence", "target_region"],
        ),
    ]
    descriptions = {
        "query_uniprot": (
            "Query UniProt protein sequences, accessions, genes and annotations " "using a direct REST endpoint."
        ),
        "query_ensembl": (
            "Query Ensembl gene/transcript identifiers, coordinates and genomic, CDS or protein "
            "sequences using a direct REST endpoint."
        ),
        "query_interpro": "Query InterPro protein families, domains and signatures using a direct REST endpoint.",
    }
    for name in sorted(ONLINE_NAMES):
        properties = {"endpoint": {"type": "string", "minLength": 1, "maxLength": 4000}}
        if name == "query_ensembl":
            properties["verbose"] = boolean
        else:
            properties["max_results"] = {"type": "integer", "minimum": 1, "maximum": 20}
        tools.append(
            spec(
                name,
                descriptions[name] + " Supply an API path or an official HTTPS URL. "
                "Returns live database evidence; use explicit query parameters for pagination.",
                properties,
                ["endpoint"],
            )
        )
    return tools


def prepare(name, arguments):
    """Validate cross-field constraints before invoking the official implementation."""
    args = dict(arguments)
    if name in ONLINE_NAMES:
        args["endpoint"] = endpoint_url(name, args["endpoint"])
    if name == "find_sequence_mutations":
        if len(args["query_sequence"]) != len(args["reference_sequence"]):
            raise ValueError("Sequences must already be aligned and have equal lengths")
        args["query_sequence"] = args["query_sequence"].upper()
        args["reference_sequence"] = args["reference_sequence"].upper()
    if name in {"design_primer", "design_verification_primers"}:
        seq_key = "sequence" if name == "design_primer" else "plasmid_sequence"
        args[seq_key] = args[seq_key].upper()
        size = len(args[seq_key])
        if args.get("primer_length", 20) > size:
            raise ValueError("primer_length exceeds the sequence length")
        if args.get("min_gc", 0.4) > args.get("max_gc", 0.6) or args.get("min_tm", 55) > args.get("max_tm", 65):
            raise ValueError("Minimum GC/Tm must not exceed maximum")
        if name == "design_primer" and args["start_pos"] >= size:
            raise ValueError("start_pos is outside the sequence")
        if name == "design_verification_primers":
            start, end = args["target_region"]
            max_end = start + size - 1 if args.get("is_circular", True) else size - 1
            if not 0 <= start < size or not start <= end <= max_end or end - start + 1 > 2000:
                raise ValueError("Target region must fit one molecule and span at most 2000 bases")
            # Bound the official pool-alignment and greedy-coverage work.
            primers = args.get("existing_primers")
            if primers is not None:
                total = sum(len(p["sequence"]) for p in primers)
                if 2 * size * total > 20_000_000:
                    raise ValueError("Primer alignment request is too large")
    return args


# --------------------------------------------------------- tool specs and dispatch

REFS = DATA / "seqqa/refs"
_KEY_LOCK = threading.Lock()
_KEY_INDEX = os.getpid()


def _redact(text):
    for key in _keys():
        text = text.replace(key, "[REDACTED]")
    return re.sub(r"(?i)(api_key[=:])[^&\s]+", r"\1[REDACTED]", text)


def _rate_limit(key):
    # The same stamp files as common/online.py, so dbqa, screen and seqqa share each key's rate budget.
    _limit(f"{NCBI_HOST}_sha256_{hashlib.sha256(key.encode()).hexdigest()[:12]}", NCBI_KEY_INTERVAL)


class _EntrezProxy:
    """Use bounded requests and a per-request key; never mutate Bio.Entrez."""

    email = "biomni-research-agent@example.org"
    read = staticmethod(Entrez.read)

    def _request(self, method, params):
        global _KEY_INDEX
        keys = _keys()
        with _KEY_LOCK:
            key = keys[_KEY_INDEX % len(keys)]
            _KEY_INDEX += 1
        _rate_limit(key)
        params = dict(params, email=self.email, tool="biomni-research-agent", api_key=key)
        with requests.get(
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/{method}.fcgi",
            params=params,
            timeout=(5, 30),
            stream=True,
        ) as response:
            response.raise_for_status()
            chunks = []
            size = 0
            for chunk in response.iter_content(8192):
                size += len(chunk)
                if size > 16_000_000:
                    raise ValueError("Reference response exceeded 16 MB")
                chunks.append(chunk)
            body = b"".join(chunks)
        if params.get("retmode") == "text":
            return io.StringIO(body.decode("utf-8"))
        return io.BytesIO(body)

    def esearch(self, **params):
        return self._request("esearch", params)

    def efetch(self, **params):
        return self._request("efetch", params)


_ENTREZ = _EntrezProxy()


@lru_cache(maxsize=2)
def _reference(filename):
    return SeqIO.read(REFS / filename, "genbank")


def _local_organism(organism):
    normalized = re.sub(r"[\s._-]+", "", organism).lower()
    return normalized in {
        "ecoli",
        "escherichiacoli",
        "ecolik12",
        "escherichiacolik12",
        "ecolik12mg1655",
        "escherichiacolik12mg1655",
        "escherichiacolistrk12substrmg1655",
        "nc0009133",
    }


def get_gene_coding_sequence(gene_name, organism, email=None):
    if not _local_organism(organism):
        return _ncbi_gene_coding_sequence(gene_name, organism, email)
    genome = _reference("EcoliK12-NC_000913.3.gbwithparts")
    primary = [
        feature
        for feature in genome.features
        if feature.type == "CDS"
        and gene_name.casefold()
        in {name.casefold() for name in feature.qualifiers.get("gene", []) + feature.qualifiers.get("locus_tag", [])}
    ]
    sequences = []
    # Exact primary identifiers take priority over potentially colliding aliases.
    for feature in primary or genome.features:
        if feature.type != "CDS":
            continue
        names = feature.qualifiers.get("gene", []) + feature.qualifiers.get("locus_tag", [])
        for synonyms in feature.qualifiers.get("gene_synonym", []):
            names.extend(part.strip() for part in synonyms.split(";"))
        if gene_name.casefold() in {name.casefold() for name in names}:
            sequences.append(
                {
                    "refseq_id": genome.id,
                    "sequence": str(feature.extract(genome.seq)),
                }
            )
    if not sequences:
        raise ValueError(f"Gene {gene_name!r} is absent from local {genome.id}")
    return {
        "explanation": "sequences contains refseq_id and strand-correct CDS sequence.",
        "sequences": sequences,
    }


def get_plasmid_sequence(identifier, is_addgene=None):
    if is_addgene is not True and identifier.strip().lower() in {
        "puc19",
        "m77789",
        "m77789.2",
    }:
        record = _reference("pUC19-M77789.2.gb")
        return {
            "explanation": "source, identifier and complete plasmid sequence.",
            "source": "NCBI",
            "identifier": record.id,
            "sequence": str(record.seq),
        }
    # Addgene is outside the fixed local snapshot; keep the official route but
    # bound its otherwise unbounded requests.get call.
    return _ncbi_plasmid_sequence(identifier, is_addgene)


class _RequestsProxy:
    @staticmethod
    def get(*args, **kwargs):
        kwargs.setdefault("timeout", (5, 30))
        return requests.get(*args, **kwargs)


_bounded_get = _RequestsProxy.get


def _normalize(value):
    if hasattr(value, "_asdict"):
        return {key: _normalize(val) for key, val in value._asdict().items()}
    if isinstance(value, dict):
        return {key: _normalize(val) for key, val in value.items()}
    if isinstance(value, (tuple, list)):
        return [_normalize(val) for val in value]
    return value


def _redact_value(value):
    if isinstance(value, str):
        return _redact(value)
    if isinstance(value, dict):
        return {key: _redact_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    return value


S = {"type": "string", "minLength": 1, "maxLength": MAX_SEQUENCE}


def _spec(name, description, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


TOOL_SPECS = primitive_specs(_spec) + [
    _spec(
        "get_gene_coding_sequence",
        "Retrieve gene CDS sequences by gene name and organism. "
        "E. coli defaults to the fixed K-12 MG1655 reference; other organisms use NCBI.",
        {"gene_name": S, "organism": S, "email": {"type": ["string", "null"]}},
        ["gene_name", "organism"],
    ),
    _spec(
        "get_plasmid_sequence",
        "Retrieve a plasmid sequence by name or accession, or Addgene ID. "
        "Uses a fixed local reference when available. is_addgene can force the source.",
        {"identifier": S, "is_addgene": {"type": ["boolean", "null"]}},
        ["identifier"],
    ),
]
FUNCTIONS = {
    **PRIMITIVE_FUNCTIONS,
    "get_gene_coding_sequence": get_gene_coding_sequence,
    "get_plasmid_sequence": get_plasmid_sequence,
    **EXTRA_FUNCTIONS,
}
TOOL_SPECS.extend(extra_specs(_spec))
TOOL_SPECS.sort(key=lambda spec: spec["function"]["name"])
VALIDATORS = {
    spec["function"]["name"]: jsonschema.Draft202012Validator(spec["function"]["parameters"]) for spec in TOOL_SPECS
}


def invoke(name, arguments):
    backend = "local"
    try:
        if name not in FUNCTIONS:
            return {"status": "unknown_tool", "error": f"Unknown tool: {name}"}
        VALIDATORS[name].validate(arguments)
        if name in EXTRA_NAMES:
            arguments = prepare(name, arguments)
        if name in ONLINE_NAMES:
            backend = "online"
        if name == "get_gene_coding_sequence" and not _local_organism(arguments["organism"]):
            backend = "online"
        if name == "get_plasmid_sequence" and (
            arguments.get("is_addgene") is True
            or arguments["identifier"].strip().lower() not in {"puc19", "m77789", "m77789.2"}
        ):
            backend = "online"
        result = _normalize(FUNCTIONS[name](**arguments))
        if name in ONLINE_NAMES:
            # verbose=False may return the official formatted result directly.
            status = result.get("status", "ok") if isinstance(result, dict) else "ok"
            result = _redact_value(result)
            if status != "ok":
                return {"status": status, "source": name, "backend_source": backend, "result": result}
        if result is None:
            return {"status": "not_found", "source": name, "backend_source": backend}
        encoded = json.dumps(result, ensure_ascii=False)
        if len(encoded.encode()) > MAX_REPLY_BYTES:
            return {
                "status": "result_too_large",
                "source": name,
                "backend_source": backend,
                "error": "Result exceeds 48 KB; narrow the request.",
            }
        return {"status": "ok", "source": name, "backend_source": backend, "result": result}
    except jsonschema.ValidationError as exc:
        # Do not echo user sequences in schema errors.
        error = f"Invalid parameter at {list(exc.path)}: {exc.validator}"
        status = "invalid_arguments"
    except (ValueError, TypeError, AttributeError) as exc:
        error, status = str(exc), "invalid_arguments"
    except (requests.RequestException, OSError) as exc:
        error, status = str(exc), "backend_error"
    except Exception as exc:
        # Official Entrez helpers wrap transport exceptions in RuntimeError.
        cause = exc.__cause__
        status = "backend_error" if isinstance(cause, (requests.RequestException, OSError)) else "tool_error"
        error = f"{type(exc).__name__}: {exc}"
    return {"status": status, "source": name, "backend_source": backend, "error": _redact(error)[:1000]}
