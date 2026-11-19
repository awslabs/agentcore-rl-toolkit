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
import subprocess
import tempfile
import threading
from collections import Counter, namedtuple
from functools import lru_cache
from typing import Any
from urllib.parse import urljoin, urlsplit

import jsonschema
import pandas as pd
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


def annotate_open_reading_frames(sequence, min_length, search_reverse=False, filter_subsets=False):
    """Find all Open Reading Frames (ORFs) in a DNA sequence using Biopython.
    Searches both forward and reverse complement strands.

    Args:
        sequence (str): DNA sequence
        min_length (int): Minimum length of ORF in nucleotides
        search_reverse (bool): Whether to search the reverse complement strand (default: False unless you want to search for reverse ORFs)
        filter_nested (bool): Whether to filter out ORFs with same end but later start (default: False unless you want to remove nested ORFs)

    Returns:
        dict: Dictionary containing:
            - explanation: Explanation of the output fields
            - summary_stats: Statistical overview of found ORFs
            - orfs: List of all found ORFs, where each ORF contains:
                - sequence: Nucleotide sequence of the ORF
                - aa_sequence: Translated amino acid sequence
                - start: Start position in original sequence (0-based)
                - end: End position in original sequence
                - strand: '+' for forward strand, '-' for reverse complement
                - frame: Reading frame (1,2,3 for forward; -1,-2,-3 for reverse)

    """
    ORF = namedtuple("ORF", ["sequence", "aa_sequence", "start", "end", "strand", "frame"])

    def find_orfs_in_frame(seq_str, frame, strand):
        """Helper function to find ORFs in a specific frame."""
        orfs = []
        seq_length = len(seq_str)

        # Adjust sequence based on frame
        offset = frame - 1  # Convert 1-based frame to 0-based index
        frame_seq = seq_str[offset:]

        # Find all start codons
        start_positions = []
        stop_positions = []

        for i in range(0, len(frame_seq) - 2, 3):
            codon = frame_seq[i : i + 3]
            if len(codon) != 3:  # Skip incomplete codons
                continue

            # Record positions of start and stop codons
            if codon == "ATG":
                start_positions.append(i)
            elif codon in ["TAA", "TAG", "TGA"]:
                stop_positions.append(i)

                # Process any active ORFs that end at this stop codon
                while start_positions and start_positions[0] < i:
                    start_pos = start_positions.pop(0)
                    orf_seq = frame_seq[start_pos : i + 3]

                    # Check if ORF meets minimum length requirement
                    if len(orf_seq) >= min_length:
                        # Convert positions to original sequence coordinates
                        orig_start = start_pos + offset
                        orig_end = i + offset + 3

                        # For reverse strand, convert positions
                        if strand == "-":
                            orig_start = seq_length - orig_end
                            orig_end = seq_length - (start_pos + offset)

                        # Create Seq object for translation
                        orf_bio_seq = Seq(orf_seq)
                        aa_seq = str(orf_bio_seq.translate(to_stop=True))

                        orfs.append(
                            ORF(
                                sequence=orf_seq,
                                aa_sequence=aa_seq,
                                start=orig_start,
                                end=orig_end,
                                strand=strand,
                                frame=frame if strand == "+" else -frame,
                            )
                        )

        return orfs

    # Convert input to string and uppercase
    sequence = str(sequence).upper()
    seq_obj = Seq(sequence)

    all_orfs = []

    # Search forward strand
    for frame in range(1, 4):
        all_orfs.extend(find_orfs_in_frame(sequence, frame, "+"))

    # Search reverse complement strand
    if search_reverse:
        rev_comp = str(seq_obj.reverse_complement())
        for frame in range(1, 4):
            all_orfs.extend(find_orfs_in_frame(rev_comp, frame, "-"))

    if filter_subsets:
        # Sort ORFs by length first (longest first)
        all_orfs.sort(key=lambda x: len(x.sequence), reverse=True)

        # Filter out ORFs that are contained within other ORFs on the same strand
        filtered_orfs = []
        for _i, orf in enumerate(all_orfs):
            is_subset = False
            for larger_orf in filtered_orfs:
                if orf.strand == larger_orf.strand and orf.start >= larger_orf.start and orf.end <= larger_orf.end:
                    is_subset = True
                    break
            if not is_subset:
                filtered_orfs.append(orf)
        all_orfs = filtered_orfs

    # Sort ORFs by length (longest first)
    all_orfs.sort(key=lambda x: len(x.sequence), reverse=True)

    # Calculate summary statistics
    forward_orfs = len([orf for orf in all_orfs if orf.strand == "+"])
    reverse_orfs = len([orf for orf in all_orfs if orf.strand == "-"])
    avg_length = sum(len(orf.sequence) for orf in all_orfs) / len(all_orfs) if all_orfs else 0

    summary_stats = {
        "total_orfs": len(all_orfs),
        "forward_orfs": forward_orfs,
        "reverse_orfs": reverse_orfs,
        "avg_length": round(avg_length, 1),
    }

    explanation = (
        "Output fields:\n"
        "- summary_stats: Statistical overview of found ORFs\n"
        "  * total_orfs: Total number of ORFs found\n"
        "  * forward_orfs: Number of ORFs on forward strand\n"
        "  * reverse_orfs: Number of ORFs on reverse strand\n"
        "  * avg_length: Average length of all ORFs in base pairs\n"
        "- orfs: List of all found ORFs, where each ORF contains:\n"
        "  * sequence: Nucleotide sequence of the ORF\n"
        "  * aa_sequence: Translated amino acid sequence\n"
        "  * start: Start position in original sequence (0-based)\n"
        "  * end: End position in original sequence\n"
        "  * strand: '+' for forward strand, '-' for reverse complement\n"
        "  * frame: Reading frame (1,2,3 for forward; -1,-2,-3 for reverse)"
    )

    return {
        "explanation": explanation,
        "summary_stats": summary_stats,
        "orfs": all_orfs,
    }


def annotate_plasmid(sequence: str, is_circular: bool = True) -> dict[str, Any]:
    """Annotate a DNA sequence using pLannotate's command-line interface.

    Args:
        sequence (str): The DNA sequence to annotate
        is_circular (bool): Whether the sequence is circular (default: True)
        return_plot (bool): Whether to return the Bokeh plot object (default: False)

    Returns:
        Dict: Dictionary containing annotation results or None if annotation fails

    """
    try:
        # Create a temporary directory to store input/output files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create temporary FASTA file
            fasta_path = os.path.join(temp_dir, "input.fasta")
            with open(fasta_path, "w") as f:
                f.write(f">sequence\n{sequence}\n")

            # Prepare command
            cmd = [
                "plannotate",
                "batch",
                "-i",
                fasta_path,
                "-o",
                temp_dir,
                "-f",
                "output",
                "--csv",  # Request CSV output
            ]

            # Add linear flag if sequence is not circular
            if not is_circular:
                cmd.append("-l")

            # Add detailed flag for more comprehensive search
            cmd.append("-d")

            # Run plannotate
            subprocess.run(cmd, check=True)

            # Read results from CSV
            csv_path = os.path.join(temp_dir, "output_pLann.csv")
            results = pd.read_csv(csv_path)

            explanation = (
                "Output fields for each annotation:\n"
                "- sseqid: Subject sequence identifier (name of the feature)\n"
                "- qstart/qend: Start and end positions in query sequence\n"
                "- sstart/send: Start and end positions in subject sequence\n"
                "- sframe: Reading frame (-1 for reverse strand, 1 for forward)\n"
                "- score: Alignment score\n"
                "- evalue: Expected value for the alignment\n"
                "- qseq: Query sequence\n"
                "- length: Length of the alignment\n"
                "- slen: Length of the subject sequence\n"
                "- pident: Percentage identity\n"
                "- qlen: Length of the query sequence\n"
                "- db: Database source (e.g., snapgene)\n"
                "- Feature: Feature name\n"
                "- Description: Detailed description of the feature\n"
                "- Type: Feature type (e.g., CDS, promoter, rep_origin)\n"
                "- priority: Priority score\n"
                "- percmatch: Percentage match\n"
                "- abs percmatch: Absolute percentage match\n"
                "- pi_permatch: Per-identity match\n"
                "- wiggle: Wiggle room for alignment\n"
                "- wstart/wend: Wiggle start and end positions\n"
                "- kind: Feature kind\n"
                "- qstart_dup/qend_dup: Duplicate query positions\n"
                "- fragment: Boolean indicating if feature is fragmented"
            )

            return {
                "explanation": explanation,
                "annotations": results.to_dict("records"),
            }

    except Exception as e:
        print(f"Error during annotation: {e}")
        return None


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


def pcr_simple(sequence: str, forward_primer: str, reverse_primer: str, circular: bool = False) -> dict:
    """Simulate PCR amplification with given primers and sequence.

    Args:
        sequence (str): Either a sequence string or path to plasmid file
        forward_primer (str): Forward primer sequence (5' to 3')
        reverse_primer (str): Reverse primer sequence (5' to 3')
        circular (bool): Whether the sequence is circular (default: False)

    Returns:
        dict: Results of PCR simulation including products and primer binding details

    """
    # First check if primers are valid using existing function
    fwd_result = align_sequences(sequence, forward_primer)["sequences"][0]["alignments"]
    rev_result = align_sequences(sequence, str(Seq(reverse_primer).reverse_complement()))["sequences"][0]["alignments"]

    if not fwd_result or not rev_result:
        return {
            "explanation": (
                "Output fields:\n"
                "- success: Boolean indicating if any PCR products were found\n"
                "- message: Error message if no products found\n"
                "- products: Empty list when no products found\n"
                "- forward_binding_sites: Number of forward primer binding locations\n"
                "- reverse_binding_sites: Number of reverse primer binding locations"
            ),
            "success": False,
            "message": "One or both primers do not align to the sequence.",
            "products": [],
            "forward_binding_sites": len(fwd_result),
            "reverse_binding_sites": len(rev_result),
        }

    # Get primer positions
    fwd_positions = [align["position"] for align in fwd_result]
    rev_positions = [align["position"] for align in rev_result]
    sequence_length = len(sequence)

    # Find all possible PCR products
    products = []

    for fwd_pos in fwd_positions:
        for rev_pos in rev_positions:
            # Calculate product size and sequence
            if fwd_pos < rev_pos:
                # Standard linear product
                size = rev_pos - fwd_pos + len(reverse_primer)
                product = sequence[fwd_pos : rev_pos + len(reverse_primer)]
            elif circular:
                # Handle circular template - wrap around from end to start
                size = sequence_length - fwd_pos + rev_pos + len(reverse_primer)
                product = sequence[fwd_pos:] + sequence[: rev_pos + len(reverse_primer)]
            else:
                # Skip invalid combinations for linear sequences
                continue

            products.append(
                {
                    "size": size,
                    "forward_position": fwd_pos,
                    "reverse_position": rev_pos,
                    "sequence": product,
                    "forward_mismatches": fwd_result[fwd_positions.index(fwd_pos)]["mismatches"],
                    "reverse_mismatches": rev_result[rev_positions.index(rev_pos)]["mismatches"],
                }
            )

    if not products:
        return {
            "explanation": (
                "Output fields:\n"
                "- success: Boolean indicating if any PCR products were found\n"
                "- message: Error message if no products found\n"
                "- products: Empty list when no products found\n"
                "- forward_binding_sites: Number of forward primer binding locations\n"
                "- reverse_binding_sites: Number of reverse primer binding locations"
            ),
            "success": False,
            "message": "No valid PCR products found with these primers.",
            "products": [],
            "forward_binding_sites": len(fwd_positions),
            "reverse_binding_sites": len(rev_positions),
        }

    return {
        "explanation": (
            "Output fields:\n"
            "- success: Boolean indicating if any PCR products were found\n"
            "- products: List of all possible PCR products, where each contains:\n"
            "  * size: Length of the PCR product in base pairs\n"
            "  * sequence: The actual DNA sequence of the product\n"
            "  * forward_position: Starting position of forward primer binding\n"
            "  * reverse_position: Starting position of reverse primer binding\n"
            "  * forward_mismatches: List of mismatches in forward primer binding\n"
            "  * reverse_mismatches: List of mismatches in reverse primer binding\n"
            "- forward_binding_sites: Number of locations where forward primer can bind\n"
            "- reverse_binding_sites: Number of locations where reverse primer can bind"
        ),
        "success": True,
        "products": products,
        "forward_binding_sites": len(fwd_positions),
        "reverse_binding_sites": len(rev_positions),
    }


def digest_sequence(dna_sequence: str, enzyme_names: list[str], is_circular: bool = True) -> dict:
    """Simulates restriction enzyme digestion using Biopython's catalyze method and returns the resulting DNA fragments.

    Args:
        enzyme_names (str | list): Name of the restriction enzyme or list of enzyme names
        dna_sequence (str): Input DNA sequence
        is_circular (bool): Whether the DNA sequence is circular (True) or linear (False)

    Returns:
        Dict: Dictionary containing the digestion fragments and their properties including positions

    """
    # Convert sequence to Biopython Seq object
    seq = Seq(dna_sequence)
    seq_length = len(seq)

    # Get all cut positions for all enzymes
    all_cut_positions = []
    for enzyme_name in enzyme_names:
        enzyme_obj = getattr(Restriction, enzyme_name)
        cut_sites = enzyme_obj.search(seq, linear=not is_circular)
        all_cut_positions.extend(cut_sites)

    # Sort cut positions and remove duplicates
    all_cut_positions = sorted(set(all_cut_positions))

    # Calculate fragments with positions
    fragments = []
    if not all_cut_positions:
        # No cuts - return full sequence
        fragments.append({"fragment": str(seq), "length": seq_length, "start": 0, "end": seq_length})
    # Handle linear and circular cases
    elif is_circular:
        for i in range(len(all_cut_positions)):
            start = all_cut_positions[i]
            end = all_cut_positions[0] + seq_length if i == len(all_cut_positions) - 1 else all_cut_positions[i + 1]

            # Get fragment sequence
            if end > seq_length:
                fragment_seq = dna_sequence[start:] + dna_sequence[: end - seq_length]
            else:
                fragment_seq = dna_sequence[start:end]

            fragments.append(
                {
                    "fragment": fragment_seq,
                    "length": len(fragment_seq),
                    "start": start,
                    "end": end if end <= seq_length else end - seq_length,
                    "is_wrapped": end > seq_length,
                }
            )
    else:
        # First fragment
        if all_cut_positions[0] > 0:
            fragments.append(
                {
                    "fragment": dna_sequence[: all_cut_positions[0]],
                    "length": all_cut_positions[0],
                    "start": 0,
                    "end": all_cut_positions[0],
                }
            )

        # Middle fragments
        for i in range(len(all_cut_positions) - 1):
            start = all_cut_positions[i]
            end = all_cut_positions[i + 1]
            fragments.append(
                {
                    "fragment": dna_sequence[start:end],
                    "length": end - start,
                    "start": start,
                    "end": end,
                }
            )

        # Last fragment
        if all_cut_positions[-1] < seq_length:
            fragments.append(
                {
                    "fragment": dna_sequence[all_cut_positions[-1] :],
                    "length": seq_length - all_cut_positions[-1],
                    "start": all_cut_positions[-1],
                    "end": seq_length,
                }
            )

    # Sort fragments by length (descending)
    fragments.sort(key=lambda x: x["length"], reverse=True)

    results = {
        "explanation": (
            "Output fields:\n"
            "- sequence_info: Information about the input sequence\n"
            "  * length: Length of the sequence in base pairs\n"
            "  * is_circular: Whether sequence is circular or linear\n"
            "- digestion_info: Overview of digestion results\n"
            "  * enzymes_used: List of restriction enzymes used\n"
            "  * number_of_fragments: Total number of fragments produced\n"
            "  * cut_positions: List of all cut positions in sequence\n"
            "- fragments: List of all fragments produced, where each contains:\n"
            "  * fragment: The DNA sequence of the fragment\n"
            "  * length: Length of the fragment in base pairs\n"
            "  * start: Start position in original sequence (0-based)\n"
            "  * end: End position in original sequence\n"
            "  * is_wrapped: (Only for circular) Whether fragment wraps around sequence end"
        ),
        "sequence_info": {
            "length": len(seq),
            "is_circular": is_circular,
        },
        "digestion_info": {
            "enzymes_used": enzyme_names,
            "number_of_fragments": len(fragments),
            "cut_positions": all_cut_positions,
        },
        "fragments": fragments,
    }

    return results


def find_restriction_sites(dna_sequence: str, enzymes: list[str], is_circular: bool = True) -> dict:
    """Identifies restriction enzyme sites in a given DNA sequence for specified enzymes.

    Args:
        dna_sequence (str): Complete input DNA sequence
        enzymes (List[str]): List of restriction enzyme names to check
        is_circular (bool): Whether the DNA sequence is circular (True) or linear (False)

    Returns:
        Dict: Dictionary containing all identified restriction sites

    """
    # Convert string to Bio.Seq object
    seq = Seq(dna_sequence.upper())

    # Create Analysis batch with specified enzymes
    rb = Restriction.RestrictionBatch(enzymes)

    # Analyze sequence for restriction sites, considering topology
    analysis = rb.search(seq, linear=not is_circular)

    results = {
        "explanation": (
            "Output fields:\n"
            "- sequence_info: Information about the input sequence\n"
            "  * length: Length of the sequence in base pairs\n"
            "  * is_circular: Whether sequence is circular or linear\n"
            "- restriction_sites: Dictionary of enzymes and their sites, where each contains:\n"
            "  * recognition_sequence: The DNA sequence the enzyme recognizes\n"
            "  * cut_positions: Details about where enzyme cuts relative to site\n"
            "    - 5_prime: Cut position on 5' strand relative to start of recognition site\n"
            "    - 3_prime: Cut position on 3' strand relative to start of recognition site\n"
            "    - overhang: Length of overhang produced (negative for 3' overhang)\n"
            "    - overhang_type: 'sticky' for overhanging cuts, 'blunt' for even cuts\n"
            "  * sites: List of positions where enzyme cuts in the sequence (0-based)"
        ),
        "sequence_info": {
            "length": len(seq),
            "is_circular": is_circular,
        },
        "restriction_sites": {},
    }

    for enzyme, positions in analysis.items():
        if positions:  # Only include enzymes that have recognition sites
            enzyme_info = {
                "recognition_sequence": str(enzyme.elucidate()),
                "cut_positions": {
                    "5_prime": enzyme.fst5,  # Cut position relative to recognition site
                    "3_prime": enzyme.fst3,
                    "overhang": enzyme.ovhg,
                    "overhang_type": "sticky" if enzyme.ovhg != 0 else "blunt",
                },
                "sites": sorted(positions),  # Sort positions for better readability
            }

            results["restriction_sites"][str(enzyme)] = enzyme_info
        else:
            # Include enzymes with no sites found
            results["restriction_sites"][str(enzyme)] = {
                "recognition_sequence": str(enzyme.elucidate()),
                "sites": [],
            }
    return results


def find_restriction_enzymes(sequence: str, is_circular: bool = False) -> dict[str, list]:
    """Finds common restriction enzyme sites in a DNA sequence.

    Args:
        sequence (str): DNA sequence to analyze
        is_circular (bool): Whether the sequence is circular (default: False)

    Returns:
        Dict[str, list]: Dictionary of enzymes and their cut positions

    """
    # Convert to Bio.Seq and analyze
    seq = Seq(sequence.upper())
    analysis = Restriction.CommOnly.search(seq, linear=not is_circular)

    # Keep only enzymes that have sites
    sites = {str(enzyme): list(positions) for enzyme, positions in analysis.items() if positions}

    return {
        "explanation": (
            "Output fields:\n"
            "- enzyme_sites: Dictionary where keys are enzyme names and values are:\n"
            "  * List of cut positions in the sequence (0-based)\n"
            "  * Empty list means enzyme has no recognition sites\n"
            "  * Positions indicate where the enzyme's recognition sequence begins\n"
            "  * For circular sequences, positions wrap around the sequence end"
        ),
        "enzyme_sites": sites,
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


def design_knockout_sgrna(
    gene_name: str,
    data_lake_path: str,
    species: str = "human",
    num_guides: int = 1,
) -> dict[str, Any]:
    """Design sgRNAs for CRISPR knockout by searching pre-computed sgRNA libraries.
    Returns optimized guide RNAs for targeting a specific gene.

    Args:
        gene_name (str): Target gene symbol/name (e.g., "EGFR", "TP53")
        species (str): Target organism species (default: "human")
        num_guides (int): Number of guides to return (default: 1)

    Returns:
        Dict: Dictionary containing:
            - explanation: Explanation of the output fields
            - gene_name: Target gene name
            - species: Target species
            - guides: List of sgRNA sequences

    """
    DEFAULT_LIBRARIES = {
        "human": data_lake_path + "/sgRNA_KO_SP_human.txt",
        "mouse": data_lake_path + "/sgRNA_KO_SP_mouse.txt",
    }

    # Use fixed library pathAdd commentMore actions
    library_path = DEFAULT_LIBRARIES[species.lower()]

    # Check if library file exists
    if not os.path.exists(library_path):
        error_msg = f"Library file for {species} not found at path: {library_path}"
        error_msg += "\n\nThis error typically occurs when the datalake files are not available."
        raise FileNotFoundError(error_msg)

    # Load sgRNA library from S3
    try:
        df = pd.read_csv(library_path, delimiter="\t")
    except Exception as e:
        raise RuntimeError(f"Failed to load sgRNA library: {str(e)}") from None

    # Filter for target gene
    gene_name = gene_name.upper()  # Ensure consistent capitalization
    gene_df = df[df["Target Gene Symbol"].str.upper() == gene_name]

    if gene_df.empty:
        # Try partial matching if exact match fails
        gene_df = df[df["Target Gene Symbol"].str.upper().str.contains(gene_name)]

    if gene_df.empty:
        return {
            "explanation": "Output contains target gene name, species, and list of sgRNA sequences",
            "gene_name": gene_name,
            "species": species,
            "guides": [],
        }

    # Sort by combined rank (default priority)
    gene_df = gene_df.sort_values(by=["Combined Rank"])

    # Get top guides
    top_guides = gene_df.head(num_guides)

    # Extract just the sgRNA sequences
    guides = top_guides["sgRNA Sequence"].tolist()

    return {
        "explanation": "Output contains target gene name, species, and list of sgRNA sequences",
        "gene_name": gene_name,
        "species": species,
        "guides": guides,
    }


def get_oligo_annealing_protocol() -> dict[str, Any]:
    """Return a standard protocol for annealing oligonucleotides without phosphorylation.

    Returns:
        Dict: Dictionary containing detailed protocol steps for oligo annealing

    """
    protocol = {
        "title": "Oligo Annealing Protocol",
        "description": "Standard protocol for annealing complementary oligonucleotides",
        "steps": [
            {
                "step_number": 1,
                "title": "Prepare annealing reaction",
                "description": "Mix the following components in a PCR tube:",
                "components": [
                    {"name": "Forward oligo (100 μM)", "volume": "1 μl"},
                    {"name": "Reverse oligo (100 μM)", "volume": "1 μl"},
                    {"name": "Nuclease-free water", "volume": "8 μl"},
                ],
                "total_volume": "10 μl",
            },
            {
                "step_number": 2,
                "title": "Anneal in thermocycler",
                "description": "Run the following program on a thermocycler:",
                "program": [
                    {
                        "temperature": "95°C",
                        "time": "5 minutes",
                        "description": "Initial denaturation",
                    },
                    {
                        "temperature": "Ramp down to 25°C",
                        "rate": "5°C/minute",
                        "description": "Slow cooling for proper annealing",
                    },
                ],
            },
            {
                "step_number": 3,
                "title": "Dilute annealed oligos",
                "description": "Dilute the annealed oligos 1:200 in nuclease-free water",
                "details": "Add 1 μl of annealed oligos to 199 μl of nuclease-free water",
            },
        ],
        "notes": [
            "Store annealed and diluted oligos at -20°C for long-term storage",
            "Diluted oligos can be used directly in ligation reactions",
        ],
    }

    return protocol


def get_golden_gate_assembly_protocol(
    num_inserts: int = 1,
    enzyme_name: str = None,
    vector_length: int = None,
    vector_amount_ng: float = 75.0,
    insert_lengths: list[int] = None,
    is_library_prep: bool = False,
) -> dict[str, Any]:
    """Return a customized protocol for Golden Gate assembly based on the number of inserts
    and specific DNA sequences.

    Args:
        num_inserts (int): Number of inserts to be assembled (default: 1)
        enzyme_name (str): Type IIS restriction enzyme to be used
        vector_length (int): Length of the destination vector in bp
        vector_amount_ng (float): Amount of vector to use in ng (default: 75.0)
        insert_lengths (List[int]): List of insert lengths in bp (optional)
        is_library_prep (bool): Whether this is for library preparation (default: False)

    Returns:
        Dict: Dictionary containing detailed protocol steps for Golden Gate assembly

    """
    # Validate enzyme name
    supported_enzymes = ["BsaI", "BsmBI", "BbsI", "Esp3I", "BtgZI", "SapI"]
    if enzyme_name not in supported_enzymes:
        raise ValueError(f"Unsupported enzyme: {enzyme_name}. Currently supporting: {', '.join(supported_enzymes)}")

    # Determine which protocol to use based on number of inserts
    if num_inserts == 1:
        if is_library_prep:
            thermal_protocol = [
                {
                    "temperature": "37°C",
                    "time": "1 hour",
                    "description": "Cleavage and ligation",
                }
            ]
        else:
            thermal_protocol = [
                {
                    "temperature": "37°C",
                    "time": "5 minutes",
                    "description": "Cleavage and ligation",
                }
            ]
        thermal_protocol.append(
            {
                "temperature": "60°C",
                "time": "5 minutes",
                "description": "Enzyme inactivation",
            }
        )
    elif 2 <= num_inserts <= 10:
        thermal_protocol = [
            {
                "temperature": "Cycle (30x)",
                "description": "Cleavage and ligation cycles",
                "cycles": [
                    {
                        "temperature": "37°C",
                        "time": "1 minute",
                        "description": "Restriction digestion",
                    },
                    {
                        "temperature": "16°C",
                        "time": "1 minute",
                        "description": "Ligation",
                    },
                ],
            },
            {
                "temperature": "60°C",
                "time": "5 minutes",
                "description": "Enzyme inactivation",
            },
        ]
    else:  # 11+ inserts
        thermal_protocol = [
            {
                "temperature": "Cycle (30x)",
                "description": "Cleavage and ligation cycles",
                "cycles": [
                    {
                        "temperature": "37°C",
                        "time": "5 minutes",
                        "description": "Restriction digestion",
                    },
                    {
                        "temperature": "16°C",
                        "time": "5 minutes",
                        "description": "Ligation",
                    },
                ],
            },
            {
                "temperature": "60°C",
                "time": "5 minutes",
                "description": "Enzyme inactivation",
            },
        ]

    # Determine NEB Golden Gate Assembly Mix amount
    assembly_mix_volume = "2 μl" if num_inserts > 10 else "1 μl"

    # Calculate molar amounts
    # Formula: pmol = (weight in ng) / (base pairs × 650 g/mol per bp) × 10^6
    vector_pmol = vector_amount_ng / (vector_length * 650) * 1000000

    # Calculate insert amounts
    insert_components = []

    # If lengths are provided, use them for calculations
    if insert_lengths:
        for i, length in enumerate(insert_lengths):
            # Calculate for 2:1 molar ratio (insert:vector)
            insert_pmol = 2 * vector_pmol
            insert_ng = (insert_pmol * length * 650) / 1000000

            insert_components.append(
                {
                    "name": f"Insert {i + 1} ({length} bp)",
                    "amount": f"{insert_ng:.1f} ng ({insert_pmol:.3f} pmol)",
                    "molar_ratio": "2:1 (insert:vector)",
                }
            )
    else:
        # Generic insert information when no lengths provided
        insert_components = [
            {
                "name": "Insert DNA (precloned or amplicon)",
                "amount": "Variable based on length and concentration",
                "note": "Use 2:1 molar ratio (insert:vector) for optimal assembly",
            }
        ]

    # Create component list for the protocol
    reaction_components = [
        {
            "name": f"Destination Vector ({vector_length} bp)",
            "amount": f"{vector_amount_ng} ng ({vector_pmol:.3f} pmol)",
        }
    ]

    # Add insert components
    for component in insert_components:
        reaction_components.append(component)

    # Add remaining reagents
    reaction_components.extend(
        [
            {"name": "T4 DNA Ligase Buffer (10X)", "volume": "2 μl"},
            {
                "name": f"NEB Golden Gate Assembly Mix ({enzyme_name})",
                "volume": assembly_mix_volume,
            },
            {"name": "Nuclease-free H₂O", "volume": "to 20 μl"},
        ]
    )

    protocol = {
        "title": f"Golden Gate Assembly Protocol ({enzyme_name})",
        "description": f"Customized protocol for Golden Gate assembly with {enzyme_name} and {num_inserts} insert(s)",
        "steps": [
            {
                "step_number": 1,
                "title": "Prepare assembly reaction",
                "description": "Mix the following components in a PCR tube:",
                "components": reaction_components,
                "total_volume": "20 μl",
            },
            {
                "step_number": 2,
                "title": "Run assembly reaction",
                "description": "Run the following program on a thermocycler:",
                "program": thermal_protocol,
            },
        ],
        "notes": [
            f"Destination vector must possess {enzyme_name} restriction sites in the proper orientation",
            f"Inserts must possess {enzyme_name} restriction sites at both ends in the proper orientation",
            "For amplicon inserts, add 5′ flanking bases (6 recommended) before the restriction sites",
            f"Vector amount: {vector_amount_ng} ng = {vector_pmol:.3f} pmol",
            "Insert:vector molar ratio is 2:1 for optimal assembly efficiency",
        ],
    }

    return protocol


def get_bacterial_transformation_protocol(
    antibiotic: str = "ampicillin", is_repetitive: bool = False
) -> dict[str, Any]:
    """Return a standard protocol for bacterial transformation.

    Args:
        antibiotic (str): Selection antibiotic (default: "ampicillin")
        is_repetitive (bool): Whether the sequence contains repetitive elements (default: False)

    Returns:
        Dict: Dictionary containing detailed protocol steps for bacterial transformation

    """
    # Set incubation temperature based on whether sequence is repetitive
    incubation_temp = "30°C" if is_repetitive else "37°C"

    protocol = {
        "title": "Bacterial Transformation Protocol",
        "description": "Standard protocol for transforming DNA into competent E. coli cells",
        "steps": [
            {
                "step_number": 1,
                "title": "Add DNA to competent cells",
                "description": "Add 5 μl of DNA to 50 μl of competent E. coli cells",
                "note": "Keep cells on ice during this step and handle gently",
            },
            {
                "step_number": 2,
                "title": "Ice incubation",
                "description": "Incubate on ice for 30 minutes",
                "note": "This allows DNA to associate with the cell membrane",
            },
            {
                "step_number": 3,
                "title": "Heat shock",
                "description": "Heat shock at 42°C for 45 seconds",
                "note": "Precise timing is critical",
            },
            {
                "step_number": 4,
                "title": "Recovery on ice",
                "description": "Return to ice for 2 minutes",
                "note": "This step helps cells recover from heat shock",
            },
            {
                "step_number": 5,
                "title": "Add recovery medium",
                "description": "Add 950 μl of SOC medium",
                "note": "SOC is preferred, but LB can be used if necessary",
            },
            {
                "step_number": 6,
                "title": "Recovery incubation",
                "description": f"Incubate at {incubation_temp} for 1 hour with shaking (200-250 rpm)",
                "note": "This allows expression of antibiotic resistance genes",
            },
            {
                "step_number": 7,
                "title": "Plate on selective media",
                "description": f"Plate 100 μl on LB agar plates containing 100 μg/ml {antibiotic}",
                "note": "Spread thoroughly using sterile glass beads or a spreader",
            },
            {
                "step_number": 8,
                "title": "Incubate plates",
                "description": f"Incubate overnight (16-18 hours) at {incubation_temp}",
                "note": "Invert plates to prevent condensation from dripping onto colonies",
            },
        ],
        "notes": [
            "Always include positive and negative controls for transformation",
            f"For repetitive sequences, the lower temperature ({incubation_temp}) helps maintain sequence integrity"
            if is_repetitive
            else "Standard incubation at 37°C works well for most plasmids",
            f"Use fresh plates containing {antibiotic} for best results",
        ],
    }

    return protocol


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


def design_golden_gate_oligos(
    backbone_sequence: str,
    insert_sequence: str,
    enzyme_name: str,
    is_circular: bool = True,
) -> dict[str, Any]:
    """Design oligos for Golden Gate assembly by identifying backbone overhangs
    and creating matching insert oligos.

    Args:
        backbone_sequence (str): The plasmid/backbone sequence
        insert_sequence (str): Sequence to be inserted
        enzyme_name (str): Type IIS restriction enzyme to be used
        is_circular (bool): Whether the backbone is circular (default: True)

    Returns:
        Dict: Dictionary containing overhang information and designed oligos

    """
    # Dictionary of enzyme properties
    TYPE_IIS_PROPERTIES = {
        "BsaI": {"recognition_site": "GGTCTC", "offset_fwd": 1, "offset_rev": 5},
        "BsmBI": {"recognition_site": "CGTCTC", "offset_fwd": 1, "offset_rev": 5},
        "BbsI": {"recognition_site": "GAAGAC", "offset_fwd": 2, "offset_rev": 6},
        "Esp3I": {"recognition_site": "CGTCTC", "offset_fwd": 1, "offset_rev": 5},
        "BtgZI": {"recognition_site": "GCGATG", "offset_fwd": 10, "offset_rev": 14},
        "SapI": {"recognition_site": "GCTCTTC", "offset_fwd": 1, "offset_rev": 4},
    }

    if enzyme_name not in TYPE_IIS_PROPERTIES:
        supported = ", ".join(TYPE_IIS_PROPERTIES.keys())
        return {
            "success": False,
            "message": f"Unsupported enzyme: {enzyme_name}. Currently supporting: {supported}",
        }

    # Clean input sequences
    backbone_sequence = "".join(c for c in backbone_sequence.upper() if c in "ATGC")
    insert_sequence = "".join(c for c in insert_sequence.upper() if c in "ATGC")

    # Get enzyme properties
    enzyme_props = TYPE_IIS_PROPERTIES[enzyme_name]
    recognition_site = enzyme_props["recognition_site"]
    offset_fwd = enzyme_props["offset_fwd"]
    offset_rev = enzyme_props["offset_rev"]

    # Reverse complement function
    def reverse_complement(seq):
        """Get the reverse complement of a DNA sequence."""
        complement = {"A": "T", "T": "A", "G": "C", "C": "G"}
        return "".join(complement.get(base, "N") for base in reversed(seq))

    # Step 1: Find all restriction sites in the backbone
    restriction_sites = []
    for i in range(len(backbone_sequence)):
        # Check if this position starts a recognition site
        if is_circular and i + len(recognition_site) > len(backbone_sequence):
            site_seq = backbone_sequence[i:] + backbone_sequence[: i + len(recognition_site) - len(backbone_sequence)]
        elif i + len(recognition_site) <= len(backbone_sequence):
            site_seq = backbone_sequence[i : i + len(recognition_site)]
        else:
            continue

        # Check forward strand recognition site
        if site_seq == recognition_site:
            restriction_sites.append({"position": i, "strand": "forward"})

        # Check reverse complement recognition site
        rev_comp_site = reverse_complement(recognition_site)
        if site_seq == rev_comp_site:
            restriction_sites.append({"position": i, "strand": "reverse"})

    if len(restriction_sites) < 2:
        return {
            "success": False,
            "message": f"Need at least 2 {enzyme_name} recognition sites in the backbone for Golden Gate assembly.",
        }

    # Step 2: Determine the cut positions and resulting overhangs
    cut_sites = []

    for site in restriction_sites:
        pos = site["position"]

        if site["strand"] == "forward":
            # Calculate cut positions for forward strand site
            cut_fwd = (pos + len(recognition_site) + offset_fwd) % len(backbone_sequence)
            cut_rev = (pos + len(recognition_site) + offset_rev) % len(backbone_sequence)
        else:  # reverse strand
            # For reverse sites, cuts happen on the other side of the recognition site
            cut_rev = (pos - offset_fwd) % len(backbone_sequence)
            cut_fwd = (pos - offset_rev) % len(backbone_sequence)

            if cut_rev < 0:
                cut_rev += len(backbone_sequence)
            if cut_fwd < 0:
                cut_fwd += len(backbone_sequence)

        # Extract overhang sequence
        if cut_fwd < cut_rev:
            overhang = backbone_sequence[cut_fwd:cut_rev]
        else:  # Handle wrapping for circular sequences
            overhang = backbone_sequence[cut_fwd:] + backbone_sequence[:cut_rev]

        cut_sites.append(
            {
                "site_position": pos,
                "strand": site["strand"],
                "cut_fwd": cut_fwd,
                "cut_rev": cut_rev,
                "overhang": overhang,
            }
        )

    # Step 3: For Golden Gate, identify the two overhangs that will flank the insert
    # This is simplified - in practice, you'd want to identify which cut sites
    # correspond to the destination for your insert

    # For demonstration, let's use the first two cut sites
    # In a real application, we would need a more sophisticated algorithm to identify
    # which sites are relevant to the insertion

    # We'll assume the first overhang goes at the 5' end of the insert
    # and the second overhang (reverse complemented) goes at the 3' end
    upstream_overhang = cut_sites[0]["overhang"]
    downstream_overhang = cut_sites[1]["overhang"]

    # Step 4: Design simple oligos with matching overhangs
    # For forward oligo, add the upstream overhang
    fw_oligo = upstream_overhang + insert_sequence

    # For reverse oligo, use reverse complement of insert + reverse complement of downstream overhang
    rev_oligo = reverse_complement(insert_sequence + reverse_complement(downstream_overhang))

    return {
        "success": True,
        "overhangs": {"upstream": upstream_overhang, "downstream": downstream_overhang},
        "oligos": {
            "forward": fw_oligo,
            "reverse": rev_oligo,
            "notes": [
                f"Forward oligo: Add {upstream_overhang} to 5' end of your insert",
                f"Reverse oligo: Add {reverse_complement(downstream_overhang)} to 5' end of reverse complement of your insert",
            ],
        },
        "cut_sites": [{"position": site["site_position"], "overhang": site["overhang"]} for site in cut_sites],
        "assembly_notes": f"Found {len(restriction_sites)} {enzyme_name} sites. Using overhangs {upstream_overhang} and {downstream_overhang} for assembly.",
    }


def golden_gate_assembly(
    backbone_sequence: str,
    enzyme_name: str,
    fragments: list[dict[str, str]],
    is_circular: bool = True,
) -> dict[str, Any]:
    """Simulate Golden Gate assembly to predict final construct sequences.

    Args:
        backbone_sequence (str): Complete backbone sequence
        enzyme_name (str): Type IIS restriction enzyme to be used (e.g., "BsmBI", "BsaI")
        fragments (List[Dict[str, str]]): List of fragments to insert, containing one of:
            - name + fwd_oligo + rev_oligo: Oligo pair with matching overhangs
            - name + sequence: Double-stranded DNA fragment containing restriction sites
        is_circular (bool): Whether the backbone is circular (default: True)

    Returns:
        Dict: Dictionary containing:
            - success: Boolean indicating if assembly was successful
            - assembled_sequence: The final assembled sequence
            - message: Error message if assembly failed

    """
    # Dictionary of enzyme properties
    TYPE_IIS_PROPERTIES = {
        "BsaI": {"recognition_site": "GGTCTC", "offset_fwd": 1, "offset_rev": 5},
        "BsmBI": {"recognition_site": "CGTCTC", "offset_fwd": 1, "offset_rev": 5},
        "BbsI": {"recognition_site": "GAAGAC", "offset_fwd": 2, "offset_rev": 6},
        "Esp3I": {"recognition_site": "CGTCTC", "offset_fwd": 1, "offset_rev": 5},
        "BtgZI": {"recognition_site": "GCGATG", "offset_fwd": 10, "offset_rev": 14},
        "SapI": {"recognition_site": "GCTCTTC", "offset_fwd": 1, "offset_rev": 4},
    }

    if enzyme_name not in TYPE_IIS_PROPERTIES:
        supported = ", ".join(TYPE_IIS_PROPERTIES.keys())
        return {
            "success": False,
            "message": f"Unsupported enzyme: {enzyme_name}. Currently supporting: {supported}",
            "assembled_sequence": None,
        }

    # Get enzyme properties
    enzyme_props = TYPE_IIS_PROPERTIES[enzyme_name]
    recognition_site = enzyme_props["recognition_site"]
    offset_fwd = enzyme_props["offset_fwd"]
    offset_rev = enzyme_props["offset_rev"]

    # Prepare reverse complement function
    def reverse_complement(seq):
        """Get the reverse complement of a DNA sequence."""
        complement = {"A": "T", "T": "A", "G": "C", "C": "G"}
        return "".join(complement.get(base, "N") for base in reversed(seq))

    # Standardize and clean sequences
    backbone_sequence = backbone_sequence.upper()

    # Process fragments to ensure they all have fwd_oligo and rev_oligo
    processed_fragments = []

    for idx, fragment in enumerate(fragments):
        fragment_name = fragment.get("name", f"fragment_{idx + 1}")

        # Check if this is a double-stranded fragment (contains 'sequence' but not oligos)
        if "sequence" in fragment and not ("fwd_oligo" in fragment and "rev_oligo" in fragment):
            # Process the double-stranded fragment to extract oligos
            ds_sequence = fragment["sequence"].upper()

            # Find restriction sites in the fragment
            fwd_sites = []
            rev_sites = []
            rev_comp_site = reverse_complement(recognition_site)

            # Find forward sites
            for i in range(len(ds_sequence) - len(recognition_site) + 1):
                site_seq = ds_sequence[i : i + len(recognition_site)]
                if site_seq == recognition_site:
                    fwd_sites.append(i)
                elif site_seq == rev_comp_site:
                    rev_sites.append(i)

            # Need exactly two sites for Golden Gate (one in each direction)
            if len(fwd_sites) == 0 or len(rev_sites) == 0:
                return {
                    "success": False,
                    "message": f"Fragment '{fragment_name}' must contain at least one {enzyme_name} site in each orientation",
                    "assembled_sequence": None,
                }

            # Find the first forward and last reverse site
            fwd_site = fwd_sites[0]
            rev_site = rev_sites[-1]

            # Calculate cut positions
            fwd_cut = fwd_site + len(recognition_site) + offset_rev
            # Fix: For reverse sites, cut happens offset_rev bases BEFORE (not after) the site
            rev_cut = rev_site - offset_rev

            # If the forward cut is after the reverse cut, the fragment is invalid
            if fwd_cut >= rev_cut:
                return {
                    "success": False,
                    "message": f"Invalid restriction sites in fragment '{fragment_name}': sites must be oriented to excise the insert",
                    "assembled_sequence": None,
                }

            # Extract the insert sequence (between the cuts)
            insert_seq = ds_sequence[fwd_cut:rev_cut]
            print(insert_seq)

            # Extract the overhangs (4 bases for most Type IIS enzymes)
            fwd_overhang = ds_sequence[fwd_cut - 4 : fwd_cut]
            rev_overhang_rc = ds_sequence[rev_cut : rev_cut + 4]
            rev_overhang = reverse_complement(rev_overhang_rc)

            # Create equivalent oligos for assembly
            fwd_oligo = fwd_overhang + insert_seq
            rev_oligo = rev_overhang + reverse_complement(insert_seq)

            processed_fragments.append(
                {
                    "name": fragment_name,
                    "fwd_oligo": fwd_oligo,
                    "rev_oligo": rev_oligo,
                    "original_sequence": ds_sequence,
                }
            )

        # Handle traditional oligo format
        elif "fwd_oligo" in fragment and "rev_oligo" in fragment:
            processed_fragments.append(
                {
                    "name": fragment_name,
                    "fwd_oligo": fragment["fwd_oligo"],
                    "rev_oligo": fragment["rev_oligo"],
                }
            )
        else:
            return {
                "success": False,
                "message": f"Fragment '{fragment_name}' must contain either 'sequence' or both 'fwd_oligo' and 'rev_oligo'",
                "assembled_sequence": None,
            }
    print(processed_fragments)

    # Step 1: Find all restriction sites in the backbone
    restriction_sites = []
    # For every position in the backbone sequence
    for i in range(len(backbone_sequence)):
        # Check if this position starts a recognition site
        # Handle circular sequences by wrapping around
        if is_circular:
            # Check if we need to wrap around for the recognition site
            if i + len(recognition_site) > len(backbone_sequence):
                site_seq = (
                    backbone_sequence[i:] + backbone_sequence[: i + len(recognition_site) - len(backbone_sequence)]
                )
            else:
                site_seq = backbone_sequence[i : i + len(recognition_site)]
        # For linear sequences, just check if we're in bounds
        elif i + len(recognition_site) <= len(backbone_sequence):
            site_seq = backbone_sequence[i : i + len(recognition_site)]
        else:
            continue

        # Check forward strand recognition site
        if site_seq == recognition_site:
            restriction_sites.append({"position": i, "strand": "forward"})

        # Check reverse complement recognition site
        rev_comp_site = reverse_complement(recognition_site)
        if site_seq == rev_comp_site:
            restriction_sites.append({"position": i, "strand": "reverse"})

    if not restriction_sites:
        return {
            "success": False,
            "message": f"No {enzyme_name} recognition sites found in the backbone",
            "assembled_sequence": None,
        }

    # Continue with the rest of the function using processed_fragments
    # (remaining code is unchanged, just operate on processed_fragments)

    # Step 2: Determine the cut positions and resulting overhangs
    cut_fragments = []

    for site in restriction_sites:
        pos = site["position"]

        if site["strand"] == "forward":
            # Calculate cut positions for forward strand site
            cut_fwd = (pos + len(recognition_site) + offset_fwd) % len(backbone_sequence)
            cut_rev = (pos + len(recognition_site) + offset_rev) % len(backbone_sequence)
        else:  # reverse strand
            # For reverse sites, cuts happen on the other side of the recognition site
            cut_rev = (pos - offset_fwd) % len(backbone_sequence)
            cut_fwd = (pos - offset_rev) % len(backbone_sequence)

            if cut_rev < 0:
                cut_rev += len(backbone_sequence)
            if cut_fwd < 0:
                cut_fwd += len(backbone_sequence)

        # Ensure cut positions are valid
        if 0 <= cut_fwd < len(backbone_sequence) and 0 <= cut_rev < len(backbone_sequence):
            # Extract overhang sequence
            if cut_fwd < cut_rev:
                overhang = backbone_sequence[cut_fwd:cut_rev]
            else:  # Handle wrapping for circular sequences
                overhang = backbone_sequence[cut_fwd:] + backbone_sequence[:cut_rev]

            cut_fragments.append(
                {
                    "site_position": pos,
                    "strand": site["strand"],
                    "cut_fwd": cut_fwd,
                    "cut_rev": cut_rev,
                    "overhang": overhang,
                }
            )

    if not cut_fragments:
        return {
            "success": False,
            "message": f"Could not determine valid cut positions for {enzyme_name} sites",
            "assembled_sequence": None,
        }

    # Step 3: Extract fragment overhangs from oligos
    fragment_overhangs = []
    for idx, fragment in enumerate(processed_fragments):
        # Clean oligo sequences
        fwd_oligo = "".join(c for c in fragment.get("fwd_oligo", "").upper() if c in "ATGC")
        rev_oligo = "".join(c for c in fragment.get("rev_oligo", "").upper() if c in "ATGC")

        # Get the overhang length (typically 4 bases for most Type IIS enzymes)
        overhang_length = 4

        # Extract overhangs and insert sequence
        fwd_overhang = fwd_oligo[:overhang_length]
        rev_overhang = rev_oligo[:overhang_length]

        # The actual insert is everything after the overhang in the forward oligo
        insert_seq = fwd_oligo[overhang_length:]

        fragment_overhangs.append(
            {
                "name": fragment.get("name", f"fragment_{idx + 1}"),
                "fwd_overhang": fwd_overhang,
                "rev_overhang": rev_overhang,
                "insert": insert_seq,
                "rc_rev_overhang": reverse_complement(rev_overhang),
            }
        )

    # Step 4: For multi-fragment assembly, determine the correct order
    # Create a map of overhangs to fragments
    fwd_overhang_map = {}  # Maps fwd overhangs to fragment indices
    rev_overhang_map = {}  # Maps rev overhangs to fragment indices

    for i, frag in enumerate(fragment_overhangs):
        fwd_overhang_map[frag["fwd_overhang"]] = i
        rev_overhang_map[frag["rc_rev_overhang"]] = i

    # Try to identify an entry point where a fragment's fwd overhang matches a backbone cut
    possible_starts = []
    for cut_idx, cut in enumerate(cut_fragments):
        if cut["overhang"] in fwd_overhang_map:
            possible_starts.append((cut_idx, fwd_overhang_map[cut["overhang"]]))

    if not possible_starts:
        return {
            "success": False,
            "message": "Could not find an entry point for assembly. No backbone cuts match fragment overhangs.",
            "assembled_sequence": None,
        }

    # Try each possible starting point
    for start_cut_idx, start_frag_idx in possible_starts:
        # Assemble a chain of fragments starting from this one
        assembled_fragments = [start_frag_idx]
        current_frag_idx = start_frag_idx

        # Keep adding fragments until we can't find a matching one
        while True:
            current_frag = fragment_overhangs[current_frag_idx]
            next_overhang = current_frag["rc_rev_overhang"]

            # Check if this overhang matches another fragment's fwd overhang
            next_frag_idx = None
            for i, frag in enumerate(fragment_overhangs):
                if i != current_frag_idx and frag["fwd_overhang"] == next_overhang:
                    next_frag_idx = i
                    break

            if next_frag_idx is None:
                # No more fragments to add - check if this overhang matches a backbone cut
                break

            # Add this fragment to the chain and continue
            assembled_fragments.append(next_frag_idx)
            current_frag_idx = next_frag_idx

            # Prevent infinite loops by checking if we've used all fragments
            if len(assembled_fragments) >= len(processed_fragments):
                break

        # Check if we've used all fragments
        if len(assembled_fragments) == len(processed_fragments):
            # Check if the last fragment's overhang matches a backbone cut
            last_frag = fragment_overhangs[assembled_fragments[-1]]
            end_overhang = last_frag["rc_rev_overhang"]

            end_cut_idx = None
            for cut_idx, cut in enumerate(cut_fragments):
                if cut["overhang"] == end_overhang and cut_idx != start_cut_idx:
                    end_cut_idx = cut_idx
                    break

            if end_cut_idx is not None:
                # We have a complete assembly path!
                start_cut = cut_fragments[start_cut_idx]
                end_cut = cut_fragments[end_cut_idx]

                # Assemble the final sequence
                # First, extract the backbone segment we're keeping
                if is_circular:
                    # For circular plasmids, we keep the region between end_cut and start_cut
                    if end_cut["cut_rev"] <= start_cut["cut_fwd"]:
                        backbone_segment = backbone_sequence[end_cut["cut_rev"] : start_cut["cut_fwd"]]
                    else:
                        # Handle wrapping around the origin
                        backbone_segment = (
                            backbone_sequence[end_cut["cut_rev"] :] + backbone_sequence[: start_cut["cut_fwd"]]
                        )
                else:
                    # For linear backbones, we keep everything outside the cut region
                    backbone_segment = (
                        backbone_sequence[: min(start_cut["cut_fwd"], end_cut["cut_rev"])]
                        + backbone_sequence[max(start_cut["cut_fwd"], end_cut["cut_rev"]) :]
                    )

                # Then concatenate all fragment inserts in the correct order
                insert_segment = ""
                for i, frag_idx in enumerate(assembled_fragments):
                    # Include the overhang for the first fragment
                    if i == 0:
                        insert_segment += fragment_overhangs[frag_idx]["fwd_overhang"]

                    # Add the main insert sequence
                    insert_segment += fragment_overhangs[frag_idx]["insert"]

                    # Include the reverse overhang for each fragment
                    insert_segment += reverse_complement(fragment_overhangs[frag_idx]["rev_overhang"])

                # Final assembled sequence
                assembled_sequence = backbone_segment + insert_segment

                return {
                    "success": True,
                    "assembled_sequence": assembled_sequence,
                    "message": f"Successfully assembled {len(processed_fragments)} fragments in correct order",
                    "fragments_used": len(processed_fragments),
                    "cuts_used": 2,  # We use exactly 2 cuts for the assembly
                    "backbone_fragment_size": len(backbone_segment),
                    "assembly_order": [processed_fragments[i]["name"] for i in assembled_fragments],
                }

    # If we get here, we couldn't find a valid assembly
    return {
        "success": False,
        "message": "Could not find a valid assembly path. Fragment overhangs may not form a complete circuit.",
        "assembled_sequence": None,
    }


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
B = {"type": "boolean"}
NAMES = {"type": "array", "items": {"type": "string", "minLength": 1, "maxLength": 100}, "minItems": 1, "maxItems": 100}


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
