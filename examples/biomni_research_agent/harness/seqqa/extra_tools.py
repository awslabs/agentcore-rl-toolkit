"""Related official tools: Biomni molecular-biology functions (official.py) and a subset of its database
functions (db_subset.py), behind the bounded transport (transport.py)."""
from . import db_subset as database
from . import official as molecular
from .transport import BASES, endpoint_url

ONLINE_NAMES = set(BASES)
LOCAL_NAMES = {
    "find_sequence_mutations",
    "design_primer",
    "design_verification_primers",
}
EXTRA_NAMES = ONLINE_NAMES | LOCAL_NAMES


FUNCTIONS = {
    "find_sequence_mutations": molecular.find_sequence_mutations,
    "design_primer": molecular.design_primer,
    "design_verification_primers": molecular.design_verification_primers,
    **{name: getattr(database, name) for name in ONLINE_NAMES},
}


def specs(spec):
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
