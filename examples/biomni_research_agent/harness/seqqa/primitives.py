"""Fixed sequence operations. No code, expressions, files or task-aware solvers."""
import re
from collections import Counter

from Bio import Restriction
from Bio.Data import CodonTable, IUPACData
from Bio.Seq import Seq

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


FUNCTIONS = {
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


def specs(spec):
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
