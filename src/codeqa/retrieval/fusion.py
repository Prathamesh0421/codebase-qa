"""Reciprocal Rank Fusion: combine several ranked lists into one, without
ever comparing their scores directly.

Vector cosine similarity and Postgres's ts_rank live on genuinely different,
untunable scales -- there's no principled constant that converts one into
the other. RRF sidesteps the problem by consuming only each item's RANK
POSITION in each list, not its score:

    RRF(d) = sum over every ranking r containing d of  1 / (k + rank_r(d))

k=60 is the constant from the original paper (Cormack, Clarke & Buettcher,
2009) and the conventional default; nothing here tunes it per-query.
"""

import re
from collections import defaultdict

_RRF_K = 60

# Bare identifier-shaped tokens (dispatch_request) and dotted qualified-name
# candidates (Flask.dispatch_request), extracted from raw query text for the
# symbol-matching component of hybrid retrieval.
#
# This function itself does NOT filter against a stopword list -- but
# hybrid.py's caller does apply filter_symbol_candidates (below) before
# sending bare tokens to the database, and that filtering turned out to be
# necessary, not optional: verified against the real Flask repo that
# ordinary English words are frequently ALSO real symbol names. "Flask",
# "view", and "request" (all present in "how does Flask dispatch a request
# to a view function?") are a real class, a real method name, and a real
# method name in Flask's own source. Unfiltered, the giant Flask class
# chunk won an exact bare-symbol match and outranked the actual answer
# (Flask.dispatch_request) in RRF fusion -- a regression caught by running
# this exact query against the real fixture, not by inspection.
_BARE_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_DOTTED_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+")


def extract_identifier_candidates(query_text: str) -> tuple[list[str], list[str]]:
    """(bare_names, dotted_names) -- candidate exact-match targets against
    chunks.symbol_name and chunks.qualified_name respectively.

    A dotted match like "Flask.dispatch_request" also independently appears
    in bare_names as "Flask" and "dispatch_request" -- both checks run, since
    a bare match against a differently-scoped chunk is still a real,
    independently useful signal, not noise to suppress.
    """
    # dict.fromkeys rather than set(): preserves first-seen order, which
    # keeps SQL parameter lists (and therefore query plans/logs) deterministic
    # across runs of the same query -- a set's iteration order is not.
    bare = list(dict.fromkeys(_BARE_TOKEN.findall(query_text)))
    dotted = list(dict.fromkeys(_DOTTED_TOKEN.findall(query_text)))
    return bare, dotted


def filter_symbol_candidates(bare: list[str], dotted: list[str]) -> tuple[list[str], list[str]]:
    """Narrow extract_identifier_candidates' output to tokens actually worth
    an exact-match database round trip against symbol_name/qualified_name.

    Filters on TOKEN SHAPE, not a hand-maintained stopword list: a
    multi-word compound identifier signals its own intent via snake_case
    (an underscore) or camelCase/PascalCase (an uppercase letter after the
    first character) -- "dispatch_request" and "SessionManager" both
    qualify. Plain lowercase English words don't, which is deliberate:
    "flask", "view", and "request" are all real symbol names in the real
    Flask repo (a class and two methods), and without this filter they
    out-rank the actual answer in RRF fusion (see _BARE_TOKEN's comment
    above for how this was found). A single-word, all-lowercase real
    symbol name (e.g. a function literally named "dispatch") loses its
    exact-match boost as a result -- accepted, since there's no shape-based
    way to distinguish that from the English verb, and false positives
    from common words are the worse failure mode of the two.

    Dotted names are never filtered: dottedness is itself a strong enough
    identifier signal (_DOTTED_TOKEN already requires it), and "Flask.
    dispatch_request" isn't a phrase that occurs in ordinary English text.
    """
    bare_filtered = [t for t in bare if "_" in t or any(c.isupper() for c in t[1:])]
    return bare_filtered, dotted


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = _RRF_K) -> dict[int, float]:
    """rankings: each a list of ids in rank order (best first). Returns every
    id that appeared in at least one ranking, mapped to its fused score --
    always higher-is-better, never meaningful compared to a raw cosine
    similarity or ts_rank value from before fusion.
    """
    scores: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] += 1.0 / (k + rank)
    return dict(scores)
