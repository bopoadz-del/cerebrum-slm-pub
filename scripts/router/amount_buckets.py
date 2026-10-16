"""Bucket exact amounts to magnitude tokens.

One tokenizer pass. The text is split on whitespace and punctuation. Ordinary
space, NBSP (U+00A0), thin space (U+2009) and narrow NBSP (U+202F) are
separator variants. Digit groups join one candidate across comma, dot, and
space-variant separators. A uniform run uses a single joiner string. A mixed
run (comma then space, or two space variants) is an amount when every group
after the first is exactly three digits, plus an optional ``.dd``, and no
date, phone, unit, or schedule guard fires. Magnitude is the digits with
every separator removed. Any other mixed run stays whole. It is not split
into an amount plus a leftover prefix.

Each candidate is classified once: amount, date, phone, unit quantity,
schedule or BOQ row, or code. Amounts become ``<AMT:~1eN>`` with
``N = floor(log10(abs(value)))`` of the separator-stripped number. A zero
amount is ``<AMT:~1e0>``. Slash-coded references become ``<REF>``. A lone
digit sitting against an amount token is a schedule row, ``<ROW>``, not an
amount fragment. Dates, percentages, years, phones, units and existing
``<...>`` tokens stay.

A written amount is two or more number-words, a hyphenated compound, or a
scale word (hundred and up). A single small cardinal ("one", "two") stays.
"""

from __future__ import annotations

import math

_SMALL: dict[str, int] = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_SCALE: dict[str, int] = {
    "hundred": 100,
    "hundreds": 100,
    "thousand": 1_000,
    "thousands": 1_000,
    "lakh": 100_000,
    "lakhs": 100_000,
    "million": 1_000_000,
    "millions": 1_000_000,
    "crore": 10_000_000,
    "crores": 10_000_000,
    "billion": 1_000_000_000,
    "billions": 1_000_000_000,
}
_NUM_WORDS = {**_SMALL, **_SCALE}
_MINOR_WORDS = frozenset(
    {"fils", "fills", "cent", "cents", "halala", "halalas"}
)
_MONTHS = frozenset(
    {
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sep",
        "sept",
        "oct",
        "nov",
        "dec",
    }
)
_UNITS = frozenset(
    {
        "mm",
        "cm",
        "km",
        "m",
        "m2",
        "m3",
        "m²",
        "m³",
        "mm²",
        "kg",
        "g",
        "t",
        "pc",
        "pcs",
        "nos",
        "nr",
        "sqm",
        "day",
        "days",
    }
)
_STRUCT = frozenset(
    {
        "clause",
        "clauses",
        "sub-clause",
        "subclause",
        "section",
        "sections",
        "table",
        "tables",
        "item",
        "items",
        "boq",
        "bill",
        "bills",
        "schedule",
        "row",
        "rows",
        "no",
        "no.",
        "rev",
        "revision",
        "para",
        "paragraph",
    }
)
_CURRENCY = frozenset(
    {
        "aed",
        "usd",
        "eur",
        "gbp",
        "sar",
        "qar",
        "dirham",
        "dirhams",
        "dollar",
        "dollars",
        "fils",
        "fills",
        "riyal",
        "riyals",
    }
)
_PHONE_WORDS = frozenset(
    {"tel", "telephone", "phone", "fax", "mobile", "cell", "mob"}
)
_SPACE = frozenset(" \u00a0\u2009\u202f")
_CODE_KINDS = frozenset({"word", "digit", "code"})
_REF = "<REF>"
_ROW = "<ROW>"


class Tok:
    __slots__ = ("kind", "start", "end", "text")

    def __init__(self, kind: str, start: int, end: int, text: str) -> None:
        self.kind = kind
        self.start = start
        self.end = end
        self.text = text


def magnitude_token(value: float) -> str:
    """``<AMT:~1eN>`` for the order of magnitude of ``value``."""
    absolute = abs(value)
    if absolute == 0:
        exponent = 0
    else:
        exponent = math.floor(math.log10(absolute))
    return f"<AMT:~1e{exponent}>"


def _tokenize(text: str) -> list[Tok]:
    tokens: list[Tok] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "<":
            end = text.find(">", i + 1, i + 64)
            if end != -1 and "\n" not in text[i : end + 1]:
                tokens.append(Tok("angle", i, end + 1, text[i : end + 1]))
                i = end + 1
                continue
        if ch.isdigit():
            j = i + 1
            while j < n and text[j].isdigit():
                j += 1
            tokens.append(Tok("digit", i, j, text[i:j]))
            i = j
            continue
        if ch.isalpha():
            j = i + 1
            while j < n and text[j].isalpha():
                j += 1
            while j < n and text[j] in "²³":
                j += 1
            tokens.append(Tok("word", i, j, text[i:j]))
            i = j
            continue
        if ch == "_":
            j = i + 1
            while j < n and text[j] == "_":
                j += 1
            tokens.append(Tok("code", i, j, text[i:j]))
            i = j
            continue
        tokens.append(Tok("sep", i, i + 1, ch))
        i += 1
    return tokens


def _is_ws(tok: Tok) -> bool:
    return tok.kind == "sep" and tok.text.isspace()


def _is_space_variant(text: str) -> bool:
    return len(text) == 1 and text in _SPACE


def _comma_joiner(text: str) -> bool:
    if "," not in text:
        return False
    if text.count(",") != 1:
        return False
    left, right = text.split(",", 1)
    if len(left) > 1 or len(right) > 1:
        return False
    if left and left not in _SPACE:
        return False
    if right and right not in _SPACE:
        return False
    return True


def _plausible_joiner(text: str) -> bool:
    return _is_space_variant(text) or text in {".", "/", "-"} or _comma_joiner(text)


def _joiner_kind(text: str) -> str:
    if "," in text:
        return "comma"
    if text == ".":
        return "dot"
    if text == "/":
        return "slash"
    if text == "-":
        return "hyphen"
    if _is_space_variant(text):
        return "space"
    return "other"


def _atom_kind(atom: str) -> tuple[str, int]:
    parts = atom.casefold().split("-")
    if len(parts) == 2 and all(part in _NUM_WORDS for part in parts):
        if parts[1] in _SCALE:
            base = _SMALL.get(parts[0], 1) or 1
            return "scale", base * _SCALE[parts[1]]
        if parts[0] in _SMALL and parts[1] in _SMALL:
            return "small", _SMALL[parts[0]] + _SMALL[parts[1]]
    word = parts[0]
    if word in _SCALE:
        return "scale", _SCALE[word]
    return "small", _SMALL[word]


def _parse_atoms(atoms: list[str]) -> int:
    total = 0
    current = 0
    for atom in atoms:
        kind, value = _atom_kind(atom)
        if kind == "scale" and value >= 1000 and "-" not in atom:
            total += (current or 1) * value
            current = 0
        elif kind == "scale" and value == 100:
            current = (current or 1) * 100
        elif kind == "scale":
            total += (current or 1) * value
            current = 0
        else:
            current += value
    return total + current


def _read_atom(tokens: list[Tok], i: int) -> tuple[int, str] | None:
    if i >= len(tokens) or tokens[i].kind != "word":
        return None
    word = tokens[i].text.casefold()
    if word not in _NUM_WORDS:
        return None
    if (
        word in _SMALL
        and i + 2 < len(tokens)
        and tokens[i + 1].kind == "sep"
        and tokens[i + 1].text == "-"
        and tokens[i + 1].start == tokens[i].end
        and tokens[i + 2].kind == "word"
        and tokens[i + 2].start == tokens[i + 1].end
        and tokens[i + 2].text.casefold() in _NUM_WORDS
    ):
        return i + 2, tokens[i].text + "-" + tokens[i + 2].text
    return i, tokens[i].text


def _read_gap(tokens: list[Tok], j: int) -> int | None:
    n = len(tokens)
    if j >= n:
        return None
    k = j
    while k < n and _is_ws(tokens[k]):
        k += 1
    if k < n and tokens[k].kind == "sep" and tokens[k].text == ",":
        k += 1
        while k < n and _is_ws(tokens[k]):
            k += 1
        return k if k != j else None
    if (
        k != j
        and k < n
        and tokens[k].kind == "word"
        and tokens[k].text.casefold() == "and"
    ):
        k += 1
        if k >= n or not _is_ws(tokens[k]):
            return None
        while k < n and _is_ws(tokens[k]):
            k += 1
        return k
    if k != j:
        return k
    return None


def _is_hundredths_at(tokens: list[Tok], k: int) -> int | None:
    """End index (exclusive token) of ``NN/100`` starting at ``k``, or None."""
    n = len(tokens)
    if k + 2 >= n or tokens[k].kind != "digit":
        return None
    if not (1 <= len(tokens[k].text) <= 3):
        return None
    slash = k + 1
    if tokens[slash].kind != "sep" or tokens[slash].text != "/":
        return None
    if tokens[slash].start != tokens[k].end:
        return None
    num = slash + 1
    if tokens[num].kind != "digit" or tokens[num].text != "100":
        return None
    if tokens[num].start != tokens[slash].end:
        return None
    return num + 1


def _extend_word_tail(tokens: list[Tok], end_index: int) -> int:
    """Char end, including an ``and fils NN/100`` or ``NN/100`` tail."""
    base = tokens[end_index].end
    n = len(tokens)
    k = end_index + 1
    if k >= n or not _is_ws(tokens[k]):
        return base
    while k < n and _is_ws(tokens[k]):
        k += 1
    if k < n and tokens[k].kind == "sep" and tokens[k].text == ",":
        k += 1
        while k < n and _is_ws(tokens[k]):
            k += 1
    if k < n and tokens[k].kind == "word" and tokens[k].text.casefold() == "and":
        nxt = k + 1
        if nxt < n and _is_ws(tokens[nxt]):
            k = nxt
            while k < n and _is_ws(tokens[k]):
                k += 1
    if k < n and tokens[k].kind == "word" and tokens[k].text.casefold() in _MINOR_WORDS:
        k += 1
        while k < n and _is_ws(tokens[k]):
            k += 1
    hundredths = _is_hundredths_at(tokens, k)
    if hundredths is None:
        return base
    return tokens[hundredths - 1].end


def _word_spans(tokens: list[Tok]) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    i = 0
    n = len(tokens)
    while i < n:
        first = _read_atom(tokens, i)
        if first is None:
            i += 1
            continue
        atoms = [first]
        j = first[0] + 1
        while True:
            gap = _read_gap(tokens, j)
            if gap is None:
                break
            nxt = _read_atom(tokens, gap)
            if nxt is None:
                break
            atoms.append(nxt)
            j = nxt[0] + 1
        texts = [atom[1] for atom in atoms]
        single = texts[0]
        hyphenated = "-" in single and all(
            part in _SMALL for part in single.casefold().split("-")
        )
        scale = single.casefold() in _SCALE and "-" not in single
        if len(atoms) >= 2 or hyphenated or scale:
            end_index = atoms[-1][0]
            end = _extend_word_tail(tokens, end_index)
            spans.append(
                (tokens[i].start, end, magnitude_token(_parse_atoms(texts)))
            )
            i = end_index + 1
            continue
        i += 1
    return spans


def _grouping_joiner(text: str) -> bool:
    return _joiner_kind(text) in {"comma", "space", "dot"}


def _peel_decimal(
    groups: list[str], joiners: list[str]
) -> tuple[list[str], list[str], str | None]:
    """Split a trailing ``.dd`` off a run. Other dots stay grouping separators."""
    if (
        joiners
        and joiners[-1] == "."
        and len(groups[-1]) == 2
        and (len(joiners) == 1 or all(item != "." for item in joiners[:-1]))
    ):
        return groups[:-1], joiners[:-1], groups[-1]
    return groups, joiners, None


def _mixed_thousands(groups: list[str], joiners: list[str]) -> bool:
    """True when a mixed run is still one thousands-grouped amount."""
    if len(joiners) < 2 or len(set(joiners)) == 1:
        return False
    if not all(_grouping_joiner(item) for item in joiners):
        return False
    return _thousands(groups, short_final=False) and not _padded(groups)


def _head_is_amount(groups: list[str], joiners: list[str]) -> bool:
    """True when the groups so far are already one amount, before a minor-unit tail."""
    if not joiners:
        return False
    if len(joiners) == 1 and joiners[0] == "." and len(groups[-1]) == 2:
        return not _padded([groups[0]])
    g, j, _decimal = _peel_decimal(groups, joiners)
    if not j:
        return False
    if len(set(j)) != 1:
        return _mixed_thousands(g, j)
    kind = _joiner_kind(j[0])
    if kind == "dot" and len(j) >= 2 and _thousands(g, False) and not _padded(g):
        return True
    if (
        kind in {"comma", "space"}
        and _thousands(g, False)
        and not _padded(g)
        and not _is_year(g, None)
    ):
        return True
    return False


def _chains(tokens: list[Tok]) -> list[tuple[int, int, list[str], list[str], int]]:
    """Maximal digit runs joined by plausible separators.

    A 1-2 digit group after a finished amount, joined by one space variant, is
    a minor-unit column. It does not extend the run (that would mix the
    column into the amount or split it off).
    """
    found: list[tuple[int, int, list[str], list[str], int]] = []
    i = 0
    n = len(tokens)
    while i < n:
        if tokens[i].kind != "digit":
            i += 1
            continue
        groups = [tokens[i].text]
        joiners: list[str] = []
        first = i
        last = i
        while True:
            m = last + 1
            while m < n and tokens[m].kind == "sep":
                m += 1
            if m >= n or tokens[m].kind != "digit":
                break
            if any(tokens[t].kind != "sep" for t in range(last + 1, m)):
                break
            joiner = "".join(tokens[t].text for t in range(last + 1, m))
            if not _plausible_joiner(joiner):
                break
            new_group = tokens[m].text
            if (
                _is_space_variant(joiner)
                and len(new_group) <= 2
                and _head_is_amount(groups, joiners)
            ):
                break
            joiners.append(joiner)
            groups.append(new_group)
            last = m
        found.append((first, last, groups, joiners, last))
        i = last + 1
    return found


def _padded(groups: list[str]) -> bool:
    head = groups[0]
    if len(head) > 1 and head.startswith("0"):
        return True
    if head == "0" and any(group.strip("0") for group in groups[1:]):
        return True
    return False


def _is_year(groups: list[str], decimal: str | None) -> bool:
    if decimal is not None:
        return False
    compact = "".join(groups)
    return len(compact) == 4 and (
        compact.startswith("19") or compact.startswith("20")
    )


def _valid_day_month(day: int, month: int) -> bool:
    return 1 <= day <= 31 and 1 <= month <= 12


def _is_date_groups(groups: list[str]) -> bool:
    if len(groups) != 3 or not all(part.isdigit() for part in groups):
        return False
    lengths = [len(part) for part in groups]
    nums = [int(part) for part in groups]
    if lengths[2] in {2, 4} and lengths[0] <= 2 and lengths[1] <= 2:
        if _valid_day_month(nums[0], nums[1]) or _valid_day_month(nums[1], nums[0]):
            return True
    if lengths[0] == 4 and lengths[1] <= 2 and lengths[2] <= 2:
        if _valid_day_month(nums[2], nums[1]):
            return True
    return False


def _thousands(groups: list[str], short_final: bool) -> bool:
    if len(groups) < 2:
        return False
    head, *rest = groups
    if not (1 <= len(head) <= 3):
        return False
    if not rest:
        return False
    if short_final and len(rest[-1]) != 3:
        if not (1 <= len(rest[-1]) <= 3):
            return False
        middles = rest[:-1]
        if not middles:
            return False
    else:
        middles = rest
    return all(len(part) == 3 for part in middles)


def _prev_word(tokens: list[Tok], index: int) -> str | None:
    k = index - 1
    while k >= 0 and tokens[k].kind == "sep" and tokens[k].text in set(" \t\u00a0\u2009\u202f.:") | _SPACE:
        k -= 1
    if k >= 0 and tokens[k].kind == "word":
        return tokens[k].text.casefold()
    return None


def _next_word(tokens: list[Tok], index: int) -> str | None:
    n = len(tokens)
    k = index + 1
    while k < n and _is_ws(tokens[k]):
        k += 1
    if k < n and tokens[k].kind == "word":
        return tokens[k].text.casefold()
    return None


def _structural(tokens: list[Tok], first: int) -> bool:
    word = _prev_word(tokens, first)
    return word in _STRUCT


def _phone_cue(text: str, tokens: list[Tok], first: int) -> bool:
    start = tokens[first].start
    pre = text[max(0, start - 6) : start]
    stripped = pre.rstrip(" \t\u00a0\u2009\u202f")
    if stripped.endswith("+") or stripped.endswith("("):
        return True
    if stripped.endswith("00") and (len(stripped) == 2 or not stripped[-3].isdigit()):
        return True
    word = _prev_word(tokens, first)
    return word in _PHONE_WORDS


def _unit_after(tokens: list[Tok], last: int) -> bool:
    n = len(tokens)
    k = last + 1
    if k < n and _is_space_variant(tokens[k].text):
        k += 1
    if k >= n:
        return False
    if tokens[k].text == "%":
        return True
    if tokens[k].kind == "word" and tokens[k].text.casefold() in _UNITS:
        return True
    return False


def _month_after(tokens: list[Tok], last: int) -> bool:
    word = _next_word(tokens, last)
    return word in _MONTHS


def _currency_bare(tokens: list[Tok], first: int, last: int, groups: list[str]) -> bool:
    if first != last or len(groups) != 1:
        return False
    if len(groups[0]) < 2:
        return False
    prev = _prev_word(tokens, first)
    nxt = _next_word(tokens, last)
    return prev in _CURRENCY or nxt in _CURRENCY


def _value(groups: list[str], kind: str, decimal: str | None) -> float:
    digits = "".join(groups)
    if decimal is not None:
        return float(f"{int(digits)}.{decimal}")
    if kind == "dot":
        return float(int(digits))
    return float(int(digits))


def _closes_month_date(tokens: list[Tok], first: int) -> bool:
    """True when this digit is the year glued to ``-Mon`` / ``/Mon``."""
    if first < 2:
        return False
    hyphen = tokens[first - 1]
    month = tokens[first - 2]
    if hyphen.kind != "sep" or hyphen.text not in "-/":
        return False
    if hyphen.end != tokens[first].start or month.end != hyphen.start:
        return False
    if month.kind != "word":
        return False
    return month.text.casefold() in _MONTHS


def _classify_chain(
    text: str,
    tokens: list[Tok],
    first: int,
    last: int,
    groups: list[str],
    joiners: list[str],
) -> str:
    """One label for the whole run: amount, or a reason to leave it."""
    if _closes_month_date(tokens, first):
        return "date"
    if not joiners:
        if _currency_bare(tokens, first, last, groups) and not _padded(groups):
            if _structural(tokens, first) or _unit_after(tokens, last):
                return "leave"
            if _is_year(groups, None):
                return "date"
            if _month_after(tokens, last):
                return "date"
            return "amount"
        return "bare"

    g, j, decimal = _peel_decimal(groups, joiners)
    if (
        decimal is not None
        and len(joiners) == 1
    ):
        if _structural(tokens, first) or _unit_after(tokens, last) or _month_after(tokens, last):
            return "leave"
        if _padded([g[0]]):
            return "leading-zero"
        if _phone_cue(text, tokens, first):
            return "phone"
        return "amount"

    if len(set(j)) != 1:
        if not _mixed_thousands(g, j):
            return "mixed"
        if _is_date_groups(g) or _is_year(g, decimal) or _month_after(tokens, last):
            return "date"
        if _phone_cue(text, tokens, first):
            return "phone"
        if _unit_after(tokens, last) or _structural(tokens, first):
            return "leave"
        return "amount"

    kind = _joiner_kind(j[0])
    if kind == "slash":
        if _is_date_groups(g):
            return "date"
        if (
            len(g) == 2
            and g[1] == "100"
            and _prev_word(tokens, first) in _MINOR_WORDS | _CURRENCY
        ):
            return "fraction"
        return "leave"
    if kind == "hyphen":
        if _is_date_groups(g):
            return "date"
        return "leave"
    if kind == "dot":
        if _is_date_groups(g):
            return "date"
        if len(j) >= 2 and _thousands(g, short_final=False) and not _padded(g):
            if _unit_after(tokens, last) or _structural(tokens, first):
                return "leave"
            return "amount"
        return "leave"
    if kind in {"comma", "space"}:
        if _is_date_groups(g) and kind == "space":
            return "date"
        if _phone_cue(text, tokens, first) and not _thousands(g, decimal is not None):
            return "phone"
        if not _thousands(g, short_final=decimal is not None):
            return "leave"
        if _padded(g):
            return "leading-zero"
        if _is_year(g, decimal):
            return "date"
        if _unit_after(tokens, last) or _structural(tokens, first) or _month_after(tokens, last):
            return "leave"
        if _phone_cue(text, tokens, first):
            return "phone"
        return "amount"
    return "leave"


def _match_minor(tokens: list[Tok], k: int) -> int | None:
    n = len(tokens)
    if k >= n or not _is_space_variant(tokens[k].text):
        return None
    m = k + 1
    while m < n and tokens[m].text == tokens[k].text:
        m += 1
    if m >= n or tokens[m].kind != "digit" or not (1 <= len(tokens[m].text) <= 2):
        return None
    nxt = m + 1
    if nxt < n and tokens[nxt].start == tokens[m].end and tokens[nxt].text in {".", "/", "%"}:
        return None
    if nxt < n and tokens[nxt].kind == "word" and tokens[nxt].start == tokens[m].end:
        return None
    word = _next_word(tokens, m)
    if word in _MONTHS:
        return None
    return tokens[m].end


def _match_hundredths(tokens: list[Tok], k: int) -> int | None:
    n = len(tokens)
    if k >= n:
        return None
    if tokens[k].kind == "sep" and tokens[k].text == ",":
        k += 1
    if k >= n or not _is_ws(tokens[k]):
        return None
    while k < n and _is_ws(tokens[k]):
        k += 1
    if k >= n or tokens[k].kind != "word" or tokens[k].text.casefold() != "and":
        return None
    k += 1
    if k >= n or not _is_ws(tokens[k]):
        return None
    while k < n and _is_ws(tokens[k]):
        k += 1
    if k < n and tokens[k].kind == "word" and tokens[k].text.casefold() in _MINOR_WORDS:
        k += 1
        while k < n and _is_ws(tokens[k]):
            k += 1
    hundredths = _is_hundredths_at(tokens, k)
    if hundredths is None:
        return None
    end = tokens[hundredths - 1].end
    w = hundredths
    while w < n and _is_ws(tokens[w]):
        w += 1
    if w < n and tokens[w].kind == "word" and tokens[w].text.casefold() in _MINOR_WORDS:
        # The minor word must follow the fraction, not a later sentence.
        if tokens[w].start - end <= 2 or _is_ws(tokens[hundredths]):
            gap = tokens[hundredths] if hundredths < n else None
            if gap is not None and _is_ws(gap) and all(
                _is_ws(tokens[t]) for t in range(hundredths, w)
            ):
                end = tokens[w].end
    return end


def _tail_end(tokens: list[Tok], index: int) -> int:
    """Fold a minor-unit column and a following ``NN/100`` tail until fixpoint."""
    end = tokens[index].end
    k = index + 1
    n = len(tokens)
    while k < n:
        if tokens[k].start > end:
            break
        if tokens[k].start < end:
            k += 1
            continue
        nxt = _match_hundredths(tokens, k)
        if nxt is None:
            nxt = _match_minor(tokens, k)
        if nxt is None or nxt <= end:
            break
        end = nxt
        while k < n and tokens[k].start < end:
            k += 1
    return end


def _digit_spans(text: str, tokens: list[Tok]) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    for first, last, groups, joiners, _last in _chains(tokens):
        label = _classify_chain(text, tokens, first, last, groups, joiners)
        if label not in {"amount", "fraction"}:
            continue
        decimal: str | None = None
        g = groups
        j = joiners
        kind = "none"
        if label == "fraction":
            value = int(g[0]) / 100.0
            end = tokens[last].end
        else:
            if (
                j
                and j[-1] == "."
                and len(g[-1]) == 2
                and (len(j) == 1 or all(item != "." for item in j[:-1]))
            ):
                if len(j) == 1:
                    decimal = g[1]
                    g = [g[0]]
                    kind = "dot"
                else:
                    decimal = g[-1]
                    g = g[:-1]
                    j = j[:-1]
                    kind = _joiner_kind(j[0])
            elif j:
                kind = _joiner_kind(j[0])
            value = _value(g, kind, decimal)
            end = _tail_end(tokens, last)
        spans.append((tokens[first].start, end, magnitude_token(value)))
    return spans


def _segment_end(tokens: list[Tok], i: int) -> int | None:
    """Inclusive end of a glued letter/digit/underscore segment."""
    n = len(tokens)
    if i >= n or tokens[i].kind not in _CODE_KINDS:
        return None
    k = i
    while (
        k + 1 < n
        and tokens[k + 1].kind in _CODE_KINDS
        and tokens[k + 1].start == tokens[k].end
    ):
        k += 1
    return k


def _ref_spans(tokens: list[Tok]) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    i = 0
    n = len(tokens)
    while i < n:
        seg = _segment_end(tokens, i)
        if seg is None:
            i += 1
            continue
        parts = ["".join(tokens[t].text for t in range(i, seg + 1))]
        k = seg
        while (
            k + 2 < n
            and tokens[k + 1].kind == "sep"
            and tokens[k + 1].text == "/"
            and tokens[k + 1].start == tokens[k].end
        ):
            nxt = _segment_end(tokens, k + 2)
            if nxt is None or tokens[k + 2].start != tokens[k + 1].end:
                break
            parts.append("".join(tokens[t].text for t in range(k + 2, nxt + 1)))
            k = nxt
        if len(parts) >= 2 and _is_ref(parts):
            spans.append((tokens[i].start, tokens[k].end, _REF))
            i = k + 1
            continue
        i = seg + 1
    return spans


def _is_ref(parts: list[str]) -> bool:
    if all(part.isdigit() for part in parts) and _is_date_groups(parts):
        return False
    if not any(ch.isdigit() for part in parts for ch in part):
        return False
    if len(parts) >= 3:
        return True
    return any(ch.isalpha() or ch == "_" for part in parts for ch in part)


def _amt_tail_spans(tokens: list[Tok]) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    for index, tok in enumerate(tokens):
        if tok.kind != "angle" or not tok.text.startswith("<AMT"):
            continue
        end = _tail_end(tokens, index)
        if end > tok.end:
            spans.append((tok.end, end, ""))
    return spans


def _row_spans(tokens: list[Tok]) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    n = len(tokens)
    for index, tok in enumerate(tokens):
        if tok.kind != "digit" or len(tok.text) != 1:
            continue
        if index + 2 >= n or not _is_space_variant(tokens[index + 1].text):
            continue
        nxt = tokens[index + 2]
        if tokens[index + 1].end != nxt.start:
            continue
        if nxt.kind != "angle" or not nxt.text.startswith("<AMT"):
            continue
        spans.append((tok.start, tok.end, _ROW))
    return spans


def _apply(text: str, spans: list[tuple[int, int, str]]) -> tuple[str, int]:
    if not spans:
        return text, 0
    spans.sort(key=lambda span: (span[0], -(span[1] - span[0])))
    out: list[str] = []
    last = 0
    count = 0
    prev_end = -1
    for start, end, token in spans:
        if start < prev_end or start < last or end <= start:
            continue
        if text[start:end] == token:
            continue
        out.append(text[last:start])
        out.append(token)
        last = end
        prev_end = end
        count += 1
    out.append(text[last:])
    return "".join(out), count


def bucket_amounts(text: str) -> tuple[str, int]:
    """Return text with exact amounts replaced, and how many replacements were made."""
    tokens = _tokenize(text)
    spans = []
    spans.extend(_word_spans(tokens))
    spans.extend(_digit_spans(text, tokens))
    spans.extend(_ref_spans(tokens))
    spans.extend(_amt_tail_spans(tokens))
    spans.extend(_row_spans(tokens))
    return _apply(text, spans)
