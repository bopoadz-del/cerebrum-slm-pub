"""Name-blind check: blank title-block and party fields by field position.

A field is a party or title-block label followed by a colon. The value after the
colon is blanked up to the next "Label:" field, a line break, or a length cap.
Signature-block fields (Signature, Signed, Sign, By, Name) are blanked the same
way. The labels are layout words. No name, company, project or code is listed
here, and nothing is matched by the value it blanks.

Party value only (check fix, owner order 2026-10-07 22:20 +04). Extracted PDF text is
often one flat line, so the window after a colon can run past the party value into
headings, table headers, line items or body text. Inside that window, by layout only:
- leading form-blank tokens (underscores, dots, dashes, ticks) stay; blanking starts at
  the first token with a letter or digit, so an empty form field blanks nothing;
- a label that ends in a number qualifier (No, Number, Ref, Reference) takes one token,
  the code itself;
- the value ends before a run of two or more ALL-CAPS words once a word with a lowercase
  letter has been seen (the next heading, table header or ALL-CAPS label), and before a
  trailing ALL-CAPS word that starts the next ALL-CAPS label (also right after a form
  blank). A word ending in a comma continues a name and never starts a run;
- the value ends where running prose starts: a sentence start (a capitalised word
  followed by a lowercase word, or a lowercase first word) from which the rest of the
  window is six or more words, most lowercase, with at least four different lowercase
  words (body text after "Dear X:" or "INSTRUCTIONS TO X:").
"""

from __future__ import annotations

import re

MASK = "███"

# Party and title-block field labels (owner order 2026-10-07 17:03 +04), then signature-block fields.
PARTY_FIELDS = (
    "from",
    "to",
    "attn",
    "attention",
    "prepared by",
    "reviewed by",
    "approved by",
    "project",
    "contractor",
    "client",
    "consultant",
    "employer",
    "site",
)
SIGNATURE_FIELDS = ("signature", "signed", "signed by", "sign", "by", "name")

_QUALIFIER = r"(?:'s|’s)?(?:\s+(?:name|no\.?|number|ref\.?|reference|address|title|location|representative|rep\.?)){0,2}"
_FIELD_WORDS = "|".join(sorted((re.escape(f).replace(r"\ ", r"\s+") for f in PARTY_FIELDS + SIGNATURE_FIELDS), key=len, reverse=True))
_FIELD = re.compile(rf"(?<![\w'’])(?P<label>(?:{_FIELD_WORDS}){_QUALIFIER})\s*[:：]", re.IGNORECASE)
_COLON = re.compile(r"[:：]")
_WORD = re.compile(r"[\w'’./&#()-]+")
_MAX_VALUE = 80
_MAX_LABEL_WORDS = 4
_TOKEN = re.compile(r"\S+")
_EDGE = ".,;:()[]{}\"'’"
_PROSE_MIN_WORDS = 6
_PROSE_MIN_DISTINCT_LOWER = 4
_CODE_QUALIFIERS = ("no", "number", "ref", "reference")


def _label_start(text: str, colon: int, floor: int) -> int:
    """Start of the label that ends at ``colon``: the word before it, widened by layout only.

    A lowercase last word widens back to the nearest capitalised word ("Description of change").
    Otherwise the label is the single word before the colon, so a value is never cut short; a
    two-word label such as "Contract No" loses its first word to the blank (stricter, never looser).
    Never more than four words, never before ``floor``.
    """
    words = [m for m in _WORD.finditer(text, floor, colon)]
    if not words:
        return colon
    take = 1
    last = words[-1].group()
    if last[:1].islower():
        while take < min(len(words), _MAX_LABEL_WORDS) and not words[-take].group()[:1].isupper():
            take += 1
    return words[-take].start()


def _value_end(text: str, start: int) -> int:
    limit = min(len(text), start + _MAX_VALUE)
    newline = text.find("\n", start, limit)
    end = newline if newline != -1 else limit
    if newline == -1 and limit < len(text) and not text[limit].isspace():
        space = text.rfind(" ", start, limit)
        if space > start:
            end = space
    colon = _COLON.search(text, start, end)
    if colon is not None:
        end = max(start, _label_start(text, colon.start(), start))
    return end


def blank_fields(text: str) -> tuple[str, int]:
    """Return the text with field values blanked, and how many values were blanked."""
    out: list[str] = []
    cursor = 0
    blanked = 0
    for match in _FIELD.finditer(text):
        if match.start() < cursor:
            continue
        start = match.end()
        end = _value_end(text, start)
        first, last = _party_span(text, start, end, match.group("label"))
        value = text[first:last]
        out.append(text[cursor:first])
        if value.strip() and value.strip() != MASK:
            lead = value[: len(value) - len(value.lstrip())]
            trail = value[len(value.rstrip()):]
            out.append(f"{lead or ' '}{MASK}{trail or ' '}")
            blanked += 1
        else:
            out.append(value)
        out.append(text[last:end])
        cursor = end
    out.append(text[cursor:])
    return "".join(out), blanked


def _is_caps_word(token: str) -> bool:
    core = token.strip(_EDGE)
    return len(core) >= 2 and core.isalpha() and core.isupper()


def _has_content(token: str) -> bool:
    return re.search(r"[^\W_]", token) is not None


def _is_prose(tokens: list[str]) -> bool:
    words = [w.strip(_EDGE) for w in tokens]
    words = [w for w in words if w.isalpha()]
    lower = [w for w in words if w[:1].islower()]
    return len(words) >= _PROSE_MIN_WORDS and len(lower) * 2 > len(words) and len(set(lower)) >= _PROSE_MIN_DISTINCT_LOWER


def _starts_sentence(words: list[str], index: int) -> bool:
    word = words[index].strip(_EDGE)
    following = words[index + 1].strip(_EDGE) if index + 1 < len(words) else ""
    if index == 0 and word[:1].islower():
        return True
    return word[:1].isupper() and following[:1].islower()


def _party_span(text: str, start: int, end: int, label: str = "") -> tuple[int, int]:
    """The party value inside the window ``text[start:end]`` (see module docstring). Empty when none."""
    tokens = list(_TOKEN.finditer(text, start, end))
    after_blank = bool(tokens) and not _has_content(tokens[0].group())
    while tokens and not _has_content(tokens[0].group()):
        tokens.pop(0)  # form blank
    if not tokens:
        return end, end
    if label.split() and label.split()[-1].lower().strip(_EDGE) in _CODE_QUALIFIERS:
        return tokens[0].start(), tokens[0].end()
    words = [token.group() for token in tokens]
    stop = len(tokens)
    for index in range(len(words)):
        if _starts_sentence(words, index) and _is_prose(words[index:]):
            stop = index
            break
    started = False
    for index in range(stop):
        word = words[index]
        if not _is_caps_word(word):
            started = started or any(ch.islower() for ch in word)
            continue
        if started and not word.endswith(",") and index + 1 < stop and _is_caps_word(words[index + 1]):
            stop = index
            break
    else:
        following = _TOKEN.search(text, end)
        if (
            (started or after_blank)
            and stop == len(tokens)
            and _is_caps_word(words[-1])
            and following is not None
            and _is_caps_word(following.group().rstrip(":："))
        ):
            stop -= 1  # the trailing ALL-CAPS word starts the next ALL-CAPS label
    if stop == 0:
        return tokens[0].start(), tokens[0].start()
    return tokens[0].start(), tokens[stop - 1].end()


def flips(gold: list[str], before: list[str], after: list[str], labels: list[str]) -> dict:
    """Per-class prediction flips between the normal and the name-blind run, keyed by gold class."""
    if not (len(gold) == len(before) == len(after)):
        raise ValueError("gold, before and after must align")
    table = {label: {"n": 0, "flips": 0, "correct_before": 0, "correct_after": 0} for label in labels}
    for g, b, a in zip(gold, before, after):
        row = table.setdefault(g, {"n": 0, "flips": 0, "correct_before": 0, "correct_after": 0})
        row["n"] += 1
        row["flips"] += int(b != a)
        row["correct_before"] += int(b == g)
        row["correct_after"] += int(a == g)
    return table


def name_blind_gate(table: dict, min_per_label: int) -> dict:
    """FAIL when any passable class (n >= min_per_label) has a flip."""
    failing = sorted(label for label, row in table.items() if row["n"] >= min_per_label and row["flips"] > 0)
    return {"passed": not failing, "failing_passable_classes": failing, "min_per_label": min_per_label}
