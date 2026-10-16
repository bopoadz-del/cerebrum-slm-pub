#!/usr/bin/env python3
"""Router training-pool practice-region share (read-only analysis).

Classifies every pool row by its CONTENT (excerpt text only) into
  a  = US-form paperwork / US practice
  b  = FIDIC-style or Gulf / UK / Commonwealth practice
  c  = indeterminate
Never reads filename, folder, project_id, source site, form number field or label.
Does not touch labels, sealed sets or training data; writes a result JSON only.

Method: each signal family is a regex; a family counts once per document
(presence, not frequency).  Weights: strong=3 (named contract regime / form
family, Gulf/UK/US place or authority, currency), medium=2 (contract-role or
instrument terminology characteristic of one practice), weak=1 (spelling,
unit system).  score_a / score_b are sums over present families.
  a if score_a >= 3 and score_a >= score_b + 2
  b if score_b >= 3 and score_b >= score_a + 2
  else c
Threshold 3 means one strong signal, or a medium + weak, or three weak
families; units or spelling alone (<=2) never classify a document.
"""
import argparse, json, re, collections, hashlib

I = re.I
US = {  # family: (weight, pattern, flags)
    "us_form_family": (3, r"\bAIA\b|\bG70[0-9]\b|\bA20[0-9]\b|ConsensusDocs|EJCDC", 0),
    "us_dot_federal": (3, r"Department of Transportation|\b[A-Z]{2,5}DOT\b|\bFHWA\b|Caltrans|\bUSACE\b|NAVFAC|\bOSHA\b|\bADA\b", 0),
    "us_state_zip":  (3, r"\b(AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY)\.?,?\s+\d{5}(-\d{4})?\b", 0),
    "us_state_name": (2, r"\b(California|Texas|Florida|New York|Illinois|Ohio|Pennsylvania|Washington State|Massachusetts|Virginia|Georgia|Arizona|Colorado|Oregon|Michigan|Minnesota|Wisconsin|North Carolina|New Jersey|Maryland|Indiana|Missouri|Tennessee|Kentucky|Louisiana|Alabama)\b", 0),
    "us_change_order_owner_architect": (2, r"change order", I),  # only counts with US role terms, see below
    "us_practice_terms": (2, r"punch ?list|retainage|schedule of values|pay app(lication)?\b|notice to proceed|general contractor|\bPCO\b|proposal request|county commission|city council|board of supervisors", I),
    "us_municipal": (1, r"\bcounty\b|\bcity of\b|\bborough of\b|\btownship\b", I),
    "us_imperial": (1, r"\b\d+(\.\d+)?\s?(psi|ksi|lbs?|cubic yards?|sq\.? ?ft|feet|ft|inch(es)?|gallons?|gpm|cfm)\b|\d+'\s?-\s?\d+\"|°F\b", I),
    "us_dollar": (1, r"\$\s?\d", 0),
    "us_spelling": (1, r"\b(color|program(?!me)|organization|center|license|labor|favor|meter|defense)\b", I),
}
B = {
    "b_contract_regime": (3, r"FIDIC|\bNEC ?[34]?\b(?! code)|\bJCT\b|\bRIBA\b|AS/NZS|AS ?4000|AS ?2124|Red Book|Yellow Book|Silver Book", 0),
    "b_gulf_place_authority": (3, r"\b(Diriyah|King Khalid Road|Saudi Building Code|Dubai|Abu Dhabi|Sharjah|Ajman|Ras Al Khaimah|Fujairah|Riyadh|Jeddah|Dammam|Khobar|NEOM|Saudi|KSA|U\.?A\.?E\.?|Oman|Muscat|Qatar|Doha|Kuwait|Bahrain|Manama|Emirates|Aramco|DEWA|ADDC|SEWA|Kahramaa|Ashghal|Makkah|Madinah|Mecca|Lusail|Civil Defen[cs]e)\b", I),
    "b_uk_au_nz_place": (3, r"\b(London|Manchester|Birmingham|Glasgow|Edinburgh|Scotland|Wales|England|United Kingdom|Australia|New Zealand|NSW|Queensland|Auckland|Wellington|Sydney|Melbourne|Brisbane|Perth WA)\b", 0),
    "b_currency": (3, r"\b(SAR|AED|OMR|QAR|KWD|BHD|GBP|AUD|NZD|Dhs|Dh|SR|RO)\s?[\d,]|£\s?\d|\b(riyals?|dirhams?|rials?)\b", I),
    "b_standards": (2, r"\bBS ?EN\b|\bBS ?\d{3,5}\b|\bCIBSE\b|\bNHBC\b|\bSMM7\b|\bNRM ?[12]?\b|\bPOMI\b|\bCESMM", 0),
    "b_roles": (2, r"\bthe Engineer\b|\bEngineer's (Representative|Instruction)\b|\bEmployer\b|\bEmployer's\b|Quantity Surveyor|\bQS\b|Contract Administrator|Superintendent's", 0),
    "b_instruments": (2, r"variation order|\bVO\b|\bIPC\b|interim payment certificate|sub-clause|extension of time|\bEOT\b|letter of acceptance|taking-over certificate|defects notification|performance certificate|engineer's instruction|architect's instruction|progress claim|payment claim", I),
    "b_spelling": (1, r"\b(colour|programme|organisation|centre|licence|labour|favour|metres?|tonnes?|defence|authorised|realise|tender(er)?s?)\b", I),
    "b_metric": (1, r"\b\d+(\.\d+)?\s?(mm|cm|m2|m3|m²|m³|kN|MPa|kg|km)\b", 0),
}
US_ROLE = re.compile(r"\b(Owner|Architect)\b")

def compile_(d):
    return {k: (w, re.compile(p, f)) for k, (w, p, f) in d.items()}
USC, BC = compile_(US), compile_(B)

SOURCE_TAG = re.compile(r"\[source:[^\]]*\]", I)

def strip_metadata(text):
    """Remove embedded file-path tags so only document content is scored."""
    return SOURCE_TAG.sub(" ", text)

def classify(text):
    text = strip_metadata(text)
    fa = [k for k, (w, rx) in USC.items() if rx.search(text)]
    if "us_change_order_owner_architect" in fa and not US_ROLE.search(text):
        fa.remove("us_change_order_owner_architect")
    fb = [k for k, (w, rx) in BC.items() if rx.search(text)]
    sa = sum(US[k][0] for k in fa); sb = sum(B[k][0] for k in fb)
    if sa >= 3 and sa >= sb + 2: cls = "a"
    elif sb >= 3 and sb >= sa + 2: cls = "b"
    else: cls = "c"
    return cls, sa, sb, fa, fb

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pool"); ap.add_argument("--out", required=True)
    ap.add_argument("--text-field", default="excerpt")
    ap.add_argument("--spot", default=None, help="write spot-check sample jsonl")
    ap.add_argument("--spot-n", type=int, default=20)
    a = ap.parse_args()
    raw = open(a.pool, "rb").read()
    rows = [json.loads(l) for l in raw.decode().splitlines() if l.strip()]
    cnt = collections.Counter(); by_label = {}; fam = collections.Counter(); per = []
    for r in rows:
        cls, sa, sb, fa, fb = classify(r.get(a.text_field) or "")
        cnt[cls] += 1; fam.update(fa); fam.update(fb)
        per.append((r.get("doc_id"), cls, sa, sb, fa, fb))
        by_label.setdefault(str(r.get("label")), collections.Counter())[cls] += 1  # reporting only, never an input
    n = len(rows)
    res = {
        "pool_file": a.pool, "pool_sha256": hashlib.sha256(raw).hexdigest(), "rows": n,
        "text_field": a.text_field,
        "counts": {k: cnt[k] for k in "abc"},
        "shares_pct": {k: round(100 * cnt[k] / n, 2) for k in "abc"},
        "b_at_least_one_third": cnt["b"] * 3 >= n,
        "by_existing_teacher_label_reporting_only": {k: {c: v[c] for c in "abc"} for k, v in sorted(by_label.items(), key=lambda x: -sum(x[1].values()))},
        "family_hits": dict(sorted(fam.items(), key=lambda x: -x[1])),
        "rule": "a if score_a>=3 and score_a>=score_b+2; b if score_b>=3 and score_b>=score_a+2; else c (family presence, weights 3/2/1)",
        "inputs_used": "excerpt text only; no filename/folder/project/source/label fields",
    }
    json.dump(res, open(a.out, "w"), indent=2)
    if a.spot:
        import random
        rnd = random.Random(20261007)
        with open(a.spot, "w") as f:
            for cls in "abc":
                pool = [p for p in per if p[1] == cls]
                for p in rnd.sample(pool, min(a.spot_n, len(pool))):
                    f.write(json.dumps({"doc_id": p[0], "cls": cls, "sa": p[2], "sb": p[3], "fa": p[4], "fb": p[5]}) + "\n")
    print(json.dumps({k: res[k] for k in ("rows", "counts", "shares_pct", "b_at_least_one_third")}))
    print(json.dumps(res["family_hits"]))

if __name__ == "__main__":
    main()
