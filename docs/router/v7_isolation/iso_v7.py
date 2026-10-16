"""Isolation of every v7 training example (train rows + materialised pairs) against the full text of every registered v9 sealed set."""
import hashlib, json, re, sys
from pathlib import Path
REPO = Path(sys.argv[1]); OUT = Path(sys.argv[2])
sys.path.insert(0, str(REPO / 'scripts')); sys.path.insert(0, str(Path(__file__).parent))
from sealed_sources import sealed_items, check_candidates, sha
from router.content import content_text
from router.v7_pairs import v7_training_examples
check_candidates()
ex, refs = v7_training_examples(REPO)
def grams(t, n=12):
    w = re.sub(r'\W+', ' ', t.lower()).split(); return {' '.join(w[i:i+n]) for i in range(len(w)-n+1)}
base_n = len(ex) - len(refs)
base_texts = {e['messages'][1]['content'] for e in ex[:base_n]}
texts = [e['messages'][1]['content'] for e in ex]
assert set(texts) == base_texts, 'pair texts must be base train texts'
utexts = sorted(set(texts))
tgrams = [grams(t) for t in utexts]
eg = set().union(*tgrams)
ef = {t[:300] for t in utexts}
etsha = {sha(t.encode()) for t in utexts}
efsha = {str(e.get('file_sha256', '')).casefold() for e in ex} - {''}
edoc = {str(e.get('doc_id', '')).casefold() for e in ex} - {''}
eproj = {e.get('project_id') for e in ex} - {None, ''}
res = {'v7_examples': len(ex), 'base_train_rows': base_n, 'pair_rows': len(refs), 'unique_texts': len(utexts), 'pair_texts_are_base_train_texts': True,
       'shingle_note': 'shared 12-word shingles are reported as the largest share of a sealed text\'s shingles found in one training text. They come from the base train rows, which v6 trained on unchanged; pairs reuse those texts, so v7 adds no new text and no new overlap (shingle12_rows_v7 == shingle12_rows_base_train).', 'method': 'file_sha256/doc_id vs sealed file sha; sha256(text); 12-word shingles; first 300 chars; project id. Sealed text = router.content.content_text(candidate) from the box candidate files (sha-checked).', 'sets': {}}
for name, items in sealed_items(REPO, content_text).items():
    ov = {'file_sha256': 0, 'text_sha256': 0, 'first300': 0, 'project': 0}
    sh_rows = 0; sh_max = 0.0
    for it in items:
        t = it['text'] or ''
        s = it['content_sha256'].casefold()
        ov['file_sha256'] += int(s in efsha or s in edoc)
        ov['text_sha256'] += int(sha(t.encode()) in etsha)
        g = grams(t)
        if g & eg:
            sh_rows += 1
            sh_max = max(sh_max, max(len(g & tg) / len(g) for tg in tgrams))
        ov['first300'] += int(bool(t) and t[:300] in ef)
        ov['project'] += int(it['project'] in eproj)
    res['sets'][name] = {'rows': len(items), 'rows_with_text': sum(1 for it in items if it['text']), 'overlaps': ov,
                         'shingle12_rows_v7': sh_rows, 'shingle12_rows_base_train': sh_rows, 'shingle12_max_share': round(sh_max, 4)}
    print(name, len(items), ov, sh_rows, round(sh_max, 4))
(OUT / 'isolation.json').write_text(json.dumps(res, indent=2) + '\n', encoding='utf-8')
