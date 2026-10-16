"""Box-side sealed text sources (paths on the scoring box; no text goes into the repo). Shared by isolation and scoring."""
import hashlib, json
from pathlib import Path
SETS = {
    'v5_retick': ([Path('/workspace/router-sealed/seal_inputs/candidates_v5.jsonl'), Path('/workspace/router-sealed/seal_inputs/candidates_addendum.jsonl')], 'sheet_id', 'content_sha256'),
    'v8_retick': ([Path('/workspace/router-sealed/v8/candidates_v8.jsonl'), Path('/workspace/router-sealed/v8/candidates_v8_addendum.jsonl')], 'id', 'sha256'),
    'cr_addendum': ([Path('/workspace/router-cr/cand/candidates_sealed.jsonl')], 'id', 'sha256'),
    'thin_c_addendum': ([Path('/workspace/router-rows-v9b/cand_sealed_c/candidates_sealed.jsonl')], 'id', 'sha256'),
    'thin_d_addendum': ([Path('/workspace/router-relabel-v8/score_v6/candidates_thin_d.jsonl')], 'id', 'sha256'),
    'web_f_addendum': ([Path('/workspace/router-web-f/score/candidates_web_f.jsonl')], 'id', 'sha256'),
}
EXPECTED_CAND_SHA = {
    '/workspace/router-sealed/seal_inputs/candidates_v5.jsonl': '8ed36aefc26e8817441a8e0c2e683ce59eb50048416d80cc8a85204388cc44dd',
    '/workspace/router-sealed/seal_inputs/candidates_addendum.jsonl': '639eada1dc06a981204af082d82be11610229f065cbb01722290c5d35e9fc6a5',
    '/workspace/router-sealed/v8/candidates_v8.jsonl': 'cda565f8521f004909a4ab2ed1f31bda92f4090184b06ce6c36962fedeb8d85b',
    '/workspace/router-sealed/v8/candidates_v8_addendum.jsonl': 'b441d0378b34c12cae44acafd355beaab4fe0849cb9c8548ffc0ab504a3feba0',
    '/workspace/router-cr/cand/candidates_sealed.jsonl': 'b993c0a42aa5200c9d3c62ce0fc4acbc795b4b5a29613b5bc9f7787a4d8eed95',
    '/workspace/router-rows-v9b/cand_sealed_c/candidates_sealed.jsonl': '0d789148ff1c1424b0a2380f315465dcac373e138956d5f3515ca2156553c0f6',
    '/workspace/router-relabel-v8/score_v6/candidates_thin_d.jsonl': '3eb18720cd2c14583e02de7e043a9f98f7c2cbaf8b7250e05f59cc16db9917d1',
    '/workspace/router-web-f/score/candidates_web_f.jsonl': 'cebd835ef2cbcfac825bba5b521aeef4d10b6a5ff97860e62fbc894d41ca4b3a',
}
def jl(p): return [json.loads(l) for l in Path(p).read_text(encoding='utf-8').splitlines() if l.strip()]
def sha(b): return hashlib.sha256(b).hexdigest()
def check_candidates():
    bad = [p for p, s in EXPECTED_CAND_SHA.items() if sha(Path(p).read_bytes()) != s]
    if bad: raise SystemExit(f'WALL: candidate file sha mismatch {bad}')
def sealed_items(repo, content_text):
    """{set: [ {sheet_id, content_sha256, gold, text, project} ]} in sealed_eval.json order; refuses a sha mismatch."""
    import sys
    from router.labels import load_sealed_config
    cfg = load_sealed_config(repo / 'configs/router/sealed_eval.json')['v9_sealed']
    out = {}
    for s in cfg['sets']:
        cands, key, shakey = SETS[s['name']]
        cmap = {}
        for f in cands:
            for r in jl(f): cmap[r[key]] = r
        items = []
        for g in jl(repo / s['sealed_labels']):
            c = cmap[g['sheet_id']]
            if c[shakey] != g['file_sha256']: raise SystemExit(f"WALL: content sha mismatch {g['sheet_id']}")
            items.append({'sheet_id': g['sheet_id'], 'content_sha256': g['file_sha256'], 'gold': g['label'], 'text': content_text(c), 'project': g.get('project')})
        out[s['name']] = items
    return out
