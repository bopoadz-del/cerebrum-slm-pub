"""Name-blind CHECK FIX re-run for router v5 epoch 1 (owner order 2026-10-07 22:20 +04).

Not a re-score. The unblanked side is the stored sealed_predictions.jsonl from the one sealed score
(docs/router/runs/v5/sealed, PR #47); it is read, never re-sampled. Only the blanked side is
re-classified, with the fixed scripts/router/name_blind.py. Same adapter, prompt, sampling, parser.
Refuses a second run. Never prints secrets.
Usage: nameblind_fix_v5e1.py --repo <main 64b1df9 + fix> --tree <tree json of 64b1df9>
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

OUT = Path('/workspace/router-relabel-v8/score_v5/out_nameblind_fix')
STORED = Path('/workspace/router-relabel-v8/score_v5/out')
FIXED_FILES = {'scripts/router/name_blind.py', 'tests/router/test_name_blind.py'}


def blob(b):
    return hashlib.sha1(b'blob %d\0' % len(b) + b).hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--repo', type=Path, required=True); ap.add_argument('--tree', type=Path, required=True)
    a = ap.parse_args(); REPO = a.repo; tree_doc = json.load(open(a.tree))
    OUT.mkdir(parents=True, exist_ok=True)
    log = open(OUT / 'nameblind_fix.log', 'a', encoding='utf-8')
    def say(*x):
        s = ' '.join(str(i) for i in x); print(s, flush=True); log.write(s + '\n'); log.flush()
    marker = OUT / 'RUN_STARTED_nameblind_fix.json'
    if marker.exists():
        say('WALL: already started; no re-runs.'); return 3
    if not (OUT / 'PRE_RUN_NOTE.txt').exists():
        say('WALL: no pre-run note'); return 3
    tree = {e['path']: e['sha'] for e in tree_doc['tree'] if e['type'] == 'blob'}
    bad = [p for p, s in tree.items() if p not in FIXED_FILES and (not (REPO / p).is_file() or blob((REPO / p).read_bytes()) != s)]
    if bad:
        say(f'WALL: repo differs from {tree_doc["sha"][:7]} outside the fix at {len(bad)} files'); return 3
    sys.path.insert(0, str(REPO / 'scripts')); sys.path.insert(0, '/workspace/router-train/driver'); sys.path.insert(0, '/workspace/router-relabel-v8/score_v5')
    import score_v5e1_v9 as base
    from router.labels import load_labels, load_sealed_config
    from router.teacher_label import prompt_for
    from router.content import content_text
    import router.train_router as tr
    import router.name_blind as nbm
    from router.sealed_gate import MIN_SEALED_PER_LABEL
    from common import load_env_file
    cfg = load_sealed_config(REPO / 'configs/router/sealed_eval.json'); v9 = cfg['v9_sealed']
    paths = {item['name']: item['sealed_labels'] for item in v9['sets']}
    for rel in paths.values():
        p = REPO / rel
        if hashlib.sha256(p.read_bytes()).hexdigest() != (p.parent / 'sealed_labels.jsonl.sha256').read_text().split()[0]:
            say(f'WALL: seal mismatch {rel}'); return 3
    labels = load_labels(REPO / v9['labels_file']); system = prompt_for(labels)
    started_score = json.loads((STORED / 'RUN_STARTED_score.json').read_text())
    if hashlib.sha256(system.encode()).hexdigest() != started_score['system_sha256']:
        say('WALL: prompt differs from the stored sealed score'); return 3
    ckpt = started_score['checkpoint']
    stored = {(r['set'], r['sheet_id']): r for r in base.jl(STORED / 'sealed_predictions.jsonl')}
    old_blind = {(r['set'], r['sheet_id']): r for r in base.jl(STORED / 'name_blind_predictions.jsonl')}
    lora = json.loads((REPO / 'configs/lora/qwen3-4b-router-r16.json').read_text())
    items = []
    for name, (cands, key, shakey) in base.SETS.items():
        cmap = {r[key]: r for f in cands for r in base.jl(f)}
        for g in base.jl(REPO / paths[name]):
            c = cmap[g['sheet_id']]
            if c[shakey] != g['file_sha256']:
                say(f"WALL: content sha mismatch {g['sheet_id']}"); return 3
            text, nb = nbm.blank_fields(content_text(c))
            items.append({'set': name, 'sheet_id': g['sheet_id'], 'gold': g['label'], 'text': text, 'blanked_fields': nb})
    blanker_sha = hashlib.sha256((REPO / 'scripts/router/name_blind.py').read_bytes()).hexdigest()
    marker.write_text(json.dumps({'started': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'kind': 'name-blind check fix, blanked side only', 'checkpoint': ckpt,
                                  'base_commit': tree_doc['sha'], 'blanker_sha256': blanker_sha, 'system_sha256': started_score['system_sha256'],
                                  'unblanked_side': 'stored docs/router/runs/v5/sealed/sealed_predictions.jsonl (not re-sampled)', 'n': len(items)}) + '\n')
    names = load_env_file(Path('/workspace/slm-secrets/tinker.env'))
    say(f'env names loaded: {len(names)}; checkpoint={ckpt}; blanker={blanker_sha[:12]}')
    import tinker
    service = tinker.ServiceClient(); client = service.create_sampling_client(model_path=ckpt); tok = client.get_tokenizer()
    params = tinker.types.SamplingParams(max_tokens=32, temperature=0, top_p=1, seed=int(lora['seed']))
    def one(it):
        msgs = [{'role': 'system', 'content': system}, {'role': 'user', 'content': it['text']}]
        for att in range(5):
            try:
                t = tr._complete(client, tok, tinker.types, msgs, params); return t, tr._as_label(t, labels)
            except Exception:
                if att == 4: raise
                time.sleep(2 * (att + 1))
    with ThreadPoolExecutor(max_workers=16) as pool:
        out = list(pool.map(one, items))
    rows = []
    for it, (raw, p) in zip(items, out):
        k = (it['set'], it['sheet_id'])
        rows.append({'set': it['set'], 'sheet_id': it['sheet_id'], 'gold': it['gold'], 'pred': p, 'raw': raw, 'blanked_fields': it['blanked_fields'],
                     'unblanked_pred_stored': stored[k]['pred'], 'old_blanker_pred': old_blind[k]['pred'], 'old_blanker_fields': old_blind[k]['blanked_fields']})
    (OUT / 'name_blind_fix_predictions.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    order = list(base.SETS); views = {k: [r for r in rows if r['set'] == k] for k in order}; views['combined'] = rows
    res = {'kind': 'name-blind CHECK FIX (not a re-score)', 'checkpoint': ckpt, 'base_commit': tree_doc['sha'], 'blanker': 'scripts/router/name_blind.py (fixed)', 'blanker_sha256': blanker_sha,
           'old_blanker_sha256': 'ddab8598152fcb17f2da072b756954f154a24bd60fb578d9a1569e9b47bdeabf', 'unblanked_side': 'stored sealed_predictions.jsonl from the one sealed score', 'sets': {}}
    for k, pr in views.items():
        gold = [r['gold'] for r in pr]; bef = [r['unblanked_pred_stored'] for r in pr]; aft = [r['pred'] for r in pr]
        table = nbm.flips(gold, bef, aft, labels); gate = nbm.name_blind_gate(table, MIN_SEALED_PER_LABEL)
        unblanked_rows = [r for r in pr if r['blanked_fields'] == 0]
        res['sets'][k] = {'n': len(gold), 'rows_with_blanked_fields': sum(r['blanked_fields'] > 0 for r in pr), 'fields_blanked': sum(r['blanked_fields'] for r in pr),
                          'flips_total': sum(b != x for b, x in zip(bef, aft)),
                          'flips_passable': sum(row['flips'] for row in table.values() if row['n'] >= MIN_SEALED_PER_LABEL),
                          'flips_not_passable': sum(row['flips'] for row in table.values() if row['n'] < MIN_SEALED_PER_LABEL),
                          'flips_on_rows_with_nothing_blanked': sum(r['pred'] != r['unblanked_pred_stored'] for r in unblanked_rows),
                          'per_class': table, 'gate': gate,
                          'accuracy_before': sum(x == y for x, y in zip(gold, bef)) / len(gold), 'accuracy_after': sum(x == y for x, y in zip(gold, aft)) / len(gold)}
        s = res['sets'][k]
        say(f"{k} name-blind (fixed): flips={s['flips_total']} (passable {s['flips_passable']}, not passable {s['flips_not_passable']}; on unblanked rows {s['flips_on_rows_with_nothing_blanked']}) gate={gate}")
    (OUT / 'name_blind_fix.json').write_text(json.dumps(res, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    try: service.close('success').result()
    except Exception: pass
    say('DONE nameblind_fix')

if __name__ == '__main__':
    sys.exit(main())
