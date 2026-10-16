"""ONE-SHOT sealed score of router v3 epoch 1 on the v9 re-ticked v5 and v8 sets, then the name-blind check.

phase "score": normal text, scored once (marker refuses a second run).
phase "blind": the same rows with title-block/party fields blanked by router.name_blind, same settings.
Never prints secrets.
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path('/workspace/router-relabel-v8')
REPO = ROOT / 'main'                     # byte-exact main 868e5af (git LF blobs verified)
BLIND_SRC = ROOT / 'pr5/repo/scripts/router/name_blind.py'
OUT = ROOT / 'score_v9/out'
CKPT = 'tinker://34e98763-665a-534a-906f-651cc7e8c6f3:train:0/sampler_weights/router-v3-qwen3.5-4b-lora-r16-epoch1'
MAIN_SHA = '868e5aff8dbee6e7d0ef65b3f7f598a01ce248a4'
SETS = {
    'v5': ('data/eval/router_unseen/v9_retick/v5', [Path('/workspace/router-sealed/seal_inputs/candidates_v5.jsonl'), Path('/workspace/router-sealed/seal_inputs/candidates_addendum.jsonl')], 'sheet_id', 'content_sha256'),
    'v8': ('data/eval/router_unseen/v9_retick/v8', [Path('/workspace/router-sealed/v8/candidates_v8.jsonl'), Path('/workspace/router-sealed/v8/candidates_v8_addendum.jsonl')], 'id', 'sha256'),
}


def jl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding='utf-8').splitlines() if l.strip()]


def sha(b):
    return hashlib.sha256(b).hexdigest()


def blob(b):
    return hashlib.sha1(b'blob %d\0' % len(b) + b).hexdigest()


def verify(say):
    tree = {e['path']: e['sha'] for e in json.load(open(ROOT / 'tree_868e5af.json'))['tree'] if e['type'] == 'blob'}
    bad = [p for p, s in tree.items() if blob((REPO / p).read_bytes()) != s]
    if bad:
        raise SystemExit(f'WALL: local main differs from 868e5af at {len(bad)} files')
    for d in ('data/eval/router_unseen', 'data/eval/router_unseen/v8', 'data/eval/router_unseen/v9_retick/v5', 'data/eval/router_unseen/v9_retick/v8'):
        for line in (REPO / d / 'SHA256SUMS').read_text(encoding='utf-8').splitlines():
            h, n = line.split('  ')
            if sha((REPO / d / n).read_bytes()) != h:
                raise SystemExit(f'WALL: seal mismatch {d}/{n}')
    side = (REPO / 'data/eval/router_unseen/v9_retick/label_changes.json.sha256').read_text().split()[0]
    if sha((REPO / 'data/eval/router_unseen/v9_retick/label_changes.json').read_bytes()) != side:
        raise SystemExit('WALL: label_changes seal mismatch')
    say(f'seals verified on main {MAIN_SHA} ({len(tree)} blobs)')


def load_rows(content_text, blank=None):
    rows = {}
    for name, (d, cands, key, shakey) in SETS.items():
        gold = jl(REPO / d / 'sealed_labels.jsonl')
        cmap = {}
        for f in cands:
            for r in jl(f):
                cmap[r[key]] = r
        items = []
        for g in gold:
            c = cmap[g['sheet_id']]
            if c[shakey] != g['file_sha256']:
                raise SystemExit(f"WALL: content sha mismatch {g['sheet_id']}")
            text = content_text(c)
            nb = 0
            if blank is not None:
                text, nb = blank(text)
            items.append({'sheet_id': g['sheet_id'], 'content_sha256': g['file_sha256'], 'gold': g['label'], 'text': text, 'blanked_fields': nb})
        rows[name] = items
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=['score', 'blind'])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    log = open(OUT / 'score.log', 'a', encoding='utf-8')

    def say(*x):
        s = ' '.join(str(i) for i in x)
        print(s, flush=True)
        log.write(s + '\n')
        log.flush()

    marker = OUT / f'RUN_STARTED_{a.phase}.json'
    if marker.exists():
        say(f'WALL: phase {a.phase} already started; no re-runs.')
        return 3
    if a.phase == 'blind' and not (OUT / 'sealed_scores.json').exists():
        say('WALL: blind phase needs the recorded score first')
        return 3
    if not (OUT / 'PRE_RUN_NOTE.txt').exists():
        say('WALL: no pre-run note')
        return 3
    sys.path.insert(0, str(REPO / 'scripts'))
    sys.path.insert(0, '/workspace/router-train/driver')
    from router.labels import load_labels, load_sealed_config
    from router.teacher_label import prompt_for
    from router.content import content_text
    import router.train_router as tr
    from router.sealed_gate import evaluate_sealed_gate_v8, macro_recall, MIN_SEALED_PER_LABEL
    from common import load_env_file
    verify(say)
    labels = load_labels(REPO / 'configs/router/labels.json')
    assert len(labels) == 22 and 'change request' in labels
    cfg = load_sealed_config(REPO / 'configs/router/sealed_eval.json')
    mcf = bool(cfg['sealed_gate'].get('majority_class_fail', cfg['majority_class_fail']))
    lora = json.loads((REPO / 'configs/lora/qwen3-4b-router-r16.json').read_text(encoding='utf-8'))
    system = prompt_for(labels)
    blank = None
    if a.phase == 'blind':
        import importlib.util
        spec = importlib.util.spec_from_file_location('name_blind', BLIND_SRC)
        nbm = importlib.util.module_from_spec(spec); spec.loader.exec_module(nbm)
        blank = nbm.blank_fields
    rows = load_rows(content_text, blank)
    marker.write_text(json.dumps({'started': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'checkpoint': CKPT, 'n': {k: len(v) for k, v in rows.items()},
                                  'system_sha256': sha(system.encode()), 'name_blind_sha256': sha(BLIND_SRC.read_bytes())}) + '\n')
    names = load_env_file(Path('/workspace/slm-secrets/tinker.env'))
    say(f'env names loaded: {len(names)}; checkpoint={CKPT}; phase={a.phase}')
    import tinker
    service = tinker.ServiceClient()
    client = service.create_sampling_client(model_path=CKPT)
    tok = client.get_tokenizer()
    params = tinker.types.SamplingParams(max_tokens=32, temperature=0, top_p=1, seed=int(lora['seed']))

    def one(item):
        msgs = [{'role': 'system', 'content': system}, {'role': 'user', 'content': item['text']}]
        for att in range(5):
            try:
                t = tr._complete(client, tok, tinker.types, msgs, params)
                return t, tr._as_label(t, labels)
            except Exception:
                if att == 4:
                    raise
                time.sleep(2 * (att + 1))
    allrows = [(k, it) for k, v in rows.items() for it in v]
    with ThreadPoolExecutor(max_workers=16) as pool:
        out = list(pool.map(lambda kv: one(kv[1]), allrows))
    preds = {}
    for (k, it), (raw, p) in zip(allrows, out):
        preds.setdefault(k, []).append({'set': k, 'sheet_id': it['sheet_id'], 'content_sha256': it['content_sha256'], 'gold': it['gold'], 'pred': p, 'raw': raw, 'blanked_fields': it['blanked_fields']})
    tag = 'sealed' if a.phase == 'score' else 'name_blind'
    (OUT / f'{tag}_predictions.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for k in ('v5', 'v8') for r in preds[k]), encoding='utf-8')
    if a.phase == 'score':
        result = {'repo_commit': MAIN_SHA, 'checkpoint': CKPT, 'labels': 'configs/router/labels.json (v9, 22)', 'system': 'router.teacher_label.prompt_for(labels)',
                  'parser': 'router.train_router._as_label -> parse_router_output (v9 exact aliases)', 'sampling': {'max_tokens': 32, 'temperature': 0, 'top_p': 1, 'seed': int(lora['seed']), 'enable_thinking': False},
                  'gate': 'evaluate_sealed_gate_v8 (gate v2 + MIN_SEALED_PER_LABEL=%d), unchanged' % MIN_SEALED_PER_LABEL, 'majority_class_fail': mcf, 'sets': {}}
        for k in ('v5', 'v8'):
            gold = [r['gold'] for r in preds[k]]; pred = [r['pred'] for r in preds[k]]
            g = evaluate_sealed_gate_v8(gold, pred, labels, majority_class_fail=mcf)
            consts = {lab: macro_recall(gold, [lab] * len(gold), labels) for lab in labels}
            result['sets'][k] = {'n': len(gold), 'accuracy': sum(x == y for x, y in zip(gold, pred)) / len(gold), 'macro_recall': g['gate_v2']['macro_recall'],
                                 'gate_v2_passed': bool(g['gate_v2']['passed']), 'gate_v2': g['gate_v2'], 'router_passed': bool(g['router_passed']), 'per_label': g['per_label'],
                                 'pred_counts': dict(Counter(pred)), 'unsure': sum(p == 'unsure' for p in pred), 'constant_macro': consts}
            say(f"{k}: n={len(gold)} macro={result['sets'][k]['macro_recall']} acc={result['sets'][k]['accuracy']:.4f} gate_v2={result['sets'][k]['gate_v2_passed']} router_passed={result['sets'][k]['router_passed']}")
        (OUT / 'sealed_scores.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    else:
        nbm_flips, nbm_gate = nbm.flips, nbm.name_blind_gate
        before = {r['sheet_id']: r['pred'] for r in jl(OUT / 'sealed_predictions.jsonl')}
        res = {'checkpoint': CKPT, 'repo_commit': MAIN_SHA, 'blanker': 'scripts/router/name_blind.py', 'blanker_sha256': sha(BLIND_SRC.read_bytes()), 'sets': {}}
        for k in ('v5', 'v8'):
            gold = [r['gold'] for r in preds[k]]; aft = [r['pred'] for r in preds[k]]; bef = [before[r['sheet_id']] for r in preds[k]]
            table = nbm_flips(gold, bef, aft, labels)
            gate = nbm_gate(table, MIN_SEALED_PER_LABEL)
            res['sets'][k] = {'n': len(gold), 'rows_with_blanked_fields': sum(r['blanked_fields'] > 0 for r in preds[k]), 'fields_blanked': sum(r['blanked_fields'] for r in preds[k]),
                              'flips_total': sum(b != x for b, x in zip(bef, aft)), 'per_class': table, 'gate': gate,
                              'accuracy_before': sum(x == y for x, y in zip(gold, bef)) / len(gold), 'accuracy_after': sum(x == y for x, y in zip(gold, aft)) / len(gold)}
            say(f"{k} name-blind: flips={res['sets'][k]['flips_total']} gate={gate}")
        (OUT / 'name_blind.json').write_text(json.dumps(res, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    try:
        service.close('success').result()
    except Exception:
        pass
    say('DONE ' + a.phase)


if __name__ == '__main__':
    sys.exit(main())
