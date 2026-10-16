"""Score router v6 (epoch 3) ONCE on the new v9 web_f sealed addendum, then the name-blind check on it once.

Same logic as docs/router/runs/v6/sealed/score_v6_v9.py (same prompt, sampling, parser, gate, blanker). Only the set changes:
v6 was already scored on v5_retick, v8_retick, cr, thin_c and thin_d (PR #61), so those are NOT re-sampled here; the combined
view is built afterwards by combine_v6_web_f.py from the recorded per-row predictions. The driver refuses to sample a row
that already has a recorded v6 prediction, and each phase refuses a second run (resume only fills rows missing after a
failed call). Never prints secrets.
Usage: score_v6_web_f.py {dry|score|blind} --main <byte-exact 92f8f0a copy> --repo <92f8f0a + addendum work tree> --cands <candidates jsonl>
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

OUT = Path(__file__).resolve().parent / 'out'
SET = 'web_f_addendum'
CKPT = 'tinker://93552b55-3c20-5862-8bbf-2320fb04b236:train:0/sampler_weights/router-v6-qwen3.5-4b-lora-r16-epoch3'
MAIN_TREE = '3b22b5d036b4ac6738f203ee4eeb48835781128f'  # tree of main 92f8f0a
SEAL_SHA256 = '9a333bfab4dcaab829c7ca695ae671ac544645dbe6915b38300eed62d4405e72'  # sealed_labels.jsonl, fixed before scoring
TRAIN_SYSTEM_SHA256 = 'b1ac9472219a4bea8b0447c2418854a85d60bcb7f154eff2252c7ef8718e3879'
BUNDLE_REPLACES = {'configs/router/sealed_eval.json'}  # files of main the addendum change edits (others are new files)


def jl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding='utf-8').splitlines() if l.strip()]


def sha(b):
    return hashlib.sha256(b).hexdigest()


def git_tree(path: Path) -> str:
    entries = []
    for child in path.iterdir():
        if child.name in ('.git', '__pycache__', '.pytest_cache'):
            continue
        if child.is_dir():
            entries.append((child.name + '/', b'40000 ' + child.name.encode() + b'\0' + bytes.fromhex(git_tree(child))))
        else:
            data = child.read_bytes()
            mode = b'100755' if os.access(child, os.X_OK) else b'100644'
            h = hashlib.sha1(b'blob %d\0' % len(data) + data).digest()
            entries.append((child.name, mode + b' ' + child.name.encode() + b'\0' + h))
    body = b''.join(e for _, e in sorted(entries, key=lambda x: x[0].encode()))
    return hashlib.sha1(b'tree %d\0' % len(body) + body).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=['dry', 'score', 'blind'])
    ap.add_argument('--main', type=Path, required=True)
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--cands', type=Path, required=True)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    log = open(OUT / 'score.log', 'a', encoding='utf-8')

    def say(*x):
        s = ' '.join(str(i) for i in x)
        print(s, flush=True)
        log.write(s + '\n'); log.flush()

    if a.phase != 'dry':
        marker = OUT / f'RUN_STARTED_{a.phase}.json'
        partial = OUT / f'{a.phase}_partial.jsonl'
        final = OUT / ('sealed_scores.json' if a.phase == 'score' else 'name_blind.json')
        resume = False
        if marker.exists():
            if final.exists() or not partial.exists():
                say(f'WALL: phase {a.phase} already started; no re-runs.')
                return 3
            resume = True
        if a.phase == 'blind' and not (OUT / 'sealed_scores.json').exists():
            say('WALL: blind phase needs the recorded score first'); return 3
        if not (OUT / 'PRE_RUN_NOTE.txt').exists():
            say('WALL: no pre-run note'); return 3
    if git_tree(a.main) != MAIN_TREE:
        say('WALL: main copy is not byte-exact 92f8f0a'); return 3
    REPO = a.repo
    differ = []
    for p in a.main.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts:
            rel = str(p.relative_to(a.main))
            if rel not in BUNDLE_REPLACES and (not (REPO / rel).is_file() or (REPO / rel).read_bytes() != p.read_bytes()):
                differ.append(rel)
    if differ:
        say(f'WALL: work tree differs from main outside the addendum change at {len(differ)} files'); return 3
    sys.path.insert(0, str(REPO / 'scripts'))
    sys.path.insert(0, '/workspace/router-train/driver')
    from router.labels import load_labels, load_sealed_config
    from router.teacher_label import prompt_for
    from router.content import content_text
    import router.train_router as tr
    import router.name_blind as nbm
    from router.sealed_gate import evaluate_sealed_gate_v8, macro_recall, MIN_SEALED_PER_LABEL
    cfg = load_sealed_config(REPO / 'configs/router/sealed_eval.json')
    v9 = cfg['v9_sealed']
    item = [s for s in v9['sets'] if s['name'] == SET]
    assert len(item) == 1
    sp = REPO / item[0]['sealed_labels']
    if sha(sp.read_bytes()) != SEAL_SHA256 or (sp.parent / 'sealed_labels.jsonl.sha256').read_text(encoding='utf-8').split()[0] != SEAL_SHA256:
        say('WALL: seal mismatch'); return 3
    rec = json.loads((REPO / 'docs/router/runs/v6/RUN_RECORD.json').read_text(encoding='utf-8'))
    assert rec['checkpoint_sampler_uri_epoch3'] == CKPT
    prior = jl(REPO / 'docs/router/runs/v6/sealed/sealed_predictions.jsonl')
    gold = jl(sp)
    if {r['sheet_id'] for r in prior} & {g['sheet_id'] for g in gold} or {r['content_sha256'] for r in prior} & {g['file_sha256'] for g in gold}:
        say('WALL: a row of this set already has a recorded v6 prediction'); return 3
    say(f'main 92f8f0a verified (tree {MAIN_TREE[:7]}); work tree = main + addendum; seal {SEAL_SHA256[:12]} verified; {len(prior)} prior v6 rows, 0 shared with this set')
    labels = load_labels(REPO / v9['labels_file'])
    assert len(labels) == 22
    mcf = bool(cfg['sealed_gate'].get('majority_class_fail', cfg['majority_class_fail']))
    lora = json.loads((REPO / 'configs/lora/qwen3-4b-router-r16.json').read_text(encoding='utf-8'))
    system = prompt_for(labels)
    if sha(system.encode()) != TRAIN_SYSTEM_SHA256:
        say('WALL: system prompt differs from the one v6 trained with'); return 3
    blind_src = REPO / 'scripts/router/name_blind.py'
    cmap = {r['id']: r for r in jl(a.cands)}
    items = []
    for g in gold:
        c = cmap[g['sheet_id']]
        if c['sha256'] != g['file_sha256']:
            say(f"WALL: content sha mismatch {g['sheet_id']}"); return 3
        text = content_text(c)
        if not text:
            say(f"WALL: empty text {g['sheet_id']}"); return 3
        nb = 0
        if a.phase == 'blind':
            text, nb = nbm.blank_fields(text)
        else:
            nb = nbm.blank_fields(text)[1]
        items.append({'sheet_id': g['sheet_id'], 'content_sha256': g['file_sha256'], 'gold': g['label'], 'text': text, 'blanked_fields': nb})
    say(f'rows {len(items)}; joined {len(items)}/{len(gold)}; empty 0; rows with a blankable field {sum(i["blanked_fields"] > 0 for i in items)}; '
        f'name_blind.py sha256 {sha(blind_src.read_bytes())[:12]}; candidates sha256 {sha(a.cands.read_bytes())[:12]}')
    if a.phase == 'dry':
        return 0
    if not resume:
        marker.write_text(json.dumps({'started': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'checkpoint': CKPT, 'repo_commit': '92f8f0a750940bb88d7d4cb6bc0975bda01837a4 + web_f addendum (seal ' + SEAL_SHA256 + ')',
                                      'set': SET, 'n': len(items), 'system_sha256': sha(system.encode()), 'name_blind_sha256': sha(blind_src.read_bytes())}) + '\n')
    else:
        say(f'RESUME {a.phase}: marker present, no final output; sampling only rows missing from {partial.name}')
    from common import load_env_file
    names = load_env_file(Path('/workspace/slm-secrets/tinker.env'))
    say(f'env names loaded: {len(names)}; checkpoint={CKPT}; phase={a.phase}')
    import tinker
    service = tinker.ServiceClient()
    client = service.create_sampling_client(model_path=CKPT)
    tok = client.get_tokenizer()
    params = tinker.types.SamplingParams(max_tokens=32, temperature=0, top_p=1, seed=int(lora['seed']))

    def one(it):
        msgs = [{'role': 'system', 'content': system}, {'role': 'user', 'content': it['text']}]
        for att in range(5):
            try:
                t = tr._complete(client, tok, tinker.types, msgs, params)
                return t, tr._as_label(t, labels)
            except Exception:
                if att == 4:
                    raise
                time.sleep(2 * (att + 1))
    done = {}
    if partial.exists():
        for r in jl(partial):
            done[r['sheet_id']] = (r['raw'], r['pred'])
    todo = [it for it in items if it['sheet_id'] not in done]
    say(f'rows total {len(items)}; already sampled {len(done)}; to sample {len(todo)}')
    import threading
    lock = threading.Lock()
    pf = open(partial, 'a', encoding='utf-8')

    def run(it):
        raw, p = one(it)
        with lock:
            pf.write(json.dumps({'set': SET, 'sheet_id': it['sheet_id'], 'raw': raw, 'pred': p}, ensure_ascii=False) + '\n'); pf.flush()
            done[it['sheet_id']] = (raw, p)
    errors = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        for fut in [pool.submit(run, it) for it in todo]:
            try:
                fut.result()
            except Exception as e:
                errors.append(type(e).__name__)
    pf.close()
    if errors:
        say(f'WALL: {len(errors)} rows failed after 5 attempts ({sorted(set(errors))}); re-run resumes only the missing rows'); return 4
    preds = [{'set': SET, 'sheet_id': it['sheet_id'], 'content_sha256': it['content_sha256'], 'gold': it['gold'], 'pred': done[it['sheet_id']][1],
              'raw': done[it['sheet_id']][0], 'blanked_fields': it['blanked_fields']} for it in items]
    tag = 'sealed' if a.phase == 'score' else 'name_blind'
    (OUT / f'{tag}_predictions.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in preds), encoding='utf-8')
    gold_l = [r['gold'] for r in preds]; pred_l = [r['pred'] for r in preds]
    if a.phase == 'score':
        g = evaluate_sealed_gate_v8(gold_l, pred_l, labels, majority_class_fail=mcf)
        result = {'repo_commit': '92f8f0a750940bb88d7d4cb6bc0975bda01837a4 + web_f addendum', 'seal_sha256': SEAL_SHA256, 'checkpoint': CKPT, 'labels': 'configs/router/labels.json (v9, 22)',
                  'system': 'router.teacher_label.prompt_for(labels)', 'parser': 'router.train_router._as_label -> parse_router_output (v9 exact aliases)',
                  'sampling': {'max_tokens': 32, 'temperature': 0, 'top_p': 1, 'seed': int(lora['seed']), 'enable_thinking': False},
                  'gate': 'evaluate_sealed_gate_v8 (gate v2 + MIN_SEALED_PER_LABEL=%d), unchanged' % MIN_SEALED_PER_LABEL, 'majority_class_fail': mcf,
                  'sets': {SET: {'n': len(gold_l), 'accuracy': sum(x == y for x, y in zip(gold_l, pred_l)) / len(gold_l), 'macro_recall': g['gate_v2']['macro_recall'],
                                 'gate_v2_passed': bool(g['gate_v2']['passed']), 'gate_v2': g['gate_v2'], 'router_passed': bool(g['router_passed']), 'per_label': g['per_label'],
                                 'pred_counts': dict(Counter(pred_l)), 'unsure': sum(p == 'unsure' for p in pred_l),
                                 'constant_macro': {lab: macro_recall(gold_l, [lab] * len(gold_l), labels) for lab in labels}}}}
        s = result['sets'][SET]
        say(f"{SET}: n={s['n']} macro={s['macro_recall']} acc={s['accuracy']:.4f} gate_v2={s['gate_v2_passed']} router_passed={s['router_passed']}")
        (OUT / 'sealed_scores.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    else:
        before = {r['sheet_id']: r['pred'] for r in jl(OUT / 'sealed_predictions.jsonl')}
        bef = [before[r['sheet_id']] for r in preds]
        table = nbm.flips(gold_l, bef, pred_l, labels)
        gate = nbm.name_blind_gate(table, MIN_SEALED_PER_LABEL)
        res = {'checkpoint': CKPT, 'repo_commit': '92f8f0a750940bb88d7d4cb6bc0975bda01837a4 + web_f addendum', 'blanker': 'scripts/router/name_blind.py', 'blanker_sha256': sha(blind_src.read_bytes()),
               'sets': {SET: {'n': len(gold_l), 'rows_with_blanked_fields': sum(r['blanked_fields'] > 0 for r in preds), 'fields_blanked': sum(r['blanked_fields'] for r in preds),
                              'flips_total': sum(b != x for b, x in zip(bef, pred_l)),
                              'flips_passable': sum(row['flips'] for row in table.values() if row['n'] >= MIN_SEALED_PER_LABEL),
                              'flips_not_passable': sum(row['flips'] for row in table.values() if row['n'] < MIN_SEALED_PER_LABEL),
                              'flips_on_rows_with_nothing_blanked': sum(r['blanked_fields'] == 0 and r['pred'] != before[r['sheet_id']] for r in preds),
                              'per_class': table, 'gate': gate,
                              'accuracy_before': sum(x == y for x, y in zip(gold_l, bef)) / len(gold_l), 'accuracy_after': sum(x == y for x, y in zip(gold_l, pred_l)) / len(gold_l)}}}
        s = res['sets'][SET]
        say(f"{SET} name-blind: flips={s['flips_total']} (passable {s['flips_passable']}, not passable {s['flips_not_passable']}) gate={gate}")
        (OUT / 'name_blind.json').write_text(json.dumps(res, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    try:
        service.close('success').result()
    except Exception:
        pass
    say('DONE ' + a.phase)


if __name__ == '__main__':
    sys.exit(main())
