"""Combined v9 sealed view for router v6 WITHOUT any new sampling.

combined (312) = v6's already-recorded per-row predictions on v5_retick, v8_retick, cr_addendum, thin_c_addendum, thin_d_addendum
(docs/router/runs/v6/sealed/sealed_predictions.jsonl, 278 rows, PR #61) + the 34 web_f predictions recorded once by
score_v6_web_f.py, concatenated in configs/router/sealed_eval.json v9_sealed order. Same gate (evaluate_sealed_gate_v8) and
the same name-blind flip table (router.name_blind.flips / name_blind_gate) on the same stored predictions. Gold labels are
re-read from the sealed files and must equal the gold stored with each prediction. Precision per class is added for reporting.
Usage: combine_v6_web_f.py --repo <work tree> --new <dir with this run's sealed/name_blind predictions> --out <dir>
"""
import argparse, json, sys
from collections import Counter
from pathlib import Path


def jl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding='utf-8').splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--new', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(a.repo / 'scripts'))
    from router.labels import load_labels, load_sealed_config
    from router.sealed_gate import evaluate_sealed_gate_v8, MIN_SEALED_PER_LABEL
    import router.name_blind as nbm
    cfg = load_sealed_config(a.repo / 'configs/router/sealed_eval.json')
    v9 = cfg['v9_sealed']
    labels = load_labels(a.repo / v9['labels_file'])
    mcf = bool(cfg['sealed_gate'].get('majority_class_fail', cfg['majority_class_fail']))
    old = a.repo / 'docs/router/runs/v6/sealed'
    sealed_p = jl(old / 'sealed_predictions.jsonl') + jl(a.new / 'sealed_predictions.jsonl')
    blind_p = jl(old / 'name_blind_predictions.jsonl') + jl(a.new / 'name_blind_predictions.jsonl')
    by = {(r['set'], r['sheet_id']): r for r in sealed_p}
    bb = {(r['set'], r['sheet_id']): r for r in blind_p}
    order = []
    for item in v9['sets']:
        for g in jl(a.repo / item['sealed_labels']):
            k = (item['name'], g['sheet_id'])
            assert by[k]['gold'] == g['label'] == bb[k]['gold'] and by[k]['content_sha256'] == g['file_sha256'], k
            order.append(k)
    assert len(order) == v9['combined']['rows'] == len(by) == len(bb)
    gold = [by[k]['gold'] for k in order]; pred = [by[k]['pred'] for k in order]; aft = [bb[k]['pred'] for k in order]
    g = evaluate_sealed_gate_v8(gold, pred, labels, majority_class_fail=mcf)
    pc = Counter(pred)
    per = []
    for row in g['per_label']:
        lab = row['label']; tp = sum(x == lab and y == lab for x, y in zip(gold, pred))
        per.append({**row, 'tp': tp, 'predicted': pc.get(lab, 0), 'precision': (tp / pc[lab]) if pc.get(lab) else None})
    sets = {}
    for name in [s['name'] for s in v9['sets']]:
        ks = [k for k in order if k[0] == name]
        gg = [by[k]['gold'] for k in ks]; pp = [by[k]['pred'] for k in ks]
        sets[name] = {'n': len(ks), 'accuracy': sum(x == y for x, y in zip(gg, pp)) / len(ks)}
    shas = Counter(by[k]['content_sha256'] for k in order)
    dup = [{'content_sha256': h[:12], 'sets': sorted({k[0] for k in order if by[k]['content_sha256'] == h})} for h, n in shas.items() if n > 1]
    scores = {'view': 'combined', 'checkpoint': by[order[0]].get('checkpoint') or 'tinker://93552b55-3c20-5862-8bbf-2320fb04b236:train:0/sampler_weights/router-v6-qwen3.5-4b-lora-r16-epoch3',
              'method': 'aggregation of recorded per-row predictions; no new sampling. 278 rows from docs/router/runs/v6/sealed/sealed_predictions.jsonl (PR #61) + 34 rows from docs/router/runs/v6/web_f/sealed_predictions.jsonl, in configs/router/sealed_eval.json v9_sealed order',
              'n': len(gold), 'accuracy': sum(x == y for x, y in zip(gold, pred)) / len(gold), 'macro_recall': g['gate_v2']['macro_recall'],
              'gate_v2_passed': bool(g['gate_v2']['passed']), 'gate_v2': g['gate_v2'], 'router_passed': bool(g['router_passed']), 'per_label': per,
              'pred_counts': dict(pc), 'per_set_accuracy': sets,
              'documents_registered_in_two_sets': dup,
              'note': 'a document registered in two sets is counted once per set in the combined view, as registered (unchanged precedent); found from content sha256, not from ids'}
    table = nbm.flips(gold, pred, aft, labels)
    gate = nbm.name_blind_gate(table, MIN_SEALED_PER_LABEL)
    flips = [{'set': k[0], 'sheet_id': k[1], 'gold': by[k]['gold'], 'before': by[k]['pred'], 'after': bb[k]['pred']} for k in order if by[k]['pred'] != bb[k]['pred']]
    blind = {'view': 'combined', 'method': 'same flip table on the recorded unblanked and blanked predictions (no new sampling): 278 rows from docs/router/runs/v6/sealed/name_blind_predictions.jsonl + 34 from docs/router/runs/v6/web_f/name_blind_predictions.jsonl',
             'n': len(gold), 'flips_total': len(flips), 'flips_passable': sum(r['flips'] for r in table.values() if r['n'] >= MIN_SEALED_PER_LABEL),
             'flips_not_passable': sum(r['flips'] for r in table.values() if r['n'] < MIN_SEALED_PER_LABEL), 'flips': flips, 'per_class': table, 'gate': gate,
             'accuracy_before': scores['accuracy'], 'accuracy_after': sum(x == y for x, y in zip(gold, aft)) / len(gold)}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'combined_scores.json').write_text(json.dumps(scores, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    (a.out / 'combined_name_blind.json').write_text(json.dumps(blind, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f"combined n={scores['n']} macro={scores['macro_recall']:.4f} acc={scores['accuracy']:.4f} gate_v2={scores['gate_v2_passed']} router_passed={scores['router_passed']}")
    print(f"combined name-blind flips={blind['flips_total']} passable={blind['flips_passable']} gate={gate}")
    for r in per:
        print(f"{r['label']:20s} n={r['n']:3d} recall={r['recall'] if r['recall'] is None else round(r['recall'],3)} precision={None if r['precision'] is None else round(r['precision'],3)} passable={r['passable']}")


if __name__ == '__main__':
    main()
