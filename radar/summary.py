"""Aggregate <results_dir>/raw/*.json (RADAR v4 eval records) into tables.

Outputs (in results_dir):
  summary.csv            one row per dataset × H × method × condition (all seeds averaged; MSE / MAE)
  summary_groups.csv     one row per dataset × H × method × (type, position) group (mean over lengths and ratios)
  summary.md             main tables: Clean + 4 types × 2 positions (9 columns) and the OOD table, MSE and MAE,
                         Δ of Ours-plugin vs each baseline, RADAR diagnostics (AUROC, gate, repair error, π)
  summary_appendix.md    breakdown by length k and by channel ratio
  summary_router.md      Router coefficients (log-odds vs E1) and π at open gates by gap length
Main methods: B0 GTR, B1 GTR+Aug, B2 GTR+Hampel, Ours-plugin; Ours (joint) / Ours-argmax are listed separately.
"""
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np

from radar.perturb import OOD_TYPES, POS_LABEL, POSITIONS, TYPE_LABEL, TYPES, parse_condition

MAIN_METHODS = ('B0', 'B1', 'B2', 'Ours-plugin')
EXTRA_METHODS = ('Ours-argmax', 'Ours')
ALL_METHODS = MAIN_METHODS + EXTRA_METHODS
BASELINES = ('B0', 'B1', 'B2')
OURS = 'Ours-plugin'
RADAR_METHODS = ('Ours-plugin', 'Ours-argmax', 'Ours')
METHOD_LABEL = {'B0': 'B0 GTR', 'B1': 'B1 GTR+Aug', 'B2': 'B2 GTR+Hampel', 'Ours-plugin': 'Ours-plugin',
                'Ours-argmax': 'Ours-plugin (argmax π)', 'Ours': 'Ours (joint)'}
DATASET_ORDER = ['etth1', 'etth2', 'ettm1', 'ettm2', 'weather', 'electricity', 'traffic', 'solar']
MAIN_GROUPS = [(t, p) for t in TYPES for p in POSITIONS]
OOD_GROUPS = [(t, p) for t in OOD_TYPES for p in POSITIONS]


def _ds_key(name):
    k = name.lower()
    return (DATASET_ORDER.index(k) if k in DATASET_ORDER else len(DATASET_ORDER), k)


def _mean(v):
    v = [x for x in v if x is not None and x == x]
    return float(np.mean(v)) if v else float('nan')


def _f(v, nd=4):
    return '–' if v is None or v != v else f'{v:.{nd}f}'


def _pct_impr(b, o):
    if b != b or o != o or not b:
        return '–'
    return f'{100.0 * (b - o) / b:+.2f}%'


def _gl(g):
    return f'{TYPE_LABEL[g[0]].replace(" (OOD)", "")}-{POS_LABEL[g[1]][0]}'


def load(raw_dir):
    evals, repro = [], []
    for fn in sorted(glob.glob(os.path.join(raw_dir, '*.json'))):
        try:
            r = json.load(open(fn))
        except Exception as e:  # pragma: no cover
            print(f'skip {fn}: {e}')
            continue
        if r.get('kind') == 'eval':
            evals.append(r)
        elif r.get('kind') == 'repro':
            repro.append(r)
    return evals, repro


def _table(md, header, rows):
    md.append('| ' + ' | '.join(header) + ' |')
    md.append('|' + '|'.join(['---'] * len(header)) + '|')
    for row in rows:
        md.append('| ' + ' | '.join(row) + ' |')
    md.append('')


def summarize(results_dir='./results_radar_v4/', print_tables=True, **_):
    raw_dir = os.path.join(results_dir, 'raw')
    evals, repro = load(raw_dir)
    if not evals:
        print(f'no eval results in {raw_dir}')
        return

    # ---------------- per condition: (ds, H, method, cond) -> {seed: (mse, mae)}
    cond_tab = defaultdict(dict)
    extras = defaultdict(dict)  # (ds, H, method) -> {seed: record}
    spec = {}
    for r in evals:
        if r['method'] not in ALL_METHODS:
            continue
        key = (r['dataset'], r['pred_len'], r['method'])
        for cond, v in r['conditions'].items():
            try:
                spec[cond] = parse_condition(cond)
            except ValueError:
                continue
            cond_tab[key + (cond,)][r['seed']] = (v['mse'], v['mae'])
        extras[key][r['seed']] = r
    cm = {k: (_mean([v[0] for v in d.values()]), _mean([v[1] for v in d.values()]), sorted(d))
          for k, d in cond_tab.items()}
    dsh = sorted({(k[0], k[1]) for k in cm}, key=lambda k: (_ds_key(k[0]), k[1]))
    methods = [m for m in ALL_METHODS if any(k[2] == m for k in cm)]
    main = [m for m in MAIN_METHODS if m in methods]

    def conds_of(ds, H, m):
        return [k[3] for k in cm if k[:3] == (ds, H, m)]

    def group_val(ds, H, m, g, idx, k=None, ratio=None):
        """mean over the conditions of group g (optionally one length / ratio). idx 0 = MSE, 1 = MAE."""
        vals = []
        for c in conds_of(ds, H, m):
            sp = spec.get(c)
            if sp is None or (sp['type'], sp['pos']) != g:
                continue
            if (k is not None and sp['k'] != k) or (ratio is not None and sp['ratio'] != ratio):
                continue
            vals.append(cm[(ds, H, m, c)][idx])
        return _mean(vals)

    def clean_val(ds, H, m, idx):
        v = cm.get((ds, H, m, 'clean'))
        return v[idx] if v else float('nan')

    groups_present = [g for g in MAIN_GROUPS + OOD_GROUPS
                      if any(spec.get(c) and (spec[c]['type'], spec[c]['pos']) == g for c in spec)]
    main_groups = [g for g in MAIN_GROUPS if g in groups_present]
    ood_groups = [g for g in OOD_GROUPS if g in groups_present]

    # ---------------- CSVs
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'summary.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dataset', 'pred_len', 'method', 'condition', 'type', 'position', 'k', 'channel_ratio',
                    'n_seeds', 'seeds', 'mse', 'mae'])
        for ds, H in dsh:
            for m in methods:
                for c in sorted(conds_of(ds, H, m), key=lambda c: (c != 'clean', c)):
                    sp = spec.get(c) or {}
                    mse, mae, seeds = cm[(ds, H, m, c)]
                    w.writerow([ds, H, m, c, sp.get('type', 'clean'), sp.get('pos', ''), sp.get('k', ''),
                                sp.get('ratio', ''), len(seeds), ' '.join(map(str, seeds)), mse, mae])
    with open(os.path.join(results_dir, 'summary_groups.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dataset', 'pred_len', 'method', 'group', 'mse', 'mae'])
        for ds, H in dsh:
            for m in methods:
                w.writerow([ds, H, m, 'clean', clean_val(ds, H, m, 0), clean_val(ds, H, m, 1)])
                for g in main_groups + ood_groups:
                    w.writerow([ds, H, m, f'{g[0]}_{g[1]}', group_val(ds, H, m, g, 0), group_val(ds, H, m, g, 1)])

    # ---------------- markdown: main tables
    md = []
    rows_avg_key = 'avg'

    def table_rows(meths, groups, idx, with_clean=True):
        rows = []
        for ds in sorted({d for d, _ in dsh}, key=_ds_key):
            Hs = [H for d, H in dsh if d == ds]
            for H in Hs + ([rows_avg_key] if len(Hs) > 1 else []):
                for m in meths:
                    def val(fn):
                        if H == rows_avg_key:
                            return _mean([fn(h) for h in Hs])
                        return fn(H)
                    if H != rows_avg_key and not conds_of(ds, H, m):
                        continue
                    row = [ds, str(H), METHOD_LABEL[m]]
                    if with_clean:
                        row.append(_f(val(lambda h: clean_val(ds, h, m, idx))))
                    row += [_f(val(lambda h, g=g: group_val(ds, h, m, g, idx))) for g in groups]
                    rows.append(row)
        return rows

    for idx, metric in ((0, 'MSE'), (1, 'MAE')):
        md.append(f'### {metric} — Clean + 4 types × 2 positions (each cell = mean over lengths and channel '
                  f'ratios; T = tail, I = internal; avg = mean over H)')
        _table(md, ['Dataset', 'H', 'Method', 'Clean'] + [_gl(g) for g in main_groups],
               table_rows(main, main_groups, idx))
        if ood_groups:
            md.append(f'### {metric} — OOD (type never used in training)')
            _table(md, ['Dataset', 'H', 'Method'] + [_gl(g) for g in ood_groups],
                   table_rows(main, ood_groups, idx, with_clean=False))
        extra = [m for m in EXTRA_METHODS if m in methods]
        if extra:
            md.append(f'### {metric} — additional RADAR variants')
            _table(md, ['Dataset', 'H', 'Method', 'Clean'] + [_gl(g) for g in main_groups + ood_groups],
                   table_rows([OURS] + extra if OURS in methods else extra, main_groups + ood_groups, idx))

    if OURS in methods:
        comps = [b for b in BASELINES if b in methods]
        md.append('### Δ MSE of Ours-plugin vs each baseline, (B − Ours) / B, + = Ours better')
        rows = []
        for ds, H in dsh:
            for b in comps:
                if not conds_of(ds, H, b) or not conds_of(ds, H, OURS):
                    continue
                row = [ds, str(H), f'vs {b}', _pct_impr(clean_val(ds, H, b, 0), clean_val(ds, H, OURS, 0))]
                row += [_pct_impr(group_val(ds, H, b, g, 0), group_val(ds, H, OURS, g, 0))
                        for g in main_groups + ood_groups]
                rows.append(row)
        _table(md, ['Dataset', 'H', 'Baseline', 'Clean'] + [_gl(g) for g in main_groups + ood_groups], rows)

    # ---------------- RADAR diagnostics
    def ex_group(rs, g, key, weight=None):
        vals, ws = [], []
        for r in rs:
            for c, e in r['extras'].items():
                if not isinstance(e, dict) or c not in spec or spec.get(c) is None:
                    continue
                if (spec[c]['type'], spec[c]['pos']) != g or key not in e:
                    continue
                v = e[key]
                if v is None or v != v:
                    continue
                vals.append(v)
                ws.append(e.get(weight, 0) if weight else 1.0)
        if not vals:
            return float('nan')
        ws = np.asarray(ws, dtype=np.float64)
        if ws.sum() <= 0:
            return float('nan')
        return float(np.average(np.asarray(vals, dtype=np.float64), axis=0, weights=ws)) \
            if np.ndim(vals[0]) == 0 else np.average(np.asarray(vals, dtype=np.float64), axis=0, weights=ws)

    rad = [m for m in RADAR_METHODS if m in methods]
    for m in rad:
        md.append(f'### {METHOD_LABEL[m]} — detection AUROC (positives = corrupted points, negatives = clean points)')
        rows = []
        for ds, H in dsh:
            rs = list(extras.get((ds, H, m), {}).values())
            if rs:
                rows.append([ds, str(H)] + [_f(ex_group(rs, g, 'auroc'), 3) for g in main_groups + ood_groups])
        _table(md, ['Dataset', 'H'] + [_gl(g) for g in main_groups + ood_groups], rows)

        md.append(f'### {METHOD_LABEL[m]} — gate recall (α>0.5 on corrupted points) / false repair on Clean '
                  f'(α>0.5 on clean points), a, b, τ')
        rows = []
        for ds, H in dsh:
            rs = list(extras.get((ds, H, m), {}).values())
            if not rs:
                continue
            fr = _mean([r['extras'].get('clean', {}).get('frac_alpha_gt_0_5') for r in rs])
            gv = rs[0]['extras'].get('gate', {})
            rows.append([ds, str(H), f'{100 * fr:.2f}%' if fr == fr else '–']
                        + [_f(ex_group(rs, g, 'gate_recall'), 3) for g in main_groups + ood_groups]
                        + [_f(gv.get('a'), 2), _f(gv.get('b'), 2), _f(gv.get('tau'), 2),
                           _f(_mean([r.get('time_ratio_vs_B0') for r in rs]), 2)])
        _table(md, ['Dataset', 'H', 'false repair'] + [_gl(g) for g in main_groups + ood_groups]
               + ['softplus(a)', 'softplus(b)', 'τ', 'time ×B0'], rows)

        md.append(f'### {METHOD_LABEL[m]} — repair error on corrupted points |z − x_clean| / s_clean '
                  f'(raw input → routed target → repaired input x̃)')
        rows = []
        for ds, H in dsh:
            rs = list(extras.get((ds, H, m), {}).values())
            if not rs:
                continue
            row = [ds, str(H)]
            for g in main_groups + ood_groups:
                row.append(f"{_f(ex_group(rs, g, 'err_raw'), 2)} → {_f(ex_group(rs, g, 'err_target'), 2)} → "
                           f"{_f(ex_group(rs, g, 'err_tilde'), 2)}")
            rows.append(row)
        _table(md, ['Dataset', 'H'] + [_gl(g) for g in main_groups + ood_groups], rows)

        md.append(f'### {METHOD_LABEL[m]} — π at open gates (α>0.5 on corrupted points), E1 / E2 / E3')
        rows = []
        for ds, H in dsh:
            rs = list(extras.get((ds, H, m), {}).values())
            if not rs:
                continue
            row = [ds, str(H)]
            for g in main_groups + ood_groups:
                v = ex_group(rs, g, 'pi_open', weight='n_open')
                row.append('–' if np.ndim(v) == 0 else '/'.join(f'{x:.2f}' for x in v))
            rows.append(row)
        _table(md, ['Dataset', 'H'] + [_gl(g) for g in main_groups + ood_groups], rows)

    if repro:
        md.append('### S0 reproduction check (Exp_Radar S0 vs original run.py, same seed)')
        _table(md, ['Setting', 'run.py MSE', 'S0 MSE', 'rel. diff', 'pass (<2%)'],
               [[r['setting'], f"{r['orig_mse']:.6f}", f"{r['s0_mse']:.6f}", f"{100 * r['rel_diff_mse']:.3f}%",
                 'yes' if r['passed'] else 'NO'] for r in repro])
    text = '\n'.join(md)
    with open(os.path.join(results_dir, 'summary.md'), 'w') as f:
        f.write(text + '\n')

    # ---------------- appendix: by length and by channel ratio
    ap = []
    order_t = list(TYPES + OOD_TYPES)
    lengths = sorted({(spec[c]['type'], spec[c]['pos'], spec[c]['k']) for c in spec if spec[c]
                      and spec[c]['type'] in order_t},
                     key=lambda v: (order_t.index(v[0]), POSITIONS.index(v[1]), v[2]))
    ratios = sorted({spec[c]['ratio'] for c in spec if spec[c]})
    for idx, metric in ((0, 'MSE'), (1, 'MAE')):
        ap.append(f'### {metric} by length k (mean over channel ratios)')
        hdr = ['Dataset', 'H', 'Method'] + [f'{_gl((t, p))} k{k}' for t, p, k in lengths]
        rows = [[ds, str(H), METHOD_LABEL[m]] + [_f(group_val(ds, H, m, (t, p), idx, k=k)) for t, p, k in lengths]
                for ds, H in dsh for m in methods if conds_of(ds, H, m)]
        _table(ap, hdr, rows)
        if len(ratios) > 1:
            ap.append(f'### {metric} by channel ratio (mean over lengths)')
            gr = [(g, c) for g in main_groups + ood_groups for c in ratios]
            hdr = ['Dataset', 'H', 'Method'] + [f'{_gl(g)} c{c}' for g, c in gr]
            rows = [[ds, str(H), METHOD_LABEL[m]] + [_f(group_val(ds, H, m, g, idx, ratio=c)) for g, c in gr]
                    for ds, H in dsh for m in methods if conds_of(ds, H, m)]
            _table(ap, hdr, rows)
    with open(os.path.join(results_dir, 'summary_appendix.md'), 'w') as f:
        f.write('\n'.join(ap) + '\n')

    # ---------------- Router report
    rt = []
    for m in rad:
        for ds, H in dsh:
            rs = list(extras.get((ds, H, m), {}).values())
            if not rs:
                continue
            ex0 = rs[0]['extras']
            coef = ex0.get('router_coef')
            names = ex0.get('experts', ['E1', 'E2', 'E3'])
            rt.append(f'### {ds} H={H} {METHOD_LABEL[m]} — router {ex0.get("router")}, E3 period '
                      f'{ex0.get("e3_period")} ({"available" if ex0.get("e3_enabled") else "unavailable"}), '
                      f'T_route {_f(ex0.get("gate", {}).get("route_temp"), 2)}')
            if coef:
                feats = ['bias'] + list(ex0.get('route_features', []))
                exps = list(coef.keys())
                rt.append('Coefficients = log-odds of the expert instead of E1 (Repairer), relative to the reference '
                          'bins (gap 0, both sides reliable, distance 0).')
                _table(rt, ['feature'] + [f'{e} vs {names[0]}' for e in exps],
                       [[ft] + [_f(coef[e].get(ft), 3) for e in exps] for ft in feats])
            bins = None
            agg = defaultdict(lambda: [np.zeros(len(names)), 0])
            for r in rs:
                for c, e in r['extras'].items():
                    if not isinstance(e, dict) or 'pi_open_by_gap' not in e:
                        continue
                    for b, v in e['pi_open_by_gap'].items():
                        n = e['n_open_by_gap'].get(b, 0)
                        if n:
                            agg[b][0] += np.asarray(v) * n
                            agg[b][1] += n
                    bins = list(e['pi_open_by_gap'].keys())
            if bins:
                rt.append('π at open gates by gap length (all corrupted conditions pooled; n = open points)')
                _table(rt, ['gap length', 'n'] + list(names),
                       [[b, str(agg[b][1])] + ([f'{x:.2f}' for x in agg[b][0] / agg[b][1]] if agg[b][1]
                                               else ['–'] * len(names)) for b in bins])
    with open(os.path.join(results_dir, 'summary_router.md'), 'w') as f:
        f.write('\n'.join(rt) + '\n')

    if print_tables:
        print(text)
    print(f'wrote {results_dir}/summary.csv, summary_groups.csv, summary.md, summary_appendix.md, '
          f'summary_router.md')
