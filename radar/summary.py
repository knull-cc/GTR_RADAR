"""Aggregate results_radar/raw/*.json into summary.csv / summary_extras.csv / summary.md and print markdown.

Main protocol: conditions clean / T1_raw / T2_k4; methods B0, B1 (GTR+Aug), B2 (GTR+Hampel), Ours-plugin.
Ours (joint training) is reported only in a separate ablation table.
"""
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np

from radar.perturb import ALL_CONDITIONS, TEST_CONDITIONS

MAIN_METHODS = ('B0', 'B1', 'B2', 'Ours-plugin')
ABLATION_METHODS = ('Ours',)
ALL_METHODS = MAIN_METHODS + ABLATION_METHODS
BASELINES = ('B0', 'B1', 'B2')
OURS = 'Ours-plugin'
METHOD_LABEL = {'B0': 'B0 GTR', 'B1': 'B1 GTR+Aug', 'B2': 'B2 GTR+Hampel', 'Ours-plugin': 'Ours-plugin',
                'Ours': 'Ours (joint)'}
DATASET_ORDER = ['etth1', 'ettm1', 'weather', 'etth2', 'ettm2', 'solar', 'electricity', 'traffic']


def _ds_key(name):
    k = name.lower()
    return (DATASET_ORDER.index(k) if k in DATASET_ORDER else len(DATASET_ORDER), k)


def _ms(vals):
    v = np.asarray(vals, dtype=np.float64)
    if v.size == 0:
        return float('nan'), float('nan')
    return float(v.mean()), (float(v.std(ddof=1)) if v.size > 1 else 0.0)


def _fmt(m, s, nd=4):
    if m != m:  # nan
        return '–'
    return f'{m:.{nd}f} ± {s:.{nd}f}'


def _pct(b, o):
    if b != b or o != o or b == 0:
        return '–'
    return f'{100.0 * (b - o) / b:+.2f}%'


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


def summarize(results_dir='./results_radar/', conditions=TEST_CONDITIONS, print_tables=True):
    conditions = [c for c in ALL_CONDITIONS if c in set(conditions)]
    raw_dir = os.path.join(results_dir, 'raw')
    evals, repro = load(raw_dir)
    if not evals:
        print(f'no eval results in {raw_dir}')
        return

    # (ds, H, method, cond) -> {seed: (mse, mae)}
    tab = defaultdict(dict)
    extras = defaultdict(dict)  # (ds, H, method) -> {seed: res}
    for r in evals:
        if r['method'] not in ALL_METHODS:
            continue
        key = (r['dataset'], r['pred_len'])
        for cond, v in r['conditions'].items():
            if cond in conditions:
                tab[key + (r['method'], cond)][r['seed']] = (v['mse'], v['mae'])
        extras[key + (r['method'],)][r['seed']] = r

    keys = sorted({(k[0], k[1]) for k in tab}, key=lambda k: (_ds_key(k[0]), k[1]))
    stats = {}
    for (ds, H, m, cond), d in tab.items():
        seeds = sorted(d)
        clean = tab.get((ds, H, m, 'clean'), {})
        stats[(ds, H, m, cond)] = dict(seeds=seeds, mse=_ms([d[s][0] for s in seeds]),
                                       mae=_ms([d[s][1] for s in seeds]),
                                       degr=_ms([d[s][0] / clean[s][0] for s in seeds if s in clean]))

    # ---------------- CSV (all methods incl. ablation)
    os.makedirs(results_dir, exist_ok=True)
    csv_fn = os.path.join(results_dir, 'summary.csv')
    with open(csv_fn, 'w', newline='') as f:
        w = csv.writer(f)
        hdr = ['dataset', 'pred_len', 'method', 'condition', 'n_seeds', 'seeds', 'mse_mean', 'mse_std',
               'mae_mean', 'mae_std', 'degradation_mean', 'degradation_std']
        hdr += [f'{mt}_impr_vs_{b}' for mt in ('mse', 'mae') for b in BASELINES]
        w.writerow(hdr)
        for ds, H in keys:
            for cond in conditions:
                for m in ALL_METHODS:
                    s = stats.get((ds, H, m, cond))
                    if s is None:
                        continue
                    imps = []
                    for mt in ('mse', 'mae'):
                        for b in BASELINES:
                            sb = stats.get((ds, H, b, cond))
                            bm = sb[mt][0] if sb else None
                            imps.append((bm - s[mt][0]) / bm if bm else '')
                    w.writerow([ds, H, m, cond, len(s['seeds']), ' '.join(map(str, s['seeds'])),
                                *s['mse'], *s['mae'], *s['degr'], *imps])

    # ---------------- extras CSV
    ex_fn = os.path.join(results_dir, 'summary_extras.csv')
    ex_rows = []
    for (ds, H, m), per_seed in sorted(extras.items(), key=lambda kv: (_ds_key(kv[0][0]), kv[0][1], kv[0][2])):
        rs = list(per_seed.values())

        def avg(fn):
            vals = []
            for r in rs:
                try:
                    vals.append(fn(r))
                except (KeyError, TypeError, IndexError):
                    pass
            return _ms(vals)[0] if vals else float('nan')

        row = dict(dataset=ds, pred_len=H, method=m, n_seeds=len(rs),
                   time_ratio_vs_B0=avg(lambda r: r['time_ratio_vs_B0']))
        if m in ('Ours', 'Ours-plugin'):
            for cond in ('clean', 'T1_raw'):
                row[f'mean_alpha_{cond}'] = avg(lambda r: r['extras'][cond]['mean_alpha'])
                row[f'frac_alpha_gt_0.5_{cond}'] = avg(lambda r: r['extras'][cond]['frac_alpha_gt_0_5'])
                row[f'alpha_last_point_{cond}'] = avg(lambda r: r['extras'][cond]['mean_alpha_last_point'])
                for k, name in enumerate(('repairer', 'localmedian', 'cycleprior')):
                    row[f'pi_{name}_{cond}'] = avg(lambda r: r['extras'][cond]['mean_pi'][k])
            row['T1_last_point_auroc'] = avg(lambda r: r['extras']['T1_last_point_auroc'])
            row['tau'] = avg(lambda r: r['extras']['tau'])
            prof = [r['extras']['clean']['mean_dbar_per_patch'] for r in rs if 'clean' in r.get('extras', {})]
            row['mean_dbar_per_patch_clean'] = ' '.join(f'{v:.3f}' for v in np.mean(prof, axis=0)) if prof else ''
        ex_rows.append(row)
    cols = []
    for r in ex_rows:
        cols += [c for c in r if c not in cols]
    with open(ex_fn, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in ex_rows:
            w.writerow(r)

    # ---------------- markdown
    md = []

    def table(header, rows):
        md.append('| ' + ' | '.join(header) + ' |')
        md.append('|' + '|'.join(['---'] * len(header)) + '|')
        for row in rows:
            md.append('| ' + ' | '.join(row) + ' |')
        md.append('')

    main = [m for m in MAIN_METHODS if any(k[2] == m for k in stats)]
    comps = [b for b in BASELINES if b in main] if OURS in main else []
    for metric in ('mse', 'mae'):
        for cond in conditions:
            md.append(f'### {metric.upper()} — {cond}  (mean ± std over seeds; Δ = (B − Ours-plugin)/B, '
                      f'+ = Ours-plugin better)')
            hdr = ['Dataset', 'H', 'seeds'] + [METHOD_LABEL[m] for m in main] + [f'Δ vs {b}' for b in comps]
            rows = []
            for ds, H in keys:
                if not any((ds, H, m, cond) in stats for m in main):
                    continue
                n = max(len(stats[(ds, H, m, cond)]['seeds']) for m in main if (ds, H, m, cond) in stats)
                row = [ds, str(H), str(n)]
                for m in main:
                    s = stats.get((ds, H, m, cond))
                    row.append(_fmt(*s[metric]) if s else '–')
                so = stats.get((ds, H, OURS, cond))
                for b in comps:
                    sb = stats.get((ds, H, b, cond))
                    row.append(_pct(sb[metric][0], so[metric][0]) if so and sb else '–')
                rows.append(row)
            table(hdr, rows)

    pert = [c for c in conditions if c != 'clean']
    if pert:
        md.append('### Degradation rate = MSE(perturbed) / MSE(clean)  (mean over seeds)')
        rows = []
        for ds, H in keys:
            for m in main:
                if (ds, H, m, 'clean') not in stats:
                    continue
                rows.append([ds, str(H), METHOD_LABEL[m]] +
                            [f"{stats[(ds, H, m, c)]['degr'][0]:.3f}" if (ds, H, m, c) in stats else '–'
                             for c in pert])
        table(['Dataset', 'H', 'Method'] + pert, rows)

    # ablation: joint training vs plug-in
    if any(k[2] == 'Ours' for k in stats):
        md.append('### Ablation — Ours (joint: backbone also fine-tuned) vs Ours-plugin (backbone frozen), MSE')
        hdr = ['Dataset', 'H'] + [f'{c}: plugin / joint' for c in conditions]
        rows = []
        for ds, H in keys:
            if (ds, H, 'Ours', 'clean') not in stats:
                continue
            row = [ds, str(H)]
            for c in conditions:
                sp, sj = stats.get((ds, H, OURS, c)), stats.get((ds, H, 'Ours', c))
                row.append(f"{sp['mse'][0]:.4f} / {sj['mse'][0]:.4f}" if sp and sj else '–')
            rows.append(row)
        table(hdr, rows)

    rows = []
    for r in ex_rows:
        if r['method'] not in ('Ours', 'Ours-plugin'):
            continue

        def f3(k):
            v = r.get(k, float('nan'))
            return '–' if v != v else f'{v:.3f}'

        rows.append([r['dataset'], str(r['pred_len']), METHOD_LABEL[r['method']], f3('mean_alpha_clean'),
                     f3('mean_alpha_T1_raw'), f3('frac_alpha_gt_0.5_clean'), f3('frac_alpha_gt_0.5_T1_raw'),
                     f3('alpha_last_point_T1_raw'), f3('T1_last_point_auroc'),
                     '/'.join(f3(f'pi_{n}_clean') for n in ('repairer', 'localmedian', 'cycleprior')),
                     '/'.join(f3(f'pi_{n}_T1_raw') for n in ('repairer', 'localmedian', 'cycleprior')),
                     f3('time_ratio_vs_B0'), r.get('mean_dbar_per_patch_clean', '')])
    if rows:
        md.append('### RADAR diagnostics (mean over seeds; π = Repairer/LocalMedian/CyclePrior)')
        table(['Dataset', 'H', 'Method', 'ᾱ clean', 'ᾱ T1', 'α>0.5 clean', 'α>0.5 T1', 'α last pt T1',
               'AUROC T1 last', 'π clean', 'π T1', 'time ×B0', 'mean d̄ per patch (oldest→newest)'], rows)

    if repro:
        md.append('### S0 reproduction check (Exp_Radar S0 vs original run.py, same seed)')
        table(['Setting', 'run.py MSE', 'S0 MSE', 'rel. diff', 'pass (<2%)'],
              [[r['setting'], f"{r['orig_mse']:.6f}", f"{r['s0_mse']:.6f}", f"{100 * r['rel_diff_mse']:.3f}%",
                'yes' if r['passed'] else 'NO'] for r in repro])

    text = '\n'.join(md)
    with open(os.path.join(results_dir, 'summary.md'), 'w') as f:
        f.write(text + '\n')
    if print_tables:
        print(text)
    print(f'wrote {csv_fn}, {ex_fn}, {os.path.join(results_dir, "summary.md")}')
