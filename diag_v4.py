"""RADAR v4 diagnostics on the test grid (VALIDATION split by default; --diag_split test to change).

    python diag_v4.py revin   <run_radar.py arguments>   step 1: RevIN 2×2 on B0 (needs base.pth)
    python diag_v4.py experts <run_radar.py arguments>   step 2: offline expert repair errors (no training needed;
                                                          E1 is added when pre_<arch>.pth exists)
    python diag_v4.py det     <run_radar.py arguments>   step 3: detector AUROC per condition (needs pre_<arch>.pth;
                                                          --det_feats v3 evaluates the v3-feature detector)

revin   For every corrupted window, four GTR (B0) forecasts:
          Clean        values clean,     RevIN statistics clean      (upper bound)
          Raw          values corrupted, statistics corrupted        (no repair)
          Stats-only   values clean,     statistics corrupted        (damage through the RevIN mean/std only)
          Direct-only  values corrupted, statistics clean            (damage through the values only)
        Shares = (variant − Clean) / (Raw − Clean). Only models/GTR.py (use_revin=1) is supported.
experts Oracle detection: R = 1 − mask (the corrupted points are known), so the numbers isolate repair quality.
        Error = |c − x_clean| / s_clean on the corrupted points (s_clean = robust scale of the clean window), plus
        the B0 forecast MSE after replacing the corrupted points by each expert (when base.pth exists).
det     Detector only (S2 checkpoint, no gate): AUROC of ℓ with positives = corrupted points of the condition and
        negatives = all points of the clean windows; recall / false-alarm rate at R < 0.5 (ℓ > 0).
Results: <results_dir>/diag/<setting>__<kind>.json and .md
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

import radar.compat  # noqa: F401
from run_radar import _conditions, build_parser, set_seed

KINDS = ('revin', 'experts', 'det')


# ----------------------------------------------------------------------------------------------------------------
def revin_forward(model, x_vals, x_stats, cycle):
    """models/GTR.py forward on x_vals with the RevIN statistics of x_stats (identical ops to Model.forward)."""
    mean = torch.mean(x_stats, dim=1, keepdim=True)
    var = torch.var(x_stats, dim=1, keepdim=True) + 1e-5
    flag = model.use_revin
    model.use_revin = 0
    try:
        out = model((x_vals - mean) / torch.sqrt(var), cycle)
    finally:
        model.use_revin = flag
    return out * torch.sqrt(var) + mean


def _groups(conds):
    """cond -> (type, pos) and (type, pos, k) keys (clean excluded)."""
    from radar.perturb import parse_condition
    g2, g3 = defaultdict(list), defaultdict(list)
    for c in conds:
        sp = parse_condition(c)
        if sp is None:
            continue
        g2[(sp['type'], sp['pos'])].append(c)
        g3[(sp['type'], sp['pos'], sp['k'])].append(c)
    return g2, g3


def _avg(per_cond, conds, key):
    vals = [per_cond[c][key] for c in conds if c in per_cond and per_cond[c].get(key) == per_cond[c].get(key)]
    return float(np.mean(vals)) if vals else float('nan')


def _f(v, nd=4):
    return '–' if v is None or v != v else f'{v:.{nd}f}'


def _pct(v):
    return '–' if v is None or v != v else f'{100 * v:.1f}%'


def _table(md, header, rows):
    md.append('| ' + ' | '.join(header) + ' |')
    md.append('|' + '|'.join(['---'] * len(header)) + '|')
    for r in rows:
        md.append('| ' + ' | '.join(r) + ' |')
    md.append('')


# ----------------------------------------------------------------------------------------------------------------
def diag_revin(exp, conds, split):
    args = exp.args
    from exp.exp_radar import ckpt_paths
    paths = ckpt_paths(args)
    b0, _ = exp.load_method('B0', paths)
    if b0 is None:
        raise SystemExit(f'missing {paths["base"]}: run train_base first')
    if not hasattr(b0, 'use_revin') or args.backbone != 'GTR':
        raise SystemExit('revin diagnostic supports the GTR backbone (models/GTR.py) only')
    if not b0.use_revin:
        raise SystemExit('this dataset runs GTR with --use_revin 0: the RevIN decomposition does not apply')
    H, fd = args.pred_len, exp.f_dim
    per = {}
    with torch.no_grad():
        for cond in conds:
            if cond == 'clean':
                continue
            se = defaultdict(float)
            n = 0
            for x, x_p, changed, y, c in exp.perturbed_batches(cond, split):
                yy = y[:, -H:, fd:]
                outs = dict(clean=b0(x, c), raw=b0(x_p, c), stats_only=revin_forward(b0, x, x_p, c),
                            direct_only=revin_forward(b0, x_p, x, c))
                for k, o in outs.items():
                    se[k] += ((o[:, -H:, fd:] - yy) ** 2).sum().item()
                n += yy.numel()
            r = {k: v / n for k, v in se.items()}
            dr = r['raw'] - r['clean']
            r.update(rel_raw=r['raw'] / r['clean'] - 1.0,
                     share_stats=(r['stats_only'] - r['clean']) / dr if abs(dr) > 1e-12 else float('nan'),
                     share_direct=(r['direct_only'] - r['clean']) / dr if abs(dr) > 1e-12 else float('nan'))
            per[cond] = r
            print(f"  {cond:20s} clean {r['clean']:.5f} raw {r['raw']:.5f} stats-only {r['stats_only']:.5f} "
                  f"direct-only {r['direct_only']:.5f}")
    md = []
    g2, g3 = _groups(conds)
    hdr = ['Type', 'Position', 'Clean', 'Raw', 'Stats-only', 'Direct-only', 'Raw vs Clean', 'stats share',
           'direct share']

    def row(key, cs):
        cl, ra = _avg(per, cs, 'clean'), _avg(per, cs, 'raw')
        so, do = _avg(per, cs, 'stats_only'), _avg(per, cs, 'direct_only')
        d = ra - cl
        return [*key, _f(cl), _f(ra), _f(so), _f(do), _pct(ra / cl - 1.0),
                _pct((so - cl) / d) if abs(d) > 1e-12 else '–', _pct((do - cl) / d) if abs(d) > 1e-12 else '–']

    md.append('### RevIN 2×2 (B0 forecast MSE; mean over lengths and channel ratios)')
    _table(md, hdr, [row((t, p), cs) for (t, p), cs in g2.items()])
    md.append('### RevIN 2×2 by length')
    _table(md, ['Type', 'Position', 'k'] + hdr[2:], [row((t, p, str(k)), cs) for (t, p, k), cs in g3.items()])
    return dict(per_condition=per), md


# ----------------------------------------------------------------------------------------------------------------
def diag_experts(exp, conds, split):
    args = exp.args
    from exp.exp_radar import ckpt_paths, load_pre_state
    from radar.modules import (CrossChannelExpert, LocalLinearExpert, LocalMedianExpert, PeriodicShapeExpert,
                               ReliableNeighborExpert, reliable_context, robust_stats, to_series)
    from radar.radar import resolve_e3_period
    paths = ckpt_paths(args)
    b0, _ = exp.load_method('B0', paths)
    W = None
    if os.path.exists(paths['pre']):
        W = exp.build_wrapper().to(exp.device)
        load_pre_state(W, paths['pre'], exp.device)
        W.eval()
    dev = exp.device
    T = resolve_e3_period(args)
    e2_old = ReliableNeighborExpert(k=args.e2_k, thr=0.5, trend=False)
    e2 = LocalLinearExpert()
    e3 = PeriodicShapeExpert(T, args.seq_len, args.e3_cycles).to(dev)
    e4 = CrossChannelExpert(k=getattr(args, 'e4_k', 3))
    med = LocalMedianExpert(3)
    names = ['no repair', 'local median', 'E2 v3 (median)', 'E2 v4 (linear)', 'E3 (periodic)', 'E3 else E2',
             'best of E2/E3', 'E4 (cross-channel)', 'E4 else E2', 'best of E2/E3/E4']
    e1_name = None
    if W is not None:
        e1_name = f'E1 refined ({W.arch}: stage 1 + residual)' if W.arch in ('v5', 'v6') else 'E1 (Repairer)'
        names.append(e1_name)
    only_avail = {'E3 (periodic)': 'a3', 'E4 (cross-channel)': 'a4'}  # MAE on the points where the expert exists
    H, fd = args.pred_len, exp.f_dim
    per = {}
    with torch.no_grad():
        for cond in conds:
            if cond == 'clean':
                continue
            ae = defaultdict(float)
            sq = defaultdict(float)
            fse = defaultdict(float)
            n_pts, n_y, n_a3, n_a4 = 0, 0, 0, 0
            for x, x_p, changed, y, c in exp.perturbed_batches(cond, split):
                R = (~changed).to(x.dtype)  # oracle reliability
                ctx = reliable_context(to_series(R) > 0.5, args.e2_k)
                c2 = e2(x_p, ctx)
                c3, a3, _ = e3(x_p, ctx, fill=c2)
                c4, a4, _ = e4(x_p, ~changed, fill=c2)
                cands = {'no repair': x_p, 'local median': med(x_p), 'E2 v3 (median)': e2_old(x_p, R),
                         'E2 v4 (linear)': c2, 'E3 (periodic)': c3, 'E3 else E2': torch.where(a3, c3, c2),
                         'E4 (cross-channel)': c4, 'E4 else E2': c4}
                best = torch.where(a3 & ((c3 - x).abs() < (c2 - x).abs()), c3, c2)
                cands['best of E2/E3'] = best
                cands['best of E2/E3/E4'] = torch.where(a4 & ((c4 - x).abs() < (best - x).abs()), c4, best)
                masks = {'a3': a3, 'a4': a4}
                if W is not None:
                    cands[e1_name] = W.repairer_candidate(x_p, R)
                _, sc = robust_stats(x)
                sc = sc.expand_as(x)
                yy = y[:, -H:, fd:]
                for nm, cv in cands.items():
                    msk = changed & masks[only_avail[nm]] if nm in only_avail else changed
                    e = ((cv - x).abs() / sc)[msk]
                    ae[nm] += e.sum().item()
                    sq[nm] += (e ** 2).sum().item()
                    if b0 is not None:
                        out = b0(torch.where(changed, cv, x_p), c)[:, -H:, fd:]
                        fse[nm] += ((out - yy) ** 2).sum().item()
                if b0 is not None:
                    fse['oracle clean'] += ((b0(x, c)[:, -H:, fd:] - yy) ** 2).sum().item()
                n_pts += int(changed.sum().item())
                n_a3 += int((changed & a3).sum().item())
                n_a4 += int((changed & a4).sum().item())
                n_y += yy.numel()
            r = dict(n_changed=n_pts, e3_avail=n_a3 / max(n_pts, 1), e4_avail=n_a4 / max(n_pts, 1))
            for nm in names:
                nn_ = {'E3 (periodic)': n_a3, 'E4 (cross-channel)': n_a4}.get(nm, n_pts)
                r[nm] = dict(mae=ae[nm] / max(nn_, 1), rmse=float(np.sqrt(sq[nm] / max(nn_, 1))),
                             fcst_mse=fse[nm] / n_y if b0 is not None else float('nan'))
            if b0 is not None:
                r['oracle clean'] = dict(fcst_mse=fse['oracle clean'] / n_y)
            per[cond] = r
            print(f'  {cond:20s} ' + ' | '.join(f'{nm} {r[nm]["mae"]:.3f}' for nm in names)
                  + f' | E3 avail {100 * r["e3_avail"]:.0f}% | E4 avail {100 * r["e4_avail"]:.0f}%')
    md = []
    g2, g3 = _groups(conds)

    def avg(cs, nm, key):
        v = [per[c][nm][key] for c in cs if c in per and per[c][nm][key] == per[c][nm][key]]
        return float(np.mean(v)) if v else float('nan')

    for key, title in (('mae', 'repair MAE on corrupted points (robust units of the clean window)'),
                       ('fcst_mse', 'B0 forecast MSE with the corrupted points replaced (oracle detection)')):
        if key == 'fcst_mse' and b0 is None:
            continue
        md.append(f'### Experts — {title}; E3 period T = {T}' + (' (E3 unavailable: T ≥ L)' if not e3.enabled else ''))
        hdr = ['Type', 'Position'] + names + (['clean input'] if key == 'fcst_mse' else []) + ['E3 avail', 'E4 avail']
        rows = []
        for (t, p), cs in g2.items():
            row = [t, p] + [_f(avg(cs, nm, key), 3) for nm in names]
            if key == 'fcst_mse':
                row.append(_f(float(np.mean([per[c]['oracle clean']['fcst_mse'] for c in cs])), 3))
            row.append(_pct(float(np.mean([per[c]['e3_avail'] for c in cs]))))
            row.append(_pct(float(np.mean([per[c]['e4_avail'] for c in cs]))))
            rows.append(row)
        _table(md, hdr, rows)
    md.append('### Experts — repair MAE by length')
    rows = [[t, p, str(k)] + [_f(avg(cs, nm, 'mae'), 3) for nm in names] for (t, p, k), cs in g3.items()]
    _table(md, ['Type', 'Position', 'k'] + names, rows)
    return dict(per_condition=per, e3_period=T, e3_enabled=bool(e3.enabled), experts=names), md


# ----------------------------------------------------------------------------------------------------------------
def diag_det(exp, conds, split):
    args = exp.args
    from exp.exp_radar import HIST_BINS, HIST_HI, HIST_LO, _hist, ckpt_paths, hist_auroc, load_pre_state
    paths = ckpt_paths(args)
    if not os.path.exists(paths['pre']):
        raise SystemExit(f'missing {paths["pre"]}: run pretrain_radar (S2) first')
    W = exp.build_wrapper().to(exp.device)
    load_pre_state(W, paths['pre'], exp.device)
    W.eval()
    edges = np.linspace(HIST_LO, HIST_HI, HIST_BINS + 1)
    above0 = edges[:-1] >= 0.0  # bins with ℓ ≥ 0  (R ≤ 0.5)
    hists = {}
    with torch.no_grad():
        for cond in conds:
            h_all, h_pos = np.zeros(HIST_BINS), np.zeros(HIST_BINS)
            for x, x_p, changed, y, c in exp.perturbed_batches(cond, split):
                W.repair(x_p, c, compute_dep=False, need_route=False)
                lg = W.last_aux['logit']
                h_all += _hist(lg)
                if changed.any():
                    h_pos += _hist(lg[changed])
            hists[cond] = (h_all, h_pos)
    neg = hists['clean'][0]
    far = float(neg[above0].sum() / max(neg.sum(), 1))
    per = {}
    for cond, (h_all, h_pos) in hists.items():
        if cond == 'clean':
            continue
        per[cond] = dict(auroc=hist_auroc(h_pos, neg), recall=float(h_pos[above0].sum() / max(h_pos.sum(), 1)),
                         n_pos=int(h_pos.sum()))
        print(f"  {cond:20s} AUROC {per[cond]['auroc']:.4f}  recall@R<0.5 {100 * per[cond]['recall']:.1f}%")
    md = [f'### Detector ({args.det_feats} features, {os.path.basename(paths["pre"])}) — AUROC vs clean points; '
          f'false-alarm rate on clean points at R < 0.5: {100 * far:.2f}%']
    g2, g3 = _groups(conds)
    _table(md, ['Type', 'Position', 'AUROC', 'recall @ R<0.5'],
           [[t, p, _f(_avg(per, cs, 'auroc')), _pct(_avg(per, cs, 'recall'))] for (t, p), cs in g2.items()])
    md.append('### Detector AUROC by length')
    _table(md, ['Type', 'Position', 'k', 'AUROC', 'recall @ R<0.5'],
           [[t, p, str(k), _f(_avg(per, cs, 'auroc')), _pct(_avg(per, cs, 'recall'))]
            for (t, p, k), cs in g3.items()])
    return dict(per_condition=per, false_alarm_clean=far, det_feats=args.det_feats,
                ckpt=os.path.basename(paths['pre'])), md


# ----------------------------------------------------------------------------------------------------------------
def main():
    if len(sys.argv) < 2 or sys.argv[1] not in KINDS:
        raise SystemExit(f'usage: python diag_v4.py {{{",".join(KINDS)}}} <run_radar.py arguments>')
    kind = sys.argv[1]
    args = build_parser().parse_args(['eval'] + sys.argv[2:])
    if args.quick:
        args.pred_len, args.random_seed = 96, 2024
    if args.use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    from exp.exp_radar import Exp_Radar, setting_name
    set_seed(args.random_seed)
    exp = Exp_Radar(args, use_radar=False)
    setting = setting_name(args)
    conds = _conditions(args)
    split = args.diag_split
    print(f'=== diag {kind} | {setting} | split {split} | {len(conds) - 1} conditions ===')
    fn = {'revin': diag_revin, 'experts': diag_experts, 'det': diag_det}[kind]
    res, md = fn(exp, conds, split)
    res.update(kind=f'diag_{kind}', setting=setting, split=split, grid_seed=args.grid_seed)
    d = os.path.join(args.results_dir, 'diag')
    os.makedirs(d, exist_ok=True)
    tag = f'{kind}_det{args.det_feats}' if kind == 'det' else kind
    base = os.path.join(d, f'{setting}__{tag}')
    with open(base + '.json', 'w') as f:
        json.dump(res, f, indent=1)
    text = f'## {setting} — {kind} ({split} split)\n\n' + '\n'.join(md)
    with open(base + '.md', 'w') as f:
        f.write(text + '\n')
    print('\n' + text)
    print(f'wrote {base}.json / .md')


if __name__ == '__main__':
    main()
