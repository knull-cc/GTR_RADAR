"""Clean-input headroom & gate audit for RADAR (VALIDATION split only; no training, test set untouched).

1) Oracle headroom. On CLEAN validation windows: how much could GTR (B0) improve if, for every
   (window, channel), at most one input patch could be fully replaced by a repair value and an oracle
   that knows the ground truth picked the best option (or no repair)?
     repair 'radar'  : Ours-plugin repair target Σπ·c   (needs plugin.pth)
     repair 'median' : parameter-free local median of t-3..t+3 without t
   Each repair has a '_null' control: the same edits shuffled across windows (same size, no information).
   Picking the minimum of 13 options always looks good on its own (selection bias); the control measures
   that bias, so  real headroom = oracle(repair) - oracle(null).
2) Gate audit. Ours-plugin (backbone == B0, so every difference comes from RADAR) and Ours on clean val:
   (window, channel) pairs where the gate fires (any α > 0.5) vs not -> B0 error, model error, % improved.

Usage: python diag_clean.py <same data/model args as run_radar.py>
"""
import json
import os
import sys

import numpy as np
import torch

import radar.compat  # noqa: F401
from run_radar import build_parser, set_seed


def _gate_update(st, e0, em, alpha, f_dim):
    fire = (alpha > 0.5).any(dim=1)  # (B, C)
    if f_dim:
        fire = fire[:, f_dim:]
    better = em < e0
    st['em'] += em.sum().item()
    for key, msk in (('fire', fire), ('rest', ~fire)):
        st[key + '_e0'] += e0[msk].sum().item()
        st[key + '_em'] += em[msk].sum().item()
        st[key + '_n'] += int(msk.sum().item())
        st[key + '_better'] += int((better & msk).sum().item())


def main():
    args = build_parser().parse_args(['eval'] + sys.argv[1:])
    if args.use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    from exp.exp_radar import Exp_Radar, ckpt_paths, setting_name
    from radar.modules import LocalMedianExpert

    set_seed(args.random_seed)
    exp = Exp_Radar(args, use_radar=False)
    paths, setting = ckpt_paths(args), setting_name(args)
    b0, _ = exp.load_method('B0', paths)
    if b0 is None:
        raise SystemExit(f'missing {paths["base"]}: run train_base first')
    plugin, _ = exp.load_method('Ours-plugin', paths)
    ours, _ = exp.load_method('Ours', paths)
    _, loader = exp.ordered_loader('val')

    dev, H, fd, P, L = exp.device, args.pred_len, exp.f_dim, args.radar_patch, args.seq_len
    N = -(-L // P)
    pad = N * P - L
    point_patch = (torch.arange(L, device=dev) + pad) // P
    med = LocalMedianExpert(3)
    g = torch.Generator().manual_seed(args.random_seed + 999)

    def err(model, x, c, y):
        return ((model(x, c)[:, -H:, fd:] - y) ** 2).mean(dim=1)  # (B, C')

    rep_names = (['radar'] if plugin is not None else []) + ['median']
    all_names = rep_names + [n + '_null' for n in rep_names]
    acc = {n: dict(oracle_all=0.0, oracle_last=0.0, always_last=0.0, n_helped=0, pick=np.zeros(N + 1))
           for n in all_names}
    tot = dict(e0=0.0, n=0)
    models = [(m, mod) for m, mod in (('Ours-plugin', plugin), ('Ours', ours)) if mod is not None]
    gate = {m: dict(em=0.0, fire_e0=0.0, fire_em=0.0, fire_n=0, fire_better=0,
                    rest_e0=0.0, rest_em=0.0, rest_n=0, rest_better=0) for m, _ in models}
    for _, mod in models:
        exp._set_probe_seed(mod, args.random_seed)

    with torch.no_grad():
        for bx, by, _, _, bc in loader:
            x, y, c = exp._to_dev(bx, by, bc)
            y = y[:, -H:, fd:]
            B = x.shape[0]
            e0 = err(b0, x, c, y)
            tot['e0'] += e0.sum().item()
            tot['n'] += e0.numel()

            reps = {}
            for m, mod in models:
                em = err(mod, x, c, y)
                aux = mod.last_aux
                if m == 'Ours-plugin':
                    reps['radar'] = aux['target'].detach()
                _gate_update(gate[m], e0, em, aux['alpha'], fd)
            reps['median'] = med(x)
            perm = torch.randperm(B, generator=g).to(dev)
            for n in rep_names:
                reps[n + '_null'] = x + (reps[n] - x)[perm]

            for n, r in reps.items():
                e1 = torch.stack([err(b0, torch.where((point_patch == p).view(1, L, 1), r, x), c, y)
                                  for p in range(N)])  # (N, B, C')
                best, idx = torch.cat([e0.unsqueeze(0), e1], dim=0).min(dim=0)
                a = acc[n]
                a['oracle_all'] += best.sum().item()
                a['oracle_last'] += torch.minimum(e0, e1[-1]).sum().item()
                a['always_last'] += e1[-1].sum().item()
                a['n_helped'] += int((best < 0.98 * e0).sum().item())
                a['pick'] += np.bincount(idx.flatten().cpu().numpy(), minlength=N + 1)

    n = tot['n']
    base = tot['e0'] / n
    rel = lambda v: (base - v) / base  # noqa: E731  positive = better than B0
    out = dict(kind='clean_diag', setting=setting, split='val', b0_clean_val_mse=base, n_window_channel=n,
               headroom={}, gate={})
    print(f'\n=== {setting} | CLEAN validation set | B0 MSE {base:.6f} ({n} window×channel) ===')
    print('\n[1] Oracle headroom of repairing ONE patch (MSE change vs B0; + = better)')
    print('| repair | always repair last patch | oracle: last patch or none | oracle: any one patch or none '
          '| pairs helped >2% |')
    print('|---|---|---|---|---|')
    for nm in all_names:
        a = acc[nm]
        row = dict(always_last=rel(a['always_last'] / n), oracle_last=rel(a['oracle_last'] / n),
                   oracle_any=rel(a['oracle_all'] / n), frac_helped=a['n_helped'] / n,
                   oracle_pick_hist=(a['pick'] / a['pick'].sum()).tolist())
        out['headroom'][nm] = row
        print(f"| {nm} | {100 * row['always_last']:+.2f}% | {100 * row['oracle_last']:+.2f}% | "
              f"{100 * row['oracle_any']:+.2f}% | {100 * row['frac_helped']:.1f}% |")
    for nm in rep_names:
        net_any = out['headroom'][nm]['oracle_any'] - out['headroom'][nm + '_null']['oracle_any']
        net_last = out['headroom'][nm]['oracle_last'] - out['headroom'][nm + '_null']['oracle_last']
        out['headroom'][nm]['net_oracle_any'] = net_any
        out['headroom'][nm]['net_oracle_last'] = net_last
        print(f'net headroom ({nm} − null): any patch {100 * net_any:+.2f}% | last patch {100 * net_last:+.2f}%')

    if models:
        print('\n[2] Gate audit on clean val (fire = any α>0.5 in the window/channel; MSE + = better than B0)')
        print('| model | overall vs B0 | fired pairs | fired: vs B0 | fired: % improved | not fired: vs B0 '
              '| not fired: % improved |')
        print('|---|---|---|---|---|---|---|')
    for m, _ in models:
        st = gate[m]
        fr = lambda a, b: (a - b) / a if a > 0 else float('nan')  # noqa: E731
        row = dict(overall=rel(st['em'] / n), fired_frac=st['fire_n'] / n,
                   fired_rel=fr(st['fire_e0'], st['fire_em']),
                   fired_improved=st['fire_better'] / max(st['fire_n'], 1),
                   rest_rel=fr(st['rest_e0'], st['rest_em']),
                   rest_improved=st['rest_better'] / max(st['rest_n'], 1))
        out['gate'][m] = row
        print(f"| {m} | {100 * row['overall']:+.2f}% | {100 * row['fired_frac']:.1f}% | "
              f"{100 * row['fired_rel']:+.2f}% | {100 * row['fired_improved']:.1f}% | "
              f"{100 * row['rest_rel']:+.2f}% | {100 * row['rest_improved']:.1f}% |")
    if 'Ours' in gate:
        print('(Ours also fine-tunes the backbone, so its "not fired" column mixes in backbone changes.)')

    d = os.path.join(args.results_dir, 'diag')
    os.makedirs(d, exist_ok=True)
    fn = os.path.join(d, f'{setting}__clean_diag.json')
    with open(fn, 'w') as f:
        json.dump(out, f, indent=1)
    print(f'\nwrote {fn}')


if __name__ == '__main__':
    main()
