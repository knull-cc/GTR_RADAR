"""Figures for RADAR (matplotlib, saved to figures_radar/).

1. dependency_<dataset>_pl<H>.png : mean d̄ per patch (oldest -> newest) from the eval JSONs (clean test set).
2. examples_<dataset>_pl<H>_seed<s>_<condition>_w<i>.png : clean vs corrupted input (test-grid condition),
   RADAR-repaired input, the E2 / E3 candidates, per-point α and π strips, and B0 / B1 / Ours-plugin forecasts.
"""
import glob
import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

# reference categorical palette (fixed order) + neutral ink
INK = '#52514e'
GRID = '#d9d8d4'
C_OURS = '#2a78d6'   # slot 1 blue   : RADAR (repaired input, Ours forecast)
C_B0 = '#eb6834'     # slot 2 orange : B0
C_B1 = '#1baf7a'     # slot 3 aqua   : B1
C_PERT = '#e87ba4'   # slot 5 magenta: perturbed input
C_CYC = '#6250d6'    # slot 7 violet : local linear expert (E2)
C_E3 = '#c9a227'     # ochre         : periodic shape expert (E3)


def _style(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def plot_dependency_profiles(results_dir, fig_dir, datasets=('ETTh1', 'weather')):
    os.makedirs(fig_dir, exist_ok=True)
    made = []
    files = glob.glob(os.path.join(results_dir, 'raw', '*.json'))
    runs = []
    for fn in files:
        try:
            r = json.load(open(fn))
        except Exception:
            continue
        if r.get('kind') == 'eval' and r.get('method') in ('Ours', 'Ours-plugin') and \
                'clean' in r.get('extras', {}):
            runs.append(r)
    for ds in datasets:
        for H in sorted({r['pred_len'] for r in runs if r['dataset'].lower() == ds.lower()}):
            fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=150)
            any_bar = False
            methods = [m for m in ('Ours', 'Ours-plugin')
                       if any(r['method'] == m and r['dataset'].lower() == ds.lower() and r['pred_len'] == H
                              for r in runs)]
            width = 0.8 / max(len(methods), 1)
            for j, m in enumerate(methods):
                prof = [r['extras']['clean']['mean_dbar_per_patch'] for r in runs
                        if r['method'] == m and r['dataset'].lower() == ds.lower() and r['pred_len'] == H]
                p = np.mean(prof, axis=0)
                xs = np.arange(1, len(p) + 1) + (j - (len(methods) - 1) / 2) * width
                ax.bar(xs, p, width=width * 0.92, color=C_OURS if m == 'Ours' else C_CYC,
                       label=f'{m} (n={len(prof)} seed{"s" if len(prof) > 1 else ""})')
                any_bar = True
            if not any_bar:
                plt.close(fig)
                continue
            _style(ax)
            ax.set_xticks(np.arange(1, len(p) + 1))
            ax.set_xlabel('input patch (1 = oldest, rightmost = most recent)', color=INK, fontsize=9)
            ax.set_ylabel(r'mean dependency $\tilde{D}$ (units of $\bar{D}$)', color=INK, fontsize=9)
            ax.set_ylim(bottom=0)
            ax.set_title(f'{ds}, H={H}: forecast dependency per input patch (clean test set)', fontsize=10,
                         color='#0b0b0b')
            if len(methods) > 1:
                ax.legend(frameon=False, fontsize=8)
            fn = os.path.join(fig_dir, f'dependency_{ds}_pl{H}.png')
            fig.tight_layout()
            fig.savefig(fn)
            plt.close(fig)
            made.append(fn)
    return made


def plot_examples(exp, fig_dir, n_examples=3, conditions=('zero_tail_k4_c100', 'flat_int_k8_c100')):
    """exp: Exp_Radar for the current setting (checkpoints must exist)."""
    from exp.exp_radar import ckpt_paths, setting_name
    from radar.perturb import apply_condition, condition_rng

    args = exp.args
    paths = ckpt_paths(args)
    models = {}
    for m in ('B0', 'B1', 'Ours-plugin', 'Ours'):
        model, _ = exp.load_method(m, paths)
        if model is not None:
            models[m] = model
    ours = 'Ours-plugin' if 'Ours-plugin' in models else ('Ours' if 'Ours' in models else None)
    if ours is None:
        print('[plot] no Ours-plugin / Ours checkpoint for this setting; example windows skipped')
        return []
    os.makedirs(fig_dir, exist_ok=True)
    test_ds, _ = exp.get_data('test')
    n = len(test_ds)
    idxs = [int(n * f) for f in np.linspace(0.2, 0.8, n_examples)]
    rz = exp.raw_zero('raw')
    ch = -1  # target / last channel ('OT' for ETT & Weather)
    made = []
    W = models[ours]
    for cond in conditions:
        for k, idx in enumerate(idxs):
            bx, by, _, _, bc = test_ds[idx]
            x = torch.as_tensor(np.asarray(bx)).float().unsqueeze(0).to(exp.device)
            y = torch.as_tensor(np.asarray(by)).float().unsqueeze(0).to(exp.device)
            c = exp.single_condition(bc)
            x_p, _ = apply_condition(x, cond, rz, args.zero_mode, condition_rng(args.grid_seed, cond, 10_000 + idx),
                                     guard=args.radar_patch)
            preds = {}
            with torch.no_grad():
                exp._set_probe_seed(W, args.random_seed)
                preds[ours] = W(x_p, c)[0, -args.pred_len:, ch].cpu().numpy()
                aux = {kk: v.detach() for kk, v in W.last_aux.items() if torch.is_tensor(v)}
                for m in ('B0', 'B1'):
                    if m in models:
                        preds[m] = models[m](x_p, c)[0, -args.pred_len:, ch].cpu().numpy()
            L = x.shape[1]
            t_in = np.arange(-L + 1, 1)
            t_out = np.arange(1, args.pred_len + 1)
            xc = x[0, :, ch].cpu().numpy()
            xp = x_p[0, :, ch].cpu().numpy()
            xr = aux['x_tilde'][0, :, ch].cpu().numpy()
            c2 = aux['c'][0, :, ch, 1].cpu().numpy()
            c3 = aux['c'][0, :, ch, 2].cpu().numpy()
            a3 = aux.get('cand_avail', aux['avail'])[0, :, ch, 2].cpu().numpy()
            alpha = aux['alpha'][0, :, ch].cpu().numpy()
            pi = aux['pi'][0, :, ch, :].cpu().numpy().T  # (K, L)
            truth = y[0, -args.pred_len:, ch].cpu().numpy()

            fig = plt.figure(figsize=(10, 7.0), dpi=150)
            gs = fig.add_gridspec(4, 2, height_ratios=[3, 0.45, 0.8, 3], width_ratios=[1, 0.025], hspace=0.5,
                                  wspace=0.03)
            ax1 = fig.add_subplot(gs[0, 0])
            ax_a = fig.add_subplot(gs[1, 0], sharex=ax1)
            ax_cb = fig.add_subplot(gs[1, 1])
            ax_p = fig.add_subplot(gs[2, 0], sharex=ax1)
            ax2 = fig.add_subplot(gs[3, 0])

            ax1.plot(t_in, xc, color=INK, lw=1.6, label='clean input')
            ax1.plot(t_in, xp, color=C_PERT, lw=1.6, ls='--', label=f'corrupted input ({cond})')
            ax1.plot(t_in, xr, color=C_OURS, lw=1.8, label=r'RADAR repaired $\tilde{x}$')
            ax1.plot(t_in, c2, color=C_CYC, lw=1.1, ls=':', label=r'E2 local linear $c_2$')
            if a3.any():
                ax1.plot(t_in, np.where(a3, c3, np.nan), color=C_E3, lw=1.1, ls='-.', label=r'E3 periodic $c_3$')
            _style(ax1)
            ax1.set_title(f'{setting_name(args)} | test window {idx} | channel {ch % x.shape[2]}', fontsize=9,
                          color='#0b0b0b')
            ax1.set_ylabel('scaled value', color=INK, fontsize=9)
            ax1.legend(frameon=False, fontsize=7.5, ncol=3, loc='upper left')
            plt.setp(ax1.get_xticklabels(), visible=False)

            im = ax_a.imshow(alpha[None, :], aspect='auto', cmap='Blues', vmin=0, vmax=1,
                             extent=[t_in[0] - 0.5, t_in[-1] + 0.5, 0, 1])
            ax_a.set_yticks([])
            ax_a.set_ylabel(r'$\alpha$', color=INK, fontsize=9, rotation=0, labelpad=10)
            ax_a.tick_params(colors=INK, labelsize=8)
            plt.setp(ax_a.get_xticklabels(), visible=False)
            for sp in ax_a.spines.values():
                sp.set_color(GRID)
            cb = fig.colorbar(im, cax=ax_cb)
            cb.ax.tick_params(labelsize=7, colors=INK)
            cb.outline.set_edgecolor(GRID)

            ax_p.imshow(pi, aspect='auto', cmap='Greys', vmin=0, vmax=1, interpolation='nearest',
                        extent=[t_in[0] - 0.5, t_in[-1] + 0.5, pi.shape[0] - 0.5, -0.5])
            ax_p.set_yticks(range(pi.shape[0]))
            ax_p.set_yticklabels([f'π {n}' for n in W.expert_names], fontsize=7, color=INK)
            ax_p.set_xlabel('input time step (0 = most recent)', color=INK, fontsize=8)
            ax_p.tick_params(colors=INK, labelsize=8)
            for sp in ax_p.spines.values():
                sp.set_color(GRID)

            hist = min(48, L)
            ax2.plot(np.arange(-hist + 1, 1), xc[-hist:], color=INK, lw=1.0, alpha=0.6)
            ax2.plot(t_out, truth, color=INK, lw=1.8, label='ground truth')
            for m, col in (('B0', C_B0), ('B1', C_B1), (ours, C_OURS)):
                if m in preds:
                    mse = float(np.mean((preds[m] - truth) ** 2))
                    ax2.plot(t_out, preds[m], color=col, lw=1.6, label=f'{m} (ch. MSE {mse:.3f})')
            ax2.axvline(0.5, color=GRID, lw=1.0)
            _style(ax2)
            ax2.set_xlabel('forecast step', color=INK, fontsize=9)
            ax2.set_ylabel('scaled value', color=INK, fontsize=9)
            ax2.legend(frameon=False, fontsize=7.5, ncol=2, loc='upper left')

            fn = os.path.join(fig_dir, f'examples_{args.model_id.split("_")[0]}_pl{args.pred_len}_seed'
                                       f'{args.random_seed}_{cond}_w{k + 1}.png')
            fig.savefig(fn, bbox_inches='tight')
            plt.close(fig)
            made.append(fn)
    return made


def plot_all(args):
    datasets = tuple(d.strip() for d in args.plot_datasets.split(',') if d.strip())
    made = plot_dependency_profiles(args.results_dir, args.fig_dir, datasets)
    from exp.exp_radar import Exp_Radar
    exp = Exp_Radar(args, use_radar=False)
    conds = tuple(c.strip() for c in getattr(args, 'plot_conditions', 'zero_tail_k4_c100').split(',') if c.strip())
    made += plot_examples(exp, args.fig_dir, args.n_examples, conds)
    for fn in made:
        print(f'[plot] {fn}')
    if not made:
        print('[plot] nothing to plot yet')
