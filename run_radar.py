"""RADAR experiments on GTR-family backbones.

    python run_radar.py <command> [all run.py arguments] [RADAR arguments]

commands:
  train_base      S0  plain backbone, original training recipe (no per-epoch test pass)      -> base.pth
  train_aug       S1  fine-tune backbone with the RADAR augmentation mixture (fair control)  -> aug.pth
  pretrain_radar  S2  frozen backbone, RADAR only (L_det + L_rec)                            -> pre.pth
  train_radar     S3  joint training, full loss (add --freeze_backbone for the plug-in)      -> radar.pth / plugin.pth
  eval                B0, B1 (GTR+Aug), B2 (Hampel), Ours-plugin [+ Ours ablation] x {clean, T1_raw, T2_k4}
  summarize           results_radar/raw/*.json -> results_radar/summary.csv (+ markdown tables)
  plot                figures_radar/ (dependency profiles, example windows)
  repro_check         compare S0 test MSE with the original run.py (see run_orig.py)
"""
import argparse
import json
import os
import random

import numpy as np
import torch

import radar.compat  # noqa: F401  (must precede utils.tools usage on numpy>=2)

COMMANDS = ['train_base', 'train_aug', 'pretrain_radar', 'train_radar', 'eval', 'summarize', 'plot', 'repro_check']


def build_parser():
    parser = argparse.ArgumentParser(description='RADAR on GTR')
    parser.add_argument('command', choices=COMMANDS)

    # ===================== copied from run.py (required=True relaxed so summarize/plot need no data args)
    parser.add_argument('--random_seed', type=int, default=2026, help='random seed')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--model', type=str, default='GTR', help='(set automatically from --backbone)')
    parser.add_argument('--data', type=str, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=0, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--cycle', type=int, default=24, help='cycle length')
    parser.add_argument('--model_type', type=str, default='mlp', help='model type, options: [linear, mlp]')
    parser.add_argument('--use_revin', type=int, default=1, help='1: use revin or 0: no revin')
    parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length (PatchTST)')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
    parser.add_argument('--revin', type=int, default=0, help='RevIN; True 1 False 0')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
    parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
    parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
    parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')
    parser.add_argument('--rnn_type', default='gru', help='rnn_type')
    parser.add_argument('--dec_way', default='pmf', help='decode way')
    parser.add_argument('--seg_len', type=int, default=48, help='segment length')
    parser.add_argument('--channel_id', type=int, default=1, help='Whether to enable channel position encoding')
    parser.add_argument('--embed_type', type=int, default=0, help='embedding type')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false', default=True, help='whether to use distilling in encoder')
    parser.add_argument('--dropout', type=float, default=0, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=30, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', default=False, help='(must stay off: RADAR runs in fp32)')
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', default=False, help='use multiple gpus')
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

    # ===================== RADAR
    g = parser.add_argument_group('RADAR')
    g.add_argument('--backbone', type=str, default='GTR', choices=['GTR', 'GTRiTransformer', 'GTRPatchTST', 'GTRDLinear'])
    g.add_argument('--radar_patch', type=int, default=8, help='RADAR patch length P')
    g.add_argument('--radar_d', type=int, default=64)
    g.add_argument('--radar_heads', type=int, default=4)
    g.add_argument('--radar_layers', type=int, default=2)
    g.add_argument('--radar_dff', type=int, default=128)
    g.add_argument('--radar_dropout', type=float, default=0.1)
    g.add_argument('--radar_chunk', type=int, default=8192, help='max B*C series per RADAR network call')
    g.add_argument('--radar_grad_ckpt', type=int, default=0, help='gradient checkpointing per chunk (memory)')
    g.add_argument('--u_clip', type=float, default=10.0, help='clip robust-scaled inputs/targets of the RADAR nets')
    g.add_argument('--tau_init', type=float, default=1.0, help='initial gate offset τ (log-linear gate)')
    g.add_argument('--gate_T', type=float, default=0.05, help='(unused since RADAR v2)')
    g.add_argument('--gate_eps', type=float, default=1e-2, help='ε in log(D̃ + ε)')
    g.add_argument('--scale_std_frac', type=float, default=0.1, help='robust scale floor: frac * window std')
    g.add_argument('--scale_min', type=float, default=1e-2, help='robust scale absolute floor')
    g.add_argument('--e2_k', type=int, default=3, help='E2: reliable neighbours per side')
    g.add_argument('--e2_thr', type=float, default=0.5, help='E2: reliability threshold')
    g.add_argument('--e2_trend', type=int, default=0, help='E2: linear trend at the window end (off: no gain on ETTh1)')
    g.add_argument('--dep_mode', type=str, default='ema', choices=['grad', 'ema'],
                   help='ema: backbone EMA profile (no inference overhead); grad: per-sample Hutchinson')
    g.add_argument('--dep_probes', type=int, default=4, help='Hutchinson probes K')
    g.add_argument('--dep_probe_dist', type=str, default='rademacher', choices=['rademacher', 'gaussian'])
    g.add_argument('--dep_ema', type=float, default=0.99, help='EMA momentum of the dependency profile')
    g.add_argument('--use_cycle_prior', type=int, default=0, help='(E3 removed in RADAR v2)')
    g.add_argument('--force_gate_zero', action='store_true', default=False)
    g.add_argument('--freeze_backbone', action='store_true', default=False, help='S3 plug-in mode')
    g.add_argument('--zero_mode', type=str, default='raw', choices=['raw', 'scaled'])
    g.add_argument('--p_perturb', type=float, default=0.5)
    g.add_argument('--lam_det', type=float, default=0.5)
    g.add_argument('--lam_rec', type=float, default=0.5)
    g.add_argument('--lam_hind', type=float, default=0.5)
    g.add_argument('--lam_alpha', type=float, default=0.01)
    g.add_argument('--pos_weight', type=float, default=1.0, help='L_det weight on injected points')
    g.add_argument('--pu_neg_weight', type=float, default=0.1, help='L_det weight on natural (unlabeled) points')
    g.add_argument('--plugin_train', type=str, default='all', choices=['gate', 'all'],
                   help='plug-in S3 (backbone frozen): all = every RADAR module trained with the forecast loss '
                        '(default, best so far); gate = Router + gate only')
    g.add_argument('--rec_mask_ratio', type=float, default=0.1)
    g.add_argument('--hind_level', type=str, default='sample', choices=['sample', 'channel'])
    g.add_argument('--radar_lr', type=float, default=1e-3)
    g.add_argument('--gate_lr', type=float, default=1e-2, help='lr of the gate scalars a, b, τ')
    g.add_argument('--gate_a_init', type=float, default=4.0, help='initial softplus(a)')
    g.add_argument('--gate_b_init', type=float, default=1.0, help='initial softplus(b)')
    g.add_argument('--tau_calib_frac', type=float, default=0.01,
                   help='stage B: init τ so this fraction of clean train points has α>0.5 (0 = off)')
    g.add_argument('--tau_calib_batches', type=int, default=20)
    g.add_argument('--bb_lr_scale', type=float, default=0.1, help='backbone lr = scale * script lr (S1, S3)')
    g.add_argument('--ft_epochs', type=int, default=10, help='S1/S3 max epochs')
    g.add_argument('--ft_patience', type=int, default=3, help='S1/S3 patience')
    g.add_argument('--pre_epochs', type=int, default=20, help='S2 (stage A) epochs; best epoch kept by val')
    g.add_argument('--val_seed', type=int, default=20240917, help='fixed seed of the validation perturbations')
    g.add_argument('--mimic_test_rng', type=int, default=1,
                   help='S0: consume the global-RNG draw of the removed per-epoch test pass (exact run.py stream)')
    g.add_argument('--hampel_window', type=int, default=7)
    g.add_argument('--hampel_k', type=float, default=3.0)
    g.add_argument('--hampel_mad_scale', type=float, default=1.0, help='1.0 = literal 3*MAD; 1.4826 = 3 sigma')
    g.add_argument('--use_tf32', type=int, default=1)
    g.add_argument('--quick', action='store_true', default=False, help='1 seed (2024), pred_len 96')
    g.add_argument('--skip_existing', type=int, default=0, help='skip a training stage if its checkpoint exists')
    g.add_argument('--methods', type=str, default='B0,B1,B2,Ours-plugin',
                   help='add Ours for the joint-training ablation')
    g.add_argument('--conditions', type=str, default='clean,T1_raw,T2_k4',
                   help='test conditions; all: clean,T1_raw,T1_scaled,T2_k4,T2_k8,T3_flat')
    g.add_argument('--radar_ckpt_dir', type=str, default='./checkpoints_radar/')
    g.add_argument('--results_dir', type=str, default='./results_radar/')
    g.add_argument('--fig_dir', type=str, default='./figures_radar/')
    g.add_argument('--plot_datasets', type=str, default='ETTh1,weather')
    g.add_argument('--n_examples', type=int, default=3)
    return parser


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def _conditions(args):
    from radar.perturb import ALL_CONDITIONS
    conds = [c.strip() for c in args.conditions.split(',') if c.strip()]
    bad = [c for c in conds if c not in ALL_CONDITIONS]
    if bad:
        raise SystemExit(f'unknown condition(s) {bad}; choose from {ALL_CONDITIONS}')
    if 'clean' not in conds:
        conds = ['clean'] + conds
    return conds


def raw_dir(args):
    d = os.path.join(args.results_dir, 'raw')
    os.makedirs(d, exist_ok=True)
    return d


def main():
    args = build_parser().parse_args()
    if args.quick:
        args.pred_len, args.random_seed = 96, 2024
    if args.use_amp:
        raise SystemExit('RADAR experiments run in fp32; drop --use_amp')
    if args.use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        args.device_ids = [int(i) for i in args.devices.split(',')]
        args.gpu = args.device_ids[0]

    cmd = args.command
    if cmd == 'summarize':
        from radar.summary import summarize
        summarize(args.results_dir, conditions=_conditions(args))
        return

    from exp.exp_radar import Exp_Radar, ckpt_paths, setting_name
    set_seed(args.random_seed)
    setting = setting_name(args)
    paths = ckpt_paths(args)
    print(f'>>> {cmd}  {setting}')

    if cmd == 'train_base':
        out_json = os.path.join(raw_dir(args), f'{setting}__S0.json')
        if args.skip_existing and os.path.exists(paths['base']) and os.path.exists(out_json):
            print(f'skip: {paths["base"]} exists')
            return
        print('Args in experiment:')
        print(args)
        exp = Exp_Radar(args, use_radar=False)
        exp.train(setting, ckpt_path=paths['base'])
        res = exp.test(conditions=('clean',), timing=False)
        clean = res['conditions']['clean']
        with open(out_json, 'w') as f:
            json.dump(dict(kind='S0', setting=setting, mse=clean['mse'], mae=clean['mae']), f, indent=1)
        print(f'S0 test mse:{clean["mse"]}, mae:{clean["mae"]}')

    elif cmd == 'repro_check':
        d = raw_dir(args)
        s0_fn, orig_fn = os.path.join(d, f'{setting}__S0.json'), os.path.join(d, f'{setting}__orig.json')
        if not (os.path.exists(s0_fn) and os.path.exists(orig_fn)):
            raise SystemExit(f'missing {s0_fn} or {orig_fn}')
        s0, orig = json.load(open(s0_fn)), json.load(open(orig_fn))
        rel = abs(s0['mse'] - orig['mse']) / orig['mse']
        ok = rel < 0.02
        out = dict(kind='repro', setting=setting, s0_mse=s0['mse'], orig_mse=orig['mse'], s0_mae=s0['mae'],
                   orig_mae=orig['mae'], rel_diff_mse=rel, passed=ok)
        json.dump(out, open(os.path.join(d, f'{setting}__repro.json'), 'w'), indent=1)
        print(f'[repro] original run.py mse {orig["mse"]:.6f} | Exp_Radar S0 mse {s0["mse"]:.6f} | '
              f'rel diff {100 * rel:.3f}% -> {"PASS" if ok else "FAIL (investigate before continuing)"}')
        if not ok:
            raise SystemExit(2)

    elif cmd == 'train_aug':
        if args.skip_existing and os.path.exists(paths['aug']):
            print(f'skip: {paths["aug"]} exists')
            return
        Exp_Radar(args, use_radar=False).train_aug(paths['base'], paths['aug'])

    elif cmd == 'pretrain_radar':
        if args.skip_existing and os.path.exists(paths['pre']):
            print(f'skip: {paths["pre"]} exists')
            return
        Exp_Radar(args, use_radar=True).pretrain_radar(paths['base'], paths['pre'])

    elif cmd == 'train_radar':
        out = paths['plugin'] if args.freeze_backbone else paths['radar']
        if args.skip_existing and os.path.exists(out):
            print(f'skip: {out} exists')
            return
        Exp_Radar(args, use_radar=True).train_radar(paths['base'], paths['pre'], out,
                                                     freeze_backbone=args.freeze_backbone)

    elif cmd == 'eval':
        methods = [m.strip() for m in args.methods.split(',') if m.strip()]
        Exp_Radar(args, use_radar=False).evaluate_all(methods, raw_dir(args), conditions=_conditions(args))

    elif cmd == 'plot':
        from radar.plotting import plot_all
        plot_all(args)


if __name__ == '__main__':
    main()
