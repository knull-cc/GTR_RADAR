"""Exp_Radar: train / evaluate RADAR on GTR-family backbones.

Stages (see run_radar.py):
  S0 train_base      GTR exactly as the original loop (no RADAR), no per-epoch test pass   -> base.pth
  S1 train_aug       fine-tune GTR with the RADAR augmentation mixture (fair control)       -> aug.pth
  S2 pretrain_radar  frozen backbone, RADAR only, L_det + L_rec                             -> pre.pth
  S3 train_radar     joint (or --freeze_backbone plug-in) training with the full loss       -> radar.pth / plugin.pth
The test split is used only by `evaluate` (and the S0 reproduction check), never for selection.
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

import radar.compat  # noqa: F401  (np.Inf shim for utils/tools.py on numpy>=2)
from exp.exp_main import Exp_Main
from radar.losses import (det_loss, hindsight_loss, hindsight_targets, patch_mask_to_points, radar_loss,
                          rec_loss)
from radar.modules import hampel_filter
from radar.perturb import TEST_CONDITIONS, MixturePerturber, apply_test_perturbation, raw_zero_from_scaler
from radar.radar import RADARWrapper
from utils.tools import adjust_learning_rate

METHODS = ('B0', 'B1', 'B2', 'Ours-plugin')  # main protocol; 'Ours' (joint training) = ablation


class EarlyStopper:
    """Same rule as utils.tools.EarlyStopping (delta = 0), but saves to an explicit checkpoint file."""

    def __init__(self, patience, path, verbose=True):
        self.patience = patience
        self.path = path
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_loss = float('inf')

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self._save(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save(val_loss, model)
            self.counter = 0

    def _save(self, val_loss, model):
        if self.verbose:
            print(f'Validation score improved ({self.best_loss:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.best_loss = val_loss


def setting_name(args):
    """Same format as run.py, with the backbone name in the model slot."""
    return '{}_{}_{}_ft{}_sl{}_pl{}_cycle{}_seed{}'.format(
        args.model_id, args.backbone, args.data, args.features, args.seq_len, args.pred_len, args.cycle,
        args.random_seed)


def ckpt_paths(args):
    d = os.path.join(args.radar_ckpt_dir, setting_name(args))
    os.makedirs(d, exist_ok=True)
    # RADAR v2 architecture (PU detection, reliable-neighbour expert, log-linear gate, staged training) uses new
    # file names, so v1 checkpoints are never loaded by mistake; base/aug (pure GTR) are shared.
    names = dict(base='base', aug='aug', pre='pre_v3', radar='radar_v3', plugin='plugin_v3')
    return {k: os.path.join(d, f'{v}.pth') for k, v in names.items()}


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


class Exp_Radar(Exp_Main):
    def __init__(self, args, use_radar=False):
        self.use_radar = bool(use_radar)
        if not getattr(args, 'backbone', None):
            args.backbone = 'GTR'
        # "RADAR_" + backbone keeps 'GTR' in the name, so the original model(batch_x, batch_cycle) dispatch applies
        args.model = ('RADAR_' + args.backbone) if self.use_radar else args.backbone
        self._data = {}
        super().__init__(args)
        self.f_dim = -1 if args.features == 'MS' else 0

    # ------------------------------------------------------------------ model / data
    def build_backbone(self):
        """Build the backbone from the ORIGINAL model_dict in Exp_Main._build_model."""
        saved = self.args.model
        self.args.model = self.args.backbone
        try:
            return Exp_Main._build_model(self)
        finally:
            self.args.model = saved

    def build_wrapper(self):
        return RADARWrapper(self.build_backbone(), self.args).float()

    def _build_model(self):
        return self.build_wrapper() if self.use_radar else self.build_backbone()

    @property
    def backbone(self):
        return self.model.backbone if self.use_radar else self.model

    def get_data(self, flag):
        if flag not in self._data:
            self._data[flag] = self._get_data(flag)
        return self._data[flag]

    def ordered_loader(self, flag):
        """Unshuffled, no drop_last loader (deterministic fixed-seed perturbations for S1-S3 validation)."""
        key = flag + '_ordered'
        if key not in self._data:
            ds, _ = self.get_data(flag)
            self._data[key] = (ds, DataLoader(ds, batch_size=self.args.batch_size, shuffle=False,
                                              num_workers=self.args.num_workers, drop_last=False))
        return self._data[key]

    def raw_zero(self, mode=None):
        train_ds, _ = self.get_data('train')
        return raw_zero_from_scaler(train_ds.scaler, mode or self.args.zero_mode)

    def _to_dev(self, bx, by, bc):
        return bx.float().to(self.device), by.float().to(self.device), bc.int().to(self.device)

    def _forward(self, model, x, cycle, **kw):
        if any(s in self.args.model for s in ('CycleNet', 'GTR')):
            return model(x, cycle, **kw)
        raise ValueError(f'Exp_Radar only supports GTR-family backbones, got {self.args.model}')

    def _train_mode(self, freeze_backbone=False, freeze_stage_a=False):
        self.model.train()
        if freeze_backbone and self.use_radar:
            self.model.backbone.eval()
        if freeze_stage_a and self.use_radar:
            self.model.reliability.eval()
            self.model.repairer.eval()

    def _set_probe_seed(self, model, seed):
        if isinstance(model, RADARWrapper):
            g = torch.Generator(device=self.device)
            g.manual_seed(int(seed))
            model.probe_generator = g

    @staticmethod
    def _type3(opt, epoch):
        """lradj=type3 relative to each param group's own initial lr ('lr0')."""
        for g in opt.param_groups:
            g['lr'] = g['lr0'] if epoch < 3 else g['lr0'] * (0.8 ** (epoch - 3))
        print('Updating learning rate to ' + ', '.join(f"{g['lr']:.3g}" for g in opt.param_groups))

    def _rand_patch_mask(self, BC, N):
        return torch.rand(BC, N, device=self.device) < self.args.rec_mask_ratio

    # ------------------------------------------------------------------ validation
    def vali(self, vali_data, vali_loader, criterion=None, model=None, perturber=None, preprocess=None):
        """Mean MSE over all predicted values. With the original (drop_last) val loader every batch has the
        same size, so this equals Exp_Main.vali's batch average."""
        model = self.model if model is None else model
        model.eval()
        if perturber is not None:
            perturber.reset()
        self._set_probe_seed(model, self.args.val_seed)
        se, n = 0.0, 0
        with torch.no_grad():
            for bx, by, _, _, bc in vali_loader:
                x = bx.float()
                if perturber is not None:
                    x, _, _ = perturber(x)
                x, y, c = self._to_dev(x, by, bc)
                if preprocess is not None:
                    x = preprocess(x)
                out = self._forward(model, x, c)[:, -self.args.pred_len:, self.f_dim:]
                y = y[:, -self.args.pred_len:, self.f_dim:]
                se += F.mse_loss(out, y, reduction='sum').item()
                n += out.numel()
        if isinstance(model, RADARWrapper):
            model.probe_generator = None
        model.train()
        return se / max(n, 1)

    def val_score(self, model, val_perturber):
        ds, loader = self.ordered_loader('val')
        clean = self.vali(ds, loader, model=model)
        pert = self.vali(ds, loader, model=model, perturber=val_perturber)
        return 0.5 * clean + 0.5 * pert, clean, pert

    # ------------------------------------------------------------------ S0
    def train(self, setting=None, ckpt_path=None):
        """S0: identical to Exp_Main.train except that the per-epoch test-loss pass is removed."""
        args = self.args
        assert not self.use_radar, 'S0 trains the plain backbone'
        ckpt_path = ckpt_path or ckpt_paths(args)['base']
        train_data, train_loader = self.get_data('train')
        vali_data, vali_loader = self.get_data('val')
        self.get_data('test')  # built (not iterated) exactly like the original; consumes no RNG

        model = self.model
        model_optim = optim.Adam(model.parameters(), lr=args.learning_rate)
        # The original loop always constructs OneCycleLR. Even with lradj=type3 its constructor sets the epoch-1 lr
        # to learning_rate/25 and Adam beta1 to 0.95; we keep it so S0 reproduces run.py.
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim, steps_per_epoch=len(train_loader),
                                            pct_start=args.pct_start, epochs=args.train_epochs,
                                            max_lr=args.learning_rate)
        criterion = nn.MSELoss()
        stopper = EarlyStopper(args.patience, ckpt_path)

        for epoch in range(args.train_epochs):
            model.train()
            t0 = time.time()
            train_loss = []
            for i, (bx, by, _, _, bc) in enumerate(train_loader):
                model_optim.zero_grad()
                x, y, c = self._to_dev(bx, by, bc)
                out = self._forward(model, x, c)[:, -args.pred_len:, self.f_dim:]
                loss = criterion(out, y[:, -args.pred_len:, self.f_dim:])
                train_loss.append(loss.item())
                loss.backward()
                model_optim.step()
                if args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, args, printout=False)
                    scheduler.step()
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            if args.mimic_test_rng:
                # the original's per-epoch test pass creates one DataLoader iterator = one draw from the global
                # torch RNG; consuming it keeps the shuffling stream identical to run.py without touching test data
                torch.empty((), dtype=torch.int64).random_()
            print('Epoch: {} cost time: {:.2f}s | Train Loss: {:.7f} Vali Loss: {:.7f}'.format(
                epoch + 1, time.time() - t0, np.average(train_loss), vali_loss))
            stopper(vali_loss, model)
            if stopper.early_stop:
                print('Early stopping')
                break
            if args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, args)
        model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        return model

    # ------------------------------------------------------------------ S1
    def train_aug(self, base_ckpt, out_ckpt):
        args = self.args
        assert not self.use_radar
        self.model.load_state_dict(torch.load(base_ckpt, map_location=self.device))
        _, train_loader = self.get_data('train')
        rz = self.raw_zero()
        perturber = MixturePerturber(rz, args.p_perturb, seed=args.random_seed + 101)  # same stream as S3
        val_pert = MixturePerturber(rz, args.p_perturb, seed=args.val_seed)
        lr0 = args.bb_lr_scale * args.learning_rate
        opt = optim.Adam([{'params': self.model.parameters(), 'lr': lr0, 'lr0': lr0}])
        stopper = EarlyStopper(args.ft_patience, out_ckpt)
        torch.manual_seed(args.random_seed + 7)  # S1 and S3 then see the same batch order and perturbations
        s, cl, pe = self.val_score(self.model, val_pert)
        print(f'[S1] start: val_score {s:.6f} (clean {cl:.6f}, perturbed {pe:.6f})')
        for epoch in range(args.ft_epochs):
            self._train_mode()
            t0, losses = time.time(), []
            for bx, by, _, _, bc in train_loader:
                x, _, _ = perturber(bx.float())
                x, y, c = self._to_dev(x, by, bc)
                opt.zero_grad()
                out = self._forward(self.model, x, c)[:, -args.pred_len:, self.f_dim:]
                loss = F.mse_loss(out, y[:, -args.pred_len:, self.f_dim:])
                loss.backward()
                opt.step()
                losses.append(loss.item())
            s, cl, pe = self.val_score(self.model, val_pert)
            print(f'[S1] epoch {epoch + 1} ({time.time() - t0:.1f}s) train {np.mean(losses):.6f} | '
                  f'val_score {s:.6f} (clean {cl:.6f}, perturbed {pe:.6f})')
            stopper(s, self.model)
            if stopper.early_stop:
                print('Early stopping')
                break
            self._type3(opt, epoch + 1)
        self.model.load_state_dict(torch.load(out_ckpt, map_location=self.device))

    # ------------------------------------------------------------------ S2
    def _det_rec_val(self, val_pert):
        W = self.model
        W.eval()
        val_pert.reset()
        _, loader = self.ordered_loader('val')
        tot, nb = 0.0, 0
        with torch.no_grad():
            for bx, by, _, _, bc in loader:
                x_clean = bx.float()
                x_p, mask, _ = val_pert(x_clean)
                x_p, _, c = self._to_dev(x_p, by, bc)
                W.repair(x_p, c, compute_dep=False)
                aux = W.last_aux
                l_det = det_loss(aux['logit'], mask.to(self.device), self.args.pos_weight, self.args.pu_neg_weight)
                l_rec = rec_loss(aux['u_hat'], x_clean.to(self.device), mask.to(self.device), self.args.u_clip,
                                 aux['m'], aux['s'])
                tot += (l_det + l_rec).item()
                nb += 1
        return tot / max(nb, 1)

    def pretrain_radar(self, base_ckpt, out_ckpt):
        args = self.args
        assert self.use_radar
        W = self.model
        W.backbone.load_state_dict(torch.load(base_ckpt, map_location=self.device))
        for p in W.backbone.parameters():
            p.requires_grad_(False)
        _, train_loader = self.get_data('train')
        rz = self.raw_zero()
        perturber = MixturePerturber(rz, args.p_perturb, seed=args.random_seed + 202)
        val_pert = MixturePerturber(rz, args.p_perturb, seed=args.val_seed)
        opt = optim.Adam(W.stage_a_parameters(), lr=args.radar_lr)  # stage A: ReliabilityNet + Repairer
        best = float('inf')
        for epoch in range(args.pre_epochs):
            self._train_mode(freeze_backbone=True)
            t0, parts = time.time(), []
            for bx, by, _, _, bc in train_loader:
                x_clean = bx.float()
                x_p, mask, _ = perturber(x_clean)
                x_p, _, c = self._to_dev(x_p, by, bc)
                x_clean, mask = x_clean.to(self.device), mask.to(self.device)
                B, L, C = x_p.shape
                rec_mask = self._rand_patch_mask(B * C, W.N)
                opt.zero_grad()
                W.repair(x_p, c, rec_mask=rec_mask, compute_dep=False)
                aux = W.last_aux
                l_det = det_loss(aux['logit'], mask, args.pos_weight, args.pu_neg_weight)
                sel = mask | patch_mask_to_points(rec_mask, B, C, L, W.P)
                l_rec = rec_loss(aux['u_hat'], x_clean, sel, args.u_clip, aux['m'], aux['s'])
                loss = l_det + l_rec
                loss.backward()
                opt.step()
                parts.append((l_det.item(), l_rec.item()))
            v = self._det_rec_val(val_pert)
            pd_, pr_ = np.mean(parts, axis=0)
            print(f'[S2] epoch {epoch + 1} ({time.time() - t0:.1f}s) train det {pd_:.4f} rec {pr_:.4f} | '
                  f'val det+rec {v:.4f}')
            if v < best:
                best = v
                torch.save(W.state_dict(), out_ckpt)
                print(f'[S2] saved {out_ckpt}')
        W.load_state_dict(torch.load(out_ckpt, map_location=self.device))
        for p in W.backbone.parameters():
            p.requires_grad_(True)

    # ------------------------------------------------------------------ S3
    def train_radar(self, base_ckpt, pre_ckpt, out_ckpt, freeze_backbone=False):
        args = self.args
        assert self.use_radar
        W = self.model
        if pre_ckpt and os.path.exists(pre_ckpt):
            W.load_state_dict(torch.load(pre_ckpt, map_location=self.device))
        else:
            print(f'[S3] WARNING: no S2 checkpoint at {pre_ckpt}; RADAR starts from scratch')
        W.backbone.load_state_dict(torch.load(base_ckpt, map_location=self.device))
        for p in W.backbone.parameters():
            p.requires_grad_(not freeze_backbone)
        # plug-in = stage B: backbone AND stage-A modules frozen; only Router + gate (a, b, τ) are trained
        stage_b = freeze_backbone and getattr(args, 'plugin_train', 'gate') == 'gate'
        for p in W.stage_a_parameters():
            p.requires_grad_(not stage_b)

        radar_lr = args.radar_lr
        bb_lr = args.bb_lr_scale * args.learning_rate
        gate_lr = args.gate_lr
        scalar_ids = {id(p) for p in W.gate_scalars()}
        trainable = W.gate_parameters() if stage_b else W.radar_parameters()
        rest = [p for p in trainable if id(p) not in scalar_ids]
        groups = [{'params': rest, 'lr': radar_lr, 'lr0': radar_lr},
                  {'params': W.gate_scalars(), 'lr': gate_lr, 'lr0': gate_lr}]  # a, b, τ: larger lr
        if not freeze_backbone:
            groups.append({'params': list(W.backbone.parameters()), 'lr': bb_lr, 'lr0': bb_lr})
        opt = optim.Adam(groups)

        _, train_loader = self.get_data('train')
        rz = self.raw_zero()
        perturber = MixturePerturber(rz, args.p_perturb, seed=args.random_seed + 101)  # same stream as S1
        val_pert = MixturePerturber(rz, args.p_perturb, seed=args.val_seed)
        hind_gen = torch.Generator().manual_seed(args.random_seed + 303)
        stopper = EarlyStopper(args.ft_patience, out_ckpt)
        tag = 'S3-plugin' if freeze_backbone else 'S3'

        if stage_b and args.tau_calib_frac > 0:
            self._calibrate_gate(W, train_loader)
        torch.manual_seed(args.random_seed + 7)  # same CPU RNG stream as S1 -> identical batches/perturbations
        s, cl, pe = self.val_score(W, val_pert)
        print(f'[{tag}] start: val_score {s:.6f} (clean {cl:.6f}, perturbed {pe:.6f})')
        for epoch in range(args.ft_epochs):
            self._train_mode(freeze_backbone, freeze_stage_a=stage_b)
            t0, logs = time.time(), []
            for bx, by, _, _, bc in train_loader:
                x_clean = bx.float()
                x_p, mask, _ = perturber(x_clean)
                x_p, y, c = self._to_dev(x_p, by, bc)
                x_clean, mask = x_clean.to(self.device), mask.to(self.device)
                B, L, C = x_p.shape
                rec_mask = None if stage_b else self._rand_patch_mask(B * C, W.N)
                opt.zero_grad()
                y_hat = W(x_p, c, rec_mask=rec_mask)[:, -args.pred_len:, self.f_dim:]
                aux = W.last_aux
                y = y[:, -args.pred_len:, self.f_dim:]
                hind = None
                if args.lam_hind > 0:
                    tg = hindsight_targets(W.backbone, x_p, c, y, aux['target'].detach(), W.P, args.pred_len,
                                           self.f_dim, args.hind_level, hind_gen)
                    hind = hindsight_loss(aux['alpha'], tg, W.P, self.f_dim)
                rec_pts = None if rec_mask is None else patch_mask_to_points(rec_mask, B, C, L, W.P)
                total, parts = radar_loss(y_hat, y, aux, x_clean, mask, rec_pts, hind, args,
                                          stage='B' if stage_b else 'joint')
                total.backward()
                opt.step()
                logs.append([parts[k] for k in ('mse', 'det', 'rec', 'hind', 'alpha')])
            m = np.mean(logs, axis=0)
            s, cl, pe = self.val_score(W, val_pert)
            print(f'[{tag}] epoch {epoch + 1} ({time.time() - t0:.1f}s) mse {m[0]:.5f} det {m[1]:.4f} '
                  f'rec {m[2]:.4f} hind {m[3]:.4f} mean_alpha {m[4]:.4f} tau {W.tau.item():.3f} | '
                  f'val_score {s:.6f} (clean {cl:.6f}, perturbed {pe:.6f})')
            stopper(s, W)
            if stopper.early_stop:
                print('Early stopping')
                break
            self._type3(opt, epoch + 1)
        W.load_state_dict(torch.load(out_ckpt, map_location=self.device))
        for p in W.parameters():
            p.requires_grad_(True)

    def _calibrate_gate(self, W, train_loader):
        """Stage-B initialisation on CLEAN training windows (no val/test data):
        1) fill the backbone's EMA dependency profile; 2) set τ so that only `tau_calib_frac` of clean points
        have α > 0.5 (i.e. τ = (1 − frac)-quantile of the gate logit without τ)."""
        args = self.args
        nb = max(1, int(args.tau_calib_batches))
        W.eval()
        batches = []
        for i, (bx, by, _, _, bc) in enumerate(train_loader):
            if i >= nb:
                break
            x, _, c = self._to_dev(bx, by, bc)
            batches.append((x, c))
            W.calibrate_dependency(x, c)
        zs = []
        for x, c in batches:
            z = W.gate_logit_without_tau(x, c)
            if z is not None:
                zs.append(z.flatten().float())
        if not zs:
            return
        z = torch.cat(zs)
        if z.numel() > 4_000_000:
            z = z[torch.randperm(z.numel(), device=z.device)[:4_000_000]]
        tau = torch.quantile(z, 1.0 - float(args.tau_calib_frac)).item()
        W.tau_raw.data.fill_(tau)
        print(f'[S3-plugin] calibrated on {len(batches)} clean train batches: D̄={W.dep_profile.scale().item():.4g}, '
              f'τ={tau:.3f} (target α>0.5 on {100 * args.tau_calib_frac:.1f}% of clean points)')

    # ------------------------------------------------------------------ evaluation
    def test(self, setting=None, test=0, model=None, preprocess=None, conditions=TEST_CONDITIONS,
             collect_extras=None, timing=True):
        """Evaluate one model on the test split under each condition. Metrics in scaled space (as the repo).
        Returns {'conditions': {cond: {mse, mae, model_time_s, n_batches}}, 'extras': {...}}."""
        args = self.args
        model = self.model if model is None else model
        is_radar = isinstance(model, RADARWrapper)
        collect_extras = is_radar if collect_extras is None else collect_extras
        _, test_loader = self.get_data('test')
        rz = self.raw_zero('raw')
        model.eval()
        out_res, ex = {}, {}
        auroc_scores, auroc_labels = [], []

        # warm-up (not timed)
        with torch.no_grad():
            bx, by, _, _, bc = next(iter(test_loader))
            x, _, c = self._to_dev(bx, by, bc)
            self._forward(model, preprocess(x) if preprocess else x, c)

        for cond in conditions:
            self._set_probe_seed(model, args.random_seed)
            se = ae = 0.0
            n, nb, t_model = 0, 0, 0.0
            acc = None
            if collect_extras and cond in ('clean', 'T1_raw'):
                K = int(getattr(model, 'n_experts', 3))
                acc = dict(a_sum=0.0, a_gt=0.0, a_last=0.0, n_pts=0, n_last=0, pi=np.zeros(K), pi_aw=np.zeros(K),
                           a_tot=0.0, dbar=None, n_bc=0)
            with torch.no_grad():
                for bx, by, _, _, bc in test_loader:
                    x, y, c = self._to_dev(bx, by, bc)
                    x_p, changed = apply_test_perturbation(x, cond, rz, args.zero_mode)
                    if timing:
                        _sync(self.device)
                    t0 = time.perf_counter()
                    xin = preprocess(x_p) if preprocess is not None else x_p
                    out = self._forward(model, xin, c)
                    if timing:
                        _sync(self.device)
                    t_model += time.perf_counter() - t0
                    d = (out[:, -args.pred_len:, self.f_dim:] - y[:, -args.pred_len:, self.f_dim:]).double()
                    se += (d ** 2).sum().item()
                    ae += d.abs().sum().item()
                    n += d.numel()
                    nb += 1
                    if acc is not None:
                        aux = model.last_aux
                        a, pi = aux['alpha'], aux['pi']
                        acc['a_sum'] += a.sum().item()
                        acc['a_gt'] += (a > 0.5).sum().item()
                        acc['a_last'] += a[:, -1, :].sum().item()
                        acc['n_pts'] += a.numel()
                        acc['n_last'] += a[:, -1, :].numel()
                        acc['pi'] += pi.sum(dim=(0, 1, 2)).double().cpu().numpy()
                        acc['pi_aw'] += (pi * a.unsqueeze(-1)).sum(dim=(0, 1, 2)).double().cpu().numpy()
                        acc['a_tot'] += a.sum().item()
                        dp = aux['dbar_patch'].sum(dim=(0, 1)).double().cpu().numpy()
                        acc['dbar'] = dp if acc['dbar'] is None else acc['dbar'] + dp
                        acc['n_bc'] += aux['dbar_patch'].shape[0] * aux['dbar_patch'].shape[1]
                        score = (1.0 - aux['R'][:, -1, :]).flatten().float().cpu().numpy()
                        if cond == 'clean':
                            auroc_scores.append(score)
                            auroc_labels.append(np.zeros_like(score, dtype=np.int8))
                        else:
                            keep = changed[:, -1, :].flatten().cpu().numpy()
                            auroc_scores.append(score[keep])
                            auroc_labels.append(np.ones(int(keep.sum()), dtype=np.int8))
            out_res[cond] = dict(mse=se / n, mae=ae / n, model_time_s=t_model, n_batches=nb)
            print(f'  {cond:10s} mse {se / n:.6f}  mae {ae / n:.6f}  (model time {t_model:.2f}s)')
            if acc is not None:
                ex[cond] = dict(
                    mean_alpha=acc['a_sum'] / acc['n_pts'],
                    frac_alpha_gt_0_5=acc['a_gt'] / acc['n_pts'],
                    mean_alpha_last_point=acc['a_last'] / acc['n_last'],
                    mean_pi=(acc['pi'] / acc['n_pts']).tolist(),
                    alpha_weighted_pi=(acc['pi_aw'] / max(acc['a_tot'], 1e-12)).tolist(),
                    mean_dbar_per_patch=(acc['dbar'] / acc['n_bc']).tolist(),
                )
        if collect_extras and auroc_scores:
            from sklearn.metrics import roc_auc_score
            s_all = np.concatenate(auroc_scores)
            l_all = np.concatenate(auroc_labels)
            ex['T1_last_point_auroc'] = (float(roc_auc_score(l_all, s_all))
                                         if 0 < l_all.sum() < len(l_all) else float('nan'))
            ex['T1_auroc_n_pos'] = int(l_all.sum())
            ex['T1_auroc_n_neg'] = int(len(l_all) - l_all.sum())
            ex['tau'] = float(model.tau.item())
            ex['experts'] = list(getattr(model, 'expert_names', ('Repairer', 'LocalMedian', 'CyclePrior')))
            ex['cycle_prior_enabled'] = bool(model.use_cycle_prior)
        if isinstance(model, RADARWrapper):
            model.probe_generator = None
        return {'conditions': out_res, 'extras': ex}

    def hampel(self):
        a = self.args
        return lambda x: hampel_filter(x, a.hampel_window, a.hampel_k, a.hampel_mad_scale)

    def load_method(self, method, paths):
        """Returns (model, preprocess) for a method, or (None, None) if its checkpoint is missing."""
        need = {'B0': 'base', 'B1': 'aug', 'B2': 'base', 'Ours': 'radar', 'Ours-plugin': 'plugin'}[method]
        if not os.path.exists(paths[need]):
            return None, None
        if method in ('B0', 'B1', 'B2'):
            m = self.build_backbone().to(self.device)
        else:
            m = self.build_wrapper().to(self.device)
            m.force_gate_zero = bool(self.args.force_gate_zero)
        m.load_state_dict(torch.load(paths[need], map_location=self.device))
        m.eval()
        return m, (self.hampel() if method == 'B2' else None)

    def evaluate_all(self, methods=METHODS, out_dir='results_radar/raw', conditions=TEST_CONDITIONS):
        args = self.args
        paths = ckpt_paths(args)
        setting = setting_name(args)
        os.makedirs(out_dir, exist_ok=True)
        results = {}
        for method in methods:
            model, pre = self.load_method(method, paths)
            if model is None:
                print(f'[eval] {method}: checkpoint missing, skipped')
                continue
            print(f'[eval] {method} ({setting})')
            res = self.test(model=model, preprocess=pre, conditions=tuple(conditions))
            res.update(dict(kind='eval', method=method, setting=setting, dataset=args.model_id.split('_')[0],
                            data=args.data, backbone=args.backbone, pred_len=args.pred_len, seed=args.random_seed,
                            zero_mode=args.zero_mode, dep_mode=args.dep_mode,
                            n_params=sum(p.numel() for p in model.parameters())))
            results[method] = res
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        if 'B0' in results:
            t0 = results['B0']['conditions']['clean']['model_time_s']
            for r in results.values():
                r['time_ratio_vs_B0'] = r['conditions']['clean']['model_time_s'] / max(t0, 1e-12)
        for method, r in results.items():
            fn = os.path.join(out_dir, f'{setting}__{method}.json')
            with open(fn, 'w') as f:
                json.dump(r, f, indent=1)
        return results
