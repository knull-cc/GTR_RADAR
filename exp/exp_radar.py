"""Exp_Radar: train / evaluate RADAR on GTR-family backbones.

Stages (see run_radar.py):
  S0 train_base      GTR exactly as the original loop (no RADAR), no per-epoch test pass   -> base.pth
  S1 train_aug       fine-tune GTR with the RADAR v4 augmentation mixture (fair control)    -> aug_v4.pth
  S2 pretrain_radar  frozen backbone, RADAR only, L_det + L_rec                             -> pre_v4.pth
  S3 train_radar     joint (or --freeze_backbone plug-in) training, full loss + L_route     -> radar_v4.pth / plugin_v4.pth
The test split is used only by `evaluate` (and the S0 reproduction check), never for selection.
Evaluation runs the v4 grid "type × position × length × channel ratio" (radar/perturb.py).
"""
import json
import os
import time
from collections import defaultdict

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
                          rec_loss, route_loss)
from radar.modules import LENGTH_BINS, hampel_filter, length_bin, robust_stats
from radar.perturb import TEST_CONDITIONS, MixturePerturber, apply_condition, condition_rng, raw_zero_from_scaler
from radar.radar import RADARWrapper, resolve_experts, resolve_tau_recal
from utils.tools import adjust_learning_rate

METHODS = ('B0', 'B1', 'B2', 'Ours-plugin')  # main protocol; 'Ours' (joint) and 'Ours-argmax' (hard routing) optional


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


def ckpt_suffix(args, stage):
    """Non-default switches are encoded in the RADAR checkpoint names, so ablations never reuse each other's files.
    S2 (pre) depends on the detector features (and for v5 / v6 on the experts, since E1 refines stage 1); S3 (radar /
    plugin) on every switch."""
    parts = []
    from radar.radar import resolve_experts
    arch = getattr(args, 'arch', 'v6')
    if getattr(args, 'det_feats', 'v4') != 'v4':
        parts.append(f'det{args.det_feats}')
    ex = ','.join(resolve_experts(args))
    ex_default = 'e1,e2,e3,e4' if arch == 'v6' else 'e1,e2,e3'
    if ex != ex_default and (stage in ('radar', 'plugin') or arch in ('v5', 'v6')):  # v5/v6: E1 sees stage 1
        parts.append('exp' + ex.replace(',', ''))
    if stage in ('radar', 'plugin'):
        if getattr(args, 'router', 'linear') != 'linear':
            parts.append(f'router{args.router}')
        if getattr(args, 'dep_mode', 'ema') != 'ema':
            parts.append(f'dep{args.dep_mode}')
        if float(getattr(args, 'lam_route', 0.5)) != 0.5:
            parts.append(f'lroute{float(args.lam_route):g}')
        for key, tag, default in (('gate_consistency', 'gc', arch in ('v5', 'v6')), ('seg_gate', 'sg', arch == 'v6'),
                                  ('fc_mix', 'fm', arch == 'v6')):
            v = int(getattr(args, key, -1))
            if v >= 0 and bool(v) != default:
                parts.append(f'{tag}{v}')
        if int(getattr(args, 'tail_hyp', 0)) and arch in ('v5', 'v6'):
            parts.append('th' + ('' if int(getattr(args, 'tail_hyp_len', 16)) == 16 else str(args.tail_hyp_len)))
        if stage == 'plugin' and getattr(args, 'plugin_train', 'all') != 'all':
            parts.append(f'p{args.plugin_train}')
    if getattr(args, 'ckpt_tag', ''):
        parts.append(args.ckpt_tag)
    return ('_' + '_'.join(parts)) if parts else ''


def ckpt_paths(args):
    d = os.path.join(args.radar_ckpt_dir, setting_name(args))
    os.makedirs(d, exist_ok=True)
    # RADAR checkpoints carry the architecture (v4: three competing experts; v5: E2/E3 selection + residual E1 +
    # consistency gate; v6: v5 + E4 + segment gate + forecast mixing), so architectures never load each other's
    # files. base (pure GTR) is shared; the GTR+Aug control (aug_v4) depends only on the v4 perturbation mixture and
    # is shared by v4, v5 and v6.
    arch = getattr(args, 'arch', 'v6')
    names = dict(base='base', aug='aug_v4', pre=f'pre_{arch}' + ckpt_suffix(args, 'pre'),
                 radar=f'radar_{arch}' + ckpt_suffix(args, 'radar'),
                 plugin=f'plugin_{arch}' + ckpt_suffix(args, 'plugin'))
    return {k: os.path.join(d, f'{v}.pth') for k, v in names.items()}


def load_pre_state(W, path, device):
    """Load an S2 checkpoint into a wrapper, skipping keys whose name or shape does not match (e.g. the Router of a
    different --router type: the Router is not trained in S2). Returns the skipped keys."""
    sd = torch.load(path, map_location=device)
    own = W.state_dict()
    keep = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
    skipped = sorted(set(sd) - set(keep))
    missing = sorted(k for k in own if k not in keep and not k.startswith('router.'))
    if missing:
        raise RuntimeError(f'{path}: missing / mismatched non-Router keys {missing[:8]}')
    W.load_state_dict(keep, strict=False)
    if skipped:
        print(f'[load] {os.path.basename(path)}: skipped {len(skipped)} key(s) ({skipped[:4]}...)')
    return skipped


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

    def mixture(self, rz, seed):
        """The v4 training / validation perturbation mixture (radar/perturb.py)."""
        a = self.args
        return MixturePerturber(rz, a.p_perturb, seed=seed, p_tail=getattr(a, 'p_tail', 0.5),
                                guard=int(getattr(a, 'radar_patch', 8)))

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

    def _route_temp(self, epoch):
        """Router softmax temperature, linear from route_temp_start (epoch 0) to route_temp_end (last S3 epoch)."""
        a = self.args
        t0, t1 = float(getattr(a, 'route_temp_start', 1.0)), float(getattr(a, 'route_temp_end', 0.3))
        n = max(1, int(a.ft_epochs) - 1)
        return t0 + (t1 - t0) * min(1.0, epoch / n)

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
        perturber = self.mixture(rz, args.random_seed + 101)  # same stream as S3
        val_pert = self.mixture(rz, args.val_seed)
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
                W.repair(x_p, c, compute_dep=False, need_route=False)
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
        perturber = self.mixture(rz, args.random_seed + 202)
        val_pert = self.mixture(rz, args.val_seed)
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
                W.repair(x_p, c, rec_mask=rec_mask, compute_dep=False, need_route=False)
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
            load_pre_state(W, pre_ckpt, self.device)
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
        perturber = self.mixture(rz, args.random_seed + 101)  # same stream as S1
        val_pert = self.mixture(rz, args.val_seed)
        hind_gen = torch.Generator().manual_seed(args.random_seed + 303)
        stopper = EarlyStopper(args.ft_patience, out_ckpt)
        tag = 'S3-plugin' if freeze_backbone else 'S3'

        route_params = [p for p in W.router.parameters() if p.requires_grad]
        lam_route = float(getattr(args, 'lam_route', 0.5))

        W.set_route_temp(self._route_temp(0))
        if freeze_backbone and args.tau_calib_frac > 0:  # plug-in (gate-only or all RADAR modules)
            self._calibrate_gate(W, train_loader)
        torch.manual_seed(args.random_seed + 7)  # same CPU RNG stream as S1 -> identical batches/perturbations
        s, cl, pe = self.val_score(W, val_pert)
        print(f'[{tag}] start: val_score {s:.6f} (clean {cl:.6f}, perturbed {pe:.6f})')
        for epoch in range(args.ft_epochs):
            W.set_route_temp(self._route_temp(epoch))
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
                # L_route: forecast loss with the corrupted points (mask) replaced by the routed candidate;
                # its gradient is taken w.r.t. the Router parameters only and added to theirs.
                l_route = None
                if lam_route > 0 and route_params:
                    l_route = route_loss(W, aux, x_p, mask, y, (c,), args.pred_len, self.f_dim)
                if l_route is not None:
                    g_route = torch.autograd.grad(lam_route * l_route, route_params, allow_unused=True)
                total.backward()
                if l_route is not None:
                    for p, g in zip(route_params, g_route):
                        if g is not None:
                            p.grad = g.detach() if p.grad is None else p.grad + g.detach()
                opt.step()
                parts['route'] = float(l_route.detach()) if l_route is not None else 0.0
                logs.append([parts[k] for k in ('mse', 'det', 'rec', 'hind', 'alpha', 'route')])
            m = np.mean(logs, axis=0)
            s, cl, pe = self.val_score(W, val_pert)
            print(f'[{tag}] epoch {epoch + 1} ({time.time() - t0:.1f}s) mse {m[0]:.5f} det {m[1]:.4f} '
                  f'rec {m[2]:.4f} hind {m[3]:.4f} route {m[5]:.5f} mean_alpha {m[4]:.4f} tau {W.tau.item():.3f} '
                  f'T_route {W.route_temp.item():.2f} | val_score {s:.6f} (clean {cl:.6f}, perturbed {pe:.6f})')
            stopper(s, W)
            if stopper.early_stop:
                print('Early stopping')
                break
            self._type3(opt, epoch + 1)
        W.load_state_dict(torch.load(out_ckpt, map_location=self.device))
        for p in W.parameters():
            p.requires_grad_(True)

    def _calibrate_gate(self, W, train_loader, update_dep=True, tag='S3-plugin'):
        """Stage-B initialisation on CLEAN training windows (no val/test data):
        1) fill the backbone's EMA dependency profile; 2) set τ so that only `tau_calib_frac` of clean points
        have α > 0.5 (i.e. τ = (1 − frac)-quantile of the gate logit without τ).
        v6 (segment gate / forecast mixing): the budget is per (window, channel) instead — τ = (1 − tau_calib_wfrac)-
        quantile of the per-window maximum of the gate logit, so at most that fraction of clean windows is touched;
        the point-level quantile becomes the run-extension threshold τ − δ of the segment gate."""
        args = self.args
        nb = max(1, int(args.tau_calib_batches))
        W.eval()
        batches = []
        if isinstance(train_loader, (list, tuple)):
            pick = None
        else:  # evenly spaced batches of an ordered loader; the first nb batches of a shuffled one
            n = len(train_loader)
            pick = set(np.linspace(0, n - 1, min(nb, n)).round().astype(int).tolist()) if not update_dep else None
        for i, (bx, by, _, _, bc) in enumerate(train_loader):
            if pick is None and len(batches) >= nb:
                break
            if pick is not None and i > max(pick):
                break
            if pick is not None and i not in pick:
                continue
            x, _, c = self._to_dev(bx, by, bc)
            batches.append((x, c))
            if update_dep:
                W.calibrate_dependency(x, c)
        zs, zw = [], []
        for x, c in batches:
            z = W.gate_logit_without_tau(x, c)  # point-level gate logit without τ, (B, L, C)
            if z is not None:
                zs.append(z.flatten().float())
                zw.append(z.amax(dim=1).flatten().float())  # per window and channel
        if not zs:
            return
        z = torch.cat(zs)
        if z.numel() > 4_000_000:
            z = z[torch.randperm(z.numel(), device=z.device)[:4_000_000]]
        tau_pt = torch.quantile(z, 1.0 - float(args.tau_calib_frac)).item()
        msg = f'D̄={W.dep_profile.scale().item():.4g}, point threshold {tau_pt:.3f} ({100 * args.tau_calib_frac:.1f}% of clean points)'
        if getattr(W, 'seg_gate', False) or getattr(W, 'fc_mix', False):
            # window-level false-repair budget: at most tau_calib_wfrac of clean (window, channel) pairs get α > 0.5
            wfrac = float(getattr(args, 'tau_calib_wfrac', 0.05))
            zw = torch.cat(zw)
            if zw.numel() > 4_000_000:  # torch.quantile has an input-size limit
                zw = zw[torch.randperm(zw.numel(), device=zw.device)[:4_000_000]]
            tau = torch.quantile(zw, 1.0 - wfrac).item()
            msg += f', τ={tau:.3f} ({100 * wfrac:.1f}% of clean window×channel pairs)'
            if W.seg_gate:
                W.seg_delta.fill_(max(tau - tau_pt, 0.0))
                msg += f', run extension δ={W.seg_delta.item():.3f}'
        else:
            tau = tau_pt
        W.tau_raw.data.fill_(tau)
        print(f'[{tag}] calibrated on {len(batches)} clean train batches: {msg}')

    # ------------------------------------------------------------------ evaluation
    def condition_loader(self, split='test'):
        """Unshuffled loader of a split (test: the repo's test loader; val/train: ordered, no drop_last)."""
        return self.get_data('test') if split == 'test' else self.ordered_loader(split)

    def perturbed_batches(self, cond, split='test'):
        """Yields (x_clean, x_pert, changed, y, cycle) for one grid condition. The perturbation of batch i is drawn
        from rng([grid_seed, crc32(cond), i]), so every method (and every diagnostic) sees identical windows."""
        args = self.args
        _, loader = self.condition_loader(split)
        rz = self.raw_zero('raw')
        guard = int(getattr(args, 'radar_patch', 8))
        for bi, (bx, by, _, _, bc) in enumerate(loader):
            x, y, c = self._to_dev(bx, by, bc)
            x_p, changed = apply_condition(x, cond, rz, args.zero_mode,
                                           condition_rng(args.grid_seed, cond, bi), guard=guard)
            yield x, x_p, changed, y, c

    def test(self, setting=None, test=0, model=None, preprocess=None, conditions=TEST_CONDITIONS,
             collect_extras=None, timing=True, split='test'):
        """Evaluate one model on a split under each condition. Metrics in scaled space (as the repo).
        Returns {'conditions': {cond: {mse, mae, model_time_s, n_batches}}, 'extras': {cond: {...}, ...}}."""
        args = self.args
        model = self.model if model is None else model
        is_radar = isinstance(model, RADARWrapper)
        collect_extras = is_radar if collect_extras is None else collect_extras
        _, loader = self.condition_loader(split)
        model.eval()
        out_res, ex, stats = {}, {}, {}

        # warm-up (not timed)
        with torch.no_grad():
            bx, by, _, _, bc = next(iter(loader))
            x, _, c = self._to_dev(bx, by, bc)
            self._forward(model, preprocess(x) if preprocess else x, c)

        for cond in conditions:
            self._set_probe_seed(model, args.random_seed)
            se = ae = 0.0
            n, nb, t_model = 0, 0, 0.0
            st = RadarStats(model.n_experts, len(model.cand_names)) if collect_extras else None
            with torch.no_grad():
                for x, x_p, changed, y, c in self.perturbed_batches(cond, split):
                    if getattr(model, 'oracle_gate', False):
                        model.oracle_mask = changed
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
                    if st is not None:
                        st.update(model.last_aux, x, x_p, changed)
            out_res[cond] = dict(mse=se / n, mae=ae / n, model_time_s=t_model, n_batches=nb)
            line = f'  {cond:20s} mse {se / n:.6f}  mae {ae / n:.6f}  (model time {t_model:.2f}s)'
            if st is not None:
                ex[cond] = st.result()
                stats[cond] = st
                line += f"  | ᾱ>0.5 {100 * ex[cond]['frac_alpha_gt_0_5']:.2f}%"
            print(line)
        if collect_extras and stats:
            neg = stats['clean'].hist_all if 'clean' in stats else None
            for cond, st in stats.items():
                if cond != 'clean':
                    ex[cond]['auroc'] = hist_auroc(st.hist_pos, neg) if neg is not None else float('nan')
            ex['gate'] = model.gate_values()
            ex['tau'] = float(model.tau.item())
            ex['router'] = model.router_type
            ex['router_coef'] = model.router_coefficients()
            ex['route_features'] = list(model.route_feature_names)
            ex['experts'] = list(model.expert_names)
            ex['candidates'] = list(model.cand_names)
            ex['arch'] = model.arch
            ex['gate_consistency'] = bool(model.gate_consistency)
            ex['seg_gate'] = bool(model.seg_gate)
            ex['fc_mix'] = bool(model.fc_mix)
            ex['oracle_gate'] = bool(model.oracle_gate)
            ex['tail_hyp'] = bool(model.tail_hyp)
            if hasattr(model, 'tau_trained'):
                ex['tau_trained'] = model.tau_trained
            ex['e3_period'] = int(model.e3_period)
            ex['e3_enabled'] = bool(model.periodic.enabled and bool(model.expert_enabled[2]))
            ex['det_feats'] = model.det_feats
            ex['dep_mode'] = model.dep_mode
            ex['route_hard'] = bool(model.route_hard)
        if isinstance(model, RADARWrapper):
            model.probe_generator = None
            model.oracle_mask = None
        return {'conditions': out_res, 'extras': ex}

    def hampel(self):
        a = self.args
        return lambda x: hampel_filter(x, a.hampel_window, a.hampel_k, a.hampel_mad_scale)

    METHOD_CKPT = {'B0': 'base', 'B1': 'aug', 'B2': 'base', 'Ours': 'radar', 'Ours-plugin': 'plugin',
                   'Ours-argmax': 'plugin', 'Ours-oracle': 'plugin', 'Ours-uncal': 'plugin'}

    def load_method(self, method, paths):
        """Returns (model, preprocess) for a method, or (None, None) if its checkpoint is missing.
        'Ours-argmax' = the plug-in checkpoint evaluated with hard (argmax) routing; 'Ours-oracle' = the plug-in
        with a perfect gate (α = 1 exactly on the corrupted points; diagnostic upper bound of the gate);
        'Ours-uncal' = the plug-in with τ as trained (no post-training recalibration; v6 recalibrates by default)."""
        need = self.METHOD_CKPT[method]
        if not os.path.exists(paths[need]):
            return None, None
        if method in ('B0', 'B1', 'B2'):
            m = self.build_backbone().to(self.device)
        else:
            m = self.build_wrapper().to(self.device)
            m.force_gate_zero = bool(self.args.force_gate_zero)
        m.load_state_dict(torch.load(paths[need], map_location=self.device))
        if method == 'Ours-argmax':
            m.route_hard = True
        if method == 'Ours-oracle':
            m.oracle_gate = True
        if (method in ('Ours', 'Ours-plugin', 'Ours-argmax') and resolve_tau_recal(self.args)
                and float(self.args.tau_calib_frac) > 0):
            # post-training recalibration of τ (and δ) on clean training windows: S3 keeps training the detector and
            # the gate after the initial calibration, so the false-repair budget drifts; the dependency profile is
            # left as trained
            tau0 = float(m.tau_raw)
            self._calibrate_gate(m, self.ordered_loader('train')[1], update_dep=False, tag=f'eval {method}')
            m.tau_trained = tau0
        m.eval()
        return m, (self.hampel() if method == 'B2' else None)

    def evaluate_all(self, methods=METHODS, out_dir='results_radar_v4/raw', conditions=TEST_CONDITIONS):
        args = self.args
        paths = ckpt_paths(args)
        setting = setting_name(args)
        os.makedirs(out_dir, exist_ok=True)
        results = {}
        for method in methods:
            model, pre = self.load_method(method, paths)
            if model is None:
                print(f'[eval] {method}: checkpoint {paths[self.METHOD_CKPT[method]]} missing, skipped')
                continue
            # RADAR methods of a non-default architecture are recorded as e.g. 'Ours-plugin-v4', so they can sit next
            # to the default architecture in the same results directory
            name = method
            if method not in ('B0', 'B1', 'B2'):
                if args.arch != 'v6':
                    name = f'{name}-{args.arch}'
                sfx = ckpt_suffix(args, 'radar' if method == 'Ours' else 'plugin')
                if sfx:  # ablation / variant checkpoints get their own row, e.g. 'Ours-plugin_th'
                    name += sfx
            print(f'[eval] {name} ({setting})')
            res = self.test(model=model, preprocess=pre, conditions=tuple(conditions))
            res.update(dict(kind='eval', version='v4', arch=args.arch, method=name, setting=setting,
                            dataset=args.model_id.split('_')[0], data=args.data, backbone=args.backbone,
                            pred_len=args.pred_len, seed=args.random_seed, zero_mode=args.zero_mode,
                            grid_seed=args.grid_seed, test_grid=getattr(args, 'test_grid', 'full'),
                            dep_mode=args.dep_mode, det_feats=args.det_feats, router=args.router,
                            experts=','.join(resolve_experts(args)), ckpt=os.path.basename(paths[self.METHOD_CKPT[method]]),
                            n_params=sum(p.numel() for p in model.parameters())))
            results[name] = res
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


# ----------------------------------------------------------------------------------------------------------------
# Evaluation statistics of RADAR on one condition
# ----------------------------------------------------------------------------------------------------------------
HIST_BINS, HIST_LO, HIST_HI = 2400, -12.0, 12.0  # histogram of the unreliability logit ℓ (bin width 0.01)


def _hist(v):
    v = v.detach().float().clamp(HIST_LO, HIST_HI)
    return torch.histc(v, bins=HIST_BINS, min=HIST_LO, max=HIST_HI).double().cpu().numpy()


def hist_auroc(pos, neg):
    """AUROC = P(score_pos > score_neg) + ½ P(tie) from two histograms on the same bins (exact up to binning)."""
    P, N = float(pos.sum()), float(neg.sum())
    if P == 0 or N == 0:
        return float('nan')
    below = np.concatenate([[0.0], np.cumsum(neg)[:-1]])
    return float((pos * (below + 0.5 * neg)).sum() / (P * N))


class RadarStats:
    """Accumulates, over the batches of one condition: gate statistics (α), routing (π overall / on changed points /
    where the gate opens, also per gap-length bin), repair errors on the changed points in robust units of the clean
    window (|z − x_clean| / s_clean for z = corrupted input, routed target, repaired input, each expert) and the
    ℓ histograms for the detection AUROC (positives = changed points; negatives = all points of the clean condition).
    v5 extras: |Δ| of the residual E1 (robust units) and the stage-1 log-odds of E3 vs E2 split into the
    leave-one-out prior and the learned correction, on changed points and where the gate opens.
    K: number of routed experts (π); Kc: number of candidates in aux['c'] (default K).
    """

    def __init__(self, K, Kc=None):
        self.K = K
        self.Kc = K if Kc is None else Kc
        self.s = defaultdict(float)
        self.pi_all = np.zeros(K)
        self.pi_changed = np.zeros(K)
        self.pi_open = np.zeros(K)
        self.pi_open_gap = np.zeros((len(LENGTH_BINS), K))
        self.n_open_gap = np.zeros(len(LENGTH_BINS))
        self.err_exp = np.zeros(self.Kc)
        self.n_exp = np.zeros(self.Kc)
        self.dbar = None
        self.vec = {}
        self.hist_all = np.zeros(HIST_BINS)
        self.hist_pos = np.zeros(HIST_BINS)

    @staticmethod
    def _np(t):
        return t.detach().double().cpu().numpy()

    @torch.no_grad()
    def update(self, aux, x_clean, x_p, changed):
        s = self.s
        a, pi = aux['alpha'], aux['pi']
        s['n_pts'] += a.numel()
        s['a_sum'] += a.sum().item()
        s['a_gt'] += (a > 0.5).sum().item()
        s['a_last'] += a[:, -1].sum().item()
        s['n_last'] += a[:, -1].numel()
        s['win_open'] += (a > 0.5).any(dim=1).sum().item()  # (window, channel) pairs touched by the repair
        s['n_win'] += a.shape[0] * a.shape[2]
        if 'rho' in aux:
            s['rho_sum'] += aux['rho'].sum().item()
        self.pi_all += self._np(pi.sum(dim=(0, 1, 2)))
        dp = self._np(aux['dbar_patch'].sum(dim=(0, 1)))
        self.dbar = dp if self.dbar is None else self.dbar + dp
        s['n_bc'] += aux['dbar_patch'].shape[0] * aux['dbar_patch'].shape[1]
        self.hist_all += _hist(aux['logit'])
        n_ch = int(changed.sum().item())
        if n_ch == 0:
            return
        s['n_changed'] += n_ch
        self.hist_pos += _hist(aux['logit'][changed])
        _, sc = robust_stats(x_clean)
        sc = sc.expand_as(x_clean)

        def err(z):
            return ((z - x_clean).abs() / sc)[changed].sum().item()

        s['err_raw'] += err(x_p)
        s['err_target'] += err(aux['target'])
        s['err_tilde'] += err(aux['x_tilde'])
        c, avail = aux['c'], aux.get('cand_avail', aux['avail'])
        for k in range(self.Kc):
            mk = changed & avail[..., k]
            self.n_exp[k] += mk.sum().item()
            self.err_exp[k] += ((c[..., k] - x_clean).abs() / sc)[mk].sum().item()
        a_ch = a[changed]
        s['a_changed'] += a_ch.sum().item()
        s['gate_open'] += (a_ch > 0.5).sum().item()
        open_ = changed & (a > 0.5)
        pio = pi[open_]  # (n_open, K)
        self.pi_open += self._np(pio.sum(dim=0))
        self.pi_changed += self._np(pi[changed].sum(dim=0))
        if 'delta_u' in aux:
            d = aux['delta_u'].abs()
            s['d_changed'] += d[changed].sum().item()
            s['d_open'] += d[open_].sum().item()
            for key, src, msk in (('lo_prior_open', 'route_lo_prior', open_), ('lo_corr_open', 'route_lo_corr', open_),
                                  ('lo_prior_changed', 'route_lo_prior', changed),
                                  ('lo_corr_changed', 'route_lo_corr', changed)):
                v = self._np(aux[src][msk].sum(dim=0))  # scalar (v5) or per expert vs E2 (v6)
                self.vec[key] = v if key not in self.vec else self.vec[key] + v
        gb = length_bin(aux['gap'])[open_]
        for b in range(len(LENGTH_BINS)):
            sel = gb == b
            self.n_open_gap[b] += sel.sum().item()
            self.pi_open_gap[b] += self._np(pio[sel].sum(dim=0))

    def result(self):
        s = self.s
        nan = float('nan')

        def div(a, b):
            return a / b if b else nan

        n_ch, n_open = s['n_changed'], s['gate_open']
        out = dict(
            mean_alpha=div(s['a_sum'], s['n_pts']),
            frac_alpha_gt_0_5=div(s['a_gt'], s['n_pts']),
            mean_alpha_last_point=div(s['a_last'], s['n_last']),
            frac_windows_repaired=div(s['win_open'], s['n_win']),
            mean_rho=div(s['rho_sum'], s['n_win']) if 'rho_sum' in s else nan,
            mean_pi=(self.pi_all / max(s['n_pts'], 1)).tolist(),
            mean_dbar_per_patch=(self.dbar / max(s['n_bc'], 1)).tolist() if self.dbar is not None else None,
            n_changed=int(n_ch))
        if n_ch:
            out.update(
                alpha_changed=div(s['a_changed'], n_ch),
                gate_recall=div(n_open, n_ch),
                err_raw=div(s['err_raw'], n_ch), err_target=div(s['err_target'], n_ch),
                err_tilde=div(s['err_tilde'], n_ch),
                err_expert=[div(self.err_exp[k], self.n_exp[k]) for k in range(self.Kc)],
                avail_expert=[div(self.n_exp[k], n_ch) for k in range(self.Kc)],
                pi_changed=(self.pi_changed / n_ch).tolist(),
                n_open=int(n_open),
                pi_open=(self.pi_open / n_open).tolist() if n_open else [nan] * self.K,
                n_open_by_gap={LENGTH_BINS[b]: int(self.n_open_gap[b]) for b in range(len(LENGTH_BINS))},
                pi_open_by_gap={LENGTH_BINS[b]: ((self.pi_open_gap[b] / self.n_open_gap[b]).tolist()
                                                 if self.n_open_gap[b] else [nan] * self.K)
                                for b in range(len(LENGTH_BINS))})
            if 'd_changed' in s:
                def vdiv(key, n):
                    v = self.vec.get(key)
                    if v is None or not n:
                        return nan
                    v = np.asarray(v, dtype=np.float64) / n
                    return float(v) if v.ndim == 0 else v.tolist()
                out.update(delta_changed=div(s['d_changed'], n_ch), delta_open=div(s['d_open'], n_open),
                           logodds_prior_changed=vdiv('lo_prior_changed', n_ch),
                           logodds_corr_changed=vdiv('lo_corr_changed', n_ch),
                           logodds_prior_open=vdiv('lo_prior_open', n_open),
                           logodds_corr_open=vdiv('lo_corr_open', n_open))
        return out
