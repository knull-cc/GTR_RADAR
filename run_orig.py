"""Run the ORIGINAL run.py unchanged (reproduction check for S0).

Only two things are added around it:
  * radar.compat: restores np.Inf (removed in numpy 2.x, still used by utils/tools.py::EarlyStopping);
  * the test metrics computed by Exp_Main.test are captured and written as JSON to $RADAR_REPRO_OUT.

Usage (same arguments as run.py):
    RADAR_REPRO_OUT=results_radar/raw/<setting>__orig.json python -u run_orig.py --is_training 1 --model GTR ...
"""
import json
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import radar.compat  # noqa: E402,F401
import exp.exp_main as exp_main  # noqa: E402

_captured = []
_metric = exp_main.metric


def _capture(pred, true):
    res = _metric(pred, true)
    _captured.append([float(res[0]), float(res[1])])  # mae, mse (CORR is an array, not a scalar)
    return res


exp_main.metric = _capture  # Exp_Main.test looks the name up at call time

runpy.run_path(os.path.join(HERE, 'run.py'), run_name='__main__')

out = os.environ.get('RADAR_REPRO_OUT')
if out and _captured:
    mae, mse = _captured[-1][0], _captured[-1][1]
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    with open(out, 'w') as f:
        json.dump(dict(kind='orig', mse=mse, mae=mae, argv=sys.argv[1:]), f, indent=1)
    print(f'[run_orig] captured original test mse {mse:.6f} mae {mae:.6f} -> {out}')
