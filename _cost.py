"""Cost model: how long a full hybrid run over the 16 subsets that need a search."""
import sys, time
sys.path.insert(0, '.')
import numpy as np
import scipy.io
from data_loaders import load_subset
from data_loaders.criteria import SUBSET_REGISTRY

NEEDS_SEARCH = ['A1', 'A2', 'A3', 'C1', 'C2', 'C3', 'G1', 'G2', 'G3',
                'H1', 'H2', 'H3', 'MA1', 'MA2', 'MA3', 'TH1', 'TH2', 'TH3', 'W3']

import importlib.util
spec = importlib.util.spec_from_file_location('wncrod_featurewise',
        'implementations/3_FeatureWise/wncrod_featurewise.py')
mod = importlib.util.module_from_spec(spec)
sys.modules['wncrod_featurewise'] = mod
spec.loader.exec_module(mod)
W = mod.WNCROD_featurewise

total_per_eval = 0.0
rows = []
for name in NEEDS_SEARCH:
    _, y, X_tem = load_subset(name, validate=False)
    d = scipy.io.loadmat('baseline/datasets/' + SUBSET_REGISTRY[name]['mat_file'])['trandata'][:, :-1].astype(float)
    ID = np.all(d <= 1, axis=0)
    act = int((np.std(d[:, ID], axis=0) > 0).sum()) if ID.any() else 0
    lam = np.ones(d.shape[1])
    W(d, X_tem, lam)
    t0 = time.perf_counter()
    reps = 3 if 'MA' not in name and 'A1' != name else 1
    for _ in range(reps):
        W(d, X_tem, lam)
    dt = (time.perf_counter() - t0) / reps
    total_per_eval += dt
    rows.append((name, act, len(X_tem), dt))

print('%-5s %8s %8s %10s %12s' % ('sub', 'm_act', '|X|', 's/eval', '1200 evals'))
for n, a, x, dt in rows:
    print('%-5s %8d %8d %10.4f %12.1f' % (n, a, x, dt, dt * 1200))

print()
print('sum of s/eval over the 19 subsets     : %.4f' % total_per_eval)
print('one full pass (1200 evals, 1 metric)   : %.0f s  = %.1f min' % (total_per_eval * 1200, total_per_eval * 1200 / 60))
print('AP + ROC-AUC                           : %.0f s  = %.1f min' % (total_per_eval * 2400, total_per_eval * 2400 / 60))
print('3 seeds, AP + ROC-AUC                  : %.0f s  = %.1f h' % (total_per_eval * 2400 * 3, total_per_eval * 2400 * 3 / 3600))
print()
print('current 5-pass greedy CD cost (approx 875 evals per metric):')
print('  one metric, all subsets              : %.0f s = %.1f min' % (total_per_eval * 875, total_per_eval * 875 / 60))
