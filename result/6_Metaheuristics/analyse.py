"""Final analysis of the metaheuristic benchmark: paired comparison vs GreedyCD."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

RESULT = Path('result/6_Metaheuristics')


def load(tag):
    path = RESULT / f'Metaheuristic_Comparison_{tag}.csv'
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[df['Status'] == 'ok'].copy()
    df['Best_Metric'] = pd.to_numeric(df['Best_Metric'], errors='coerce')
    df['Evaluations'] = pd.to_numeric(df['Evaluations'], errors='coerce')
    df['Wall_Sec'] = pd.to_numeric(df['Wall_Sec'], errors='coerce')
    return df


def report(tag, title):
    df = load(tag)
    if df is None:
        print(f'--- {title}: no data yet ---')
        return
    print('=' * 108)
    print(title)
    print('=' * 108)

    # mean over subsets and seeds, per method
    g = df.groupby('Method')
    agg = g.agg(mean=('Best_Metric', 'mean'),
                std=('Best_Metric', 'std'),
                worst=('Best_Metric', 'min'),
                best=('Best_Metric', 'max'),
                evals=('Evaluations', 'mean'),
                secs=('Wall_Sec', 'mean'),
                n=('Best_Metric', 'size')).sort_values('mean', ascending=False)

    # paired comparison against GreedyCD (same subset + same seed)
    base = df[df['Method'] == 'GreedyCD'].set_index(['Subset', 'Seed'])['Best_Metric']
    print('%-19s %8s %8s %8s %8s %9s %8s  %s' % ('method', 'mean', 'std', 'worst', 'best',
                                                 'evals', 'sec', 'paired vs GreedyCD'))
    for method in agg.index:
        row = agg.loc[method]
        sub = df[df['Method'] == method].set_index(['Subset', 'Seed'])['Best_Metric']
        common = base.index.intersection(sub.index)
        if len(common) and method != 'GreedyCD':
            d = (sub.loc[common] - base.loc[common])
            wins = int((d > 1e-9).sum())
            losses = int((d < -1e-9).sum())
            ties = len(d) - wins - losses
            paired = f'{wins}W/{losses}L/{ties}T  mean d={d.mean():+.4f}'
        else:
            paired = '(reference)'
        print('%-19s %8.4f %8.4f %8.4f %8.4f %9.0f %8.1f  %s'
              % (method, row['mean'], row['std'], row['worst'], row['best'],
                 row['evals'], row['secs'], paired))

    print()
    print('per subset (mean over seeds):')
    piv = df.pivot_table(index='Subset', columns='Method', values='Best_Metric', aggfunc='mean')
    print(piv.round(4).to_string())
    print()


for tag, title in [
    ('ap_budget1200_main', 'RUN 1 -- AP, budget 1200, 11 methods, 5 subsets, seeds 1-3'),
    ('ap_budget1200_heavy', 'RUN 2 -- AP, budget 1200, 6 methods, A1 + MA2 (heavy), seeds 1-2'),
    ('auroc_budget1200_main', 'RUN 3 -- ROC-AUC, budget 1200, 6 methods, 5 subsets, seeds 1-3'),
]:
    report(tag, title)
