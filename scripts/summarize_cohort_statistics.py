"""Recompute five cohort/rate configurations as one 15-test family."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from summarize_primary_statistics import holm_adjust

ROOT = Path(__file__).resolve().parents[1]
SUBJECTS = [s for s in range(1, 110) if s not in (88, 92, 100)]
CONTRASTS = [('imagery-rest', 'imagery', 'rest'),
             ('execution-rest', 'execution', 'rest'),
             ('imagery-execution', 'imagery', 'execution')]
COHORTS = {
    'A_original106_160Hz': SUBJECTS,
    'B_original105_without89_160Hz': [s for s in SUBJECTS if s != 89],
    'C_all109_160Hz': list(range(1, 110)),
    'D_108_without89_160Hz': [s for s in range(1, 110) if s != 89],
    'E_all109_128Hz': list(range(1, 110)),
}


def describe(values, seed=20261002):
    x = np.asarray(values, dtype=float)
    assert x.ndim == 1 and len(x) > 1 and np.isfinite(x).all()
    rng = np.random.default_rng(seed)
    boot = x[rng.integers(0, len(x), (10000, len(x)))].mean(axis=1)
    low, high = np.quantile(boot, [.025, .975])
    p = (1. if np.all(x == 0) else stats.wilcoxon(
        x, alternative='two-sided', zero_method='wilcox',
        correction=False, method='approx').pvalue)
    sd = x.std(ddof=1)
    return dict(n=len(x), mean_delta=float(x.mean()), median_delta=float(np.median(x)),
                dz=float(x.mean()/sd) if sd else 0., p=float(p),
                ci_low=float(low), ci_high=float(high),
                n_decrease=int((x < 0).sum()), n_increase=int((x > 0).sum()),
                n_equal=int((x == 0).sum()))


def wide(frame, subjects, metric='rsec_d3'):
    result = frame.pivot(index='subject', columns='condition', values=metric).loc[subjects]
    assert result.shape == (len(subjects), 3) and np.isfinite(result).all().all()
    return result


def paired_rows(frame, subjects, seed=20261002, metric='rsec_d3'):
    values = wide(frame, subjects, metric)
    return [{'contrast':label, 'mean_first':float(values[a].mean()),
             'mean_second':float(values[b].mean()), **describe(values[a]-values[b], seed)}
            for label, a, b in CONTRASTS]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=ROOT / 'results/summary')
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'outputs_cohort_statistics')
    args = parser.parse_args()
    all160 = pd.read_csv(args.input_dir / 'cohort_subject_means_160hz.csv')
    all128 = pd.read_csv(args.input_dir / 'cohort_subject_means_128hz.csv')
    for frame in (all160, all128):
        if frame.duplicated(['subject', 'condition']).any() or len(frame) != 327:
            raise ValueError('Expected 327 unique participant-condition rows per sampling rate.')
    rows = []
    for cohort, subjects in COHORTS.items():
        source = all128 if cohort.startswith('E_') else all160
        rows.extend([dict(cohort=cohort, **row) for row in paired_rows(source, subjects)])
    result = pd.DataFrame(rows)
    result['p_holm_within3'] = result.groupby('cohort').p.transform(holm_adjust)
    result['p_holm_all15'] = holm_adjust(result.p.to_numpy())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out_dir / 'cohort_sensitivity.csv', index=False)
    print('Recomputed all 15 cohort/rate comparisons.')


if __name__ == '__main__':
    main()
