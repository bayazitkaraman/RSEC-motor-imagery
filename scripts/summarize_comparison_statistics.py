"""Recompute the 54 tests and all 45 paired-bootstrap intervals from participant features."""
from pathlib import Path
import argparse
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SUBJECTS = [s for s in range(1, 110) if s not in (88, 92, 100)]
BANDS = ('broadband', 'mu', 'beta')
METHODS = ('rsec_d3', 'power_uv2', 'coh', 'plv', 'pli', 'wpli')
CONTRASTS = [('imagery-rest', 'imagery', 'rest'),
             ('execution-rest', 'execution', 'rest'),
             ('imagery-execution', 'imagery', 'execution')]
SEED = 20261002


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    result = np.empty_like(p)
    result[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    independent = np.array([min(1, max(p[j] * (len(p) - k)
                                      for k, j in enumerate(order) if p[j] <= value))
                            for value in p])
    np.testing.assert_allclose(result, independent, rtol=0, atol=1e-15)
    return result


def describe(delta):
    rng = np.random.default_rng(SEED)
    mean_boot = delta[rng.integers(0, len(delta), (10000, len(delta)))].mean(axis=1)
    low, high = np.quantile(mean_boot, [.025, .975])
    return dict(n=len(delta), mean_delta=float(delta.mean()), median_delta=float(np.median(delta)),
                dz=float(delta.mean()/delta.std(ddof=1)),
                p=float(stats.wilcoxon(delta, zero_method='wilcox', correction=False, method='approx').pvalue),
                ci_low=float(low), ci_high=float(high), n_decrease=int((delta < 0).sum()),
                n_increase=int((delta > 0).sum()), n_zero=int((delta == 0).sum()))


def calculate(frame):
    keys = ['subject', 'band', 'condition']
    if frame.duplicated(keys).any() or len(frame) != 106 * 9:
        raise ValueError('Expected one row per participant, band and condition (954 rows).')
    rows, delta_groups, group_keys = [], [], []
    for band in BANDS:
        wides = {metric: frame[frame.band == band].pivot(index='subject', columns='condition', values=metric).loc[SUBJECTS]
                 for metric in METHODS}
        for label, a, b in CONTRASTS:
            deltas = np.column_stack([(wides[metric][a] - wides[metric][b]).to_numpy() for metric in METHODS])
            assert deltas.shape == (106, 6) and np.isfinite(deltas).all()
            delta_groups.append(deltas)
            group_keys.append((band, label))
            for column, metric in enumerate(METHODS):
                rows.append(dict(band=band, metric=metric, contrast=label,
                                 mean_first=float(wides[metric][a].mean()),
                                 mean_second=float(wides[metric][b].mean()), **describe(deltas[:, column])))
    result = pd.DataFrame(rows)
    assert len(result) == 54
    result['p_holm_54'] = holm(result.p.to_numpy())
    # Resample participants jointly across every method, band and contrast.
    matrix = np.stack(delta_groups, axis=1)
    point = np.abs(matrix.mean(axis=0)/matrix.std(axis=0, ddof=1))
    rng = np.random.default_rng(SEED)
    bootstrap = np.empty((100000, 9, 6))
    zero_variance = np.zeros((9, 6), dtype=int)
    for start in range(0, len(bootstrap), 500):
        sample = matrix[rng.integers(0, len(SUBJECTS), (500, len(SUBJECTS)))]
        sd = sample.std(axis=1, ddof=1)
        zero_variance += (sd == 0).sum(axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            bootstrap[start:start+500] = np.abs(sample.mean(axis=1)/sd)
    assert not zero_variance.any(), f'Degenerate draws must be reported, not removed: {zero_variance}'
    assert np.isfinite(bootstrap).all()
    difference = bootstrap[:, :, 1:] - bootstrap[:, :, :1]
    tail = .05/(2*45)
    low, high = np.quantile(difference, [tail, 1-tail], axis=0)
    comparison_rows = []
    for group, (band, label) in enumerate(group_keys):
        for column, metric in enumerate(METHODS[1:]):
            comparison_rows.append(dict(band=band, contrast=label, baseline=metric,
                                        rsec_abs_dz=float(point[group, 0]), baseline_abs_dz=float(point[group, column+1]),
                                        baseline_minus_rsec_abs_dz=float(point[group, column+1]-point[group, 0]),
                                        ci_low=float(low[group, column]), ci_high=float(high[group, column]),
                                        interval_confidence=1-.05/45,
                                        direction_supported=('baseline_larger' if low[group, column] > 0 else
                                                             'rsec_larger' if high[group, column] < 0 else 'unresolved')))
    comparisons = pd.DataFrame(comparison_rows)
    assert len(comparisons) == 45
    return result, comparisons



def low_frequency_statistics(frame, reference):
    if len(frame) != 318 or frame.duplicated(['subject', 'condition']).any():
        raise ValueError('Expected 318 unique low-frequency participant-condition rows.')
    rows = []
    for metric in METHODS[2:]:
        wide = frame.pivot(index='subject', columns='condition', values=metric).loc[SUBJECTS]
        for label, a, b in CONTRASTS:
            rows.append(dict(metric=metric, contrast=label, **describe((wide[a] - wide[b]).to_numpy())))
    result = pd.DataFrame(rows)
    result['p_holm_12'] = holm(result.p.to_numpy())
    original = reference[(reference.band == 'broadband') & reference.metric.isin(METHODS[2:])]
    result = result.merge(original[['metric', 'contrast', 'dz']], on=['metric', 'contrast'],
                          suffixes=('', '_1hz'), validate='one_to_one')
    result['dz_change'] = result.dz - result.dz_1hz
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'results/summary/comparison_subject_features.csv')
    parser.add_argument('--low-frequency-input', type=Path, default=ROOT / 'results/summary/low_frequency_subject_features.csv')
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'outputs_comparison_statistics')
    args = parser.parse_args()
    result, comparisons = calculate(pd.read_csv(args.input))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out_dir / 'statistics_54.csv', index=False)
    comparisons.to_csv(args.out_dir / 'paired_effect_comparisons_45.csv', index=False)
    if args.low_frequency_input.exists():
        low_frequency_statistics(pd.read_csv(args.low_frequency_input), result).to_csv(
            args.out_dir / 'low_frequency_sensitivity_12.csv', index=False)
    print('Recomputed 54 tests and 45 intervals using 100000 joint participant resamples.')


if __name__ == '__main__':
    main()
