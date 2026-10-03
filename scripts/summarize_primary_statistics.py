"""Construct the four-test primary family with participant-bootstrap intervals."""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rsec_eeg.stats import holm_adjust

SEED = 20261001
RESAMPLES = 10000


def subject_table(frame, columns):
    if 'subject' not in frame or frame.subject.duplicated().any() or frame.subject.isna().any():
        raise ValueError('Participant IDs must be present, unique, and nonmissing.')
    result = frame.set_index('subject').sort_index()
    if len(result) < 2 or not np.isfinite(result[columns].to_numpy(dtype=float)).all():
        raise ValueError('At least two participants and finite condition values are required.')
    return result


def assemble_primary(imagery, execution):
    imagery = subject_table(imagery, ['rest', 'task'])
    execution = subject_table(execution, ['rest', 'task'])
    if not imagery.index.equals(execution.index):
        raise ValueError('Imagery and execution participant sets differ.')
    if not np.allclose(imagery.rest, execution.rest, atol=1e-12, rtol=0):
        raise ValueError('Imagery and execution must share the same participant rest values.')
    return pd.DataFrame({'rest': imagery.rest, 'imagery': imagery.task,
                         'execution': execution.task}).reset_index()


def describe(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('Paired differences must be finite and contain at least two participants.')
    rng = np.random.default_rng(SEED)
    boot = values[rng.integers(0, len(values), (RESAMPLES, len(values)))].mean(axis=1)
    low, high = np.quantile(boot, [.025, .975])
    sd = values.std(ddof=1)
    p = 1. if np.all(values == 0) else wilcoxon(values, zero_method='wilcox',
                                              alternative='two-sided', correction=False,
                                              method='approx').pvalue
    return dict(n=len(values), mean_delta=float(values.mean()), median_delta=float(np.median(values)),
                ci_low=float(low), ci_high=float(high), p=float(p),
                dz=float(values.mean() / sd) if sd else 0.,
                n_decrease=int((values < 0).sum()), n_increase=int((values > 0).sum()),
                n_equal=int((values == 0).sum()))


def summarize_primary(primary, surrogate):
    primary = subject_table(primary, ['rest', 'imagery', 'execution'])
    if set(surrogate.data_type) != {'real', 'phase_randomized_surrogate'}:
        raise ValueError('Both real and phase-randomized participant summaries are required.')
    groups = {}
    for kind in ('real', 'phase_randomized_surrogate'):
        group = subject_table(surrogate[surrogate.data_type.eq(kind)], ['rest', 'task', 'delta_task_minus_rest'])
        if not primary.index.equals(group.index):
            raise ValueError('Primary and surrogate participant sets differ.')
        if not np.allclose(group.task - group.rest, group.delta_task_minus_rest, atol=1e-12, rtol=0):
            raise ValueError('Surrogate summary difference does not match task minus rest.')
        groups[kind] = group
    real = groups['real']
    if not np.allclose(real.rest, primary.rest, atol=1e-12, rtol=0) or not np.allclose(
            real.task, primary.imagery, atol=1e-12, rtol=0):
        raise ValueError('Real surrogate-control condition means do not match the primary analysis.')
    imagery = primary.imagery - primary.rest
    execution = primary.execution - primary.rest
    surrogate_delta = groups['phase_randomized_surrogate'].task - groups['phase_randomized_surrogate'].rest
    contrasts = {'imagery-rest': imagery, 'execution-rest': execution,
                 'imagery-execution': imagery - execution, 'real-surrogate': imagery - surrogate_delta}
    result = pd.DataFrame([{'contrast': label, **describe(values)} for label, values in contrasts.items()])
    result['p_holm'] = holm_adjust(result.p.to_numpy())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--primary-summary', type=Path, default=ROOT / 'results/summary/primary_subject_summary.csv')
    parser.add_argument('--imagery-summary', type=Path)
    parser.add_argument('--execution-summary', type=Path)
    parser.add_argument('--surrogate-summary', type=Path, default=ROOT / 'results/summary/surrogate_subject_summary.csv')
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'outputs_primary_statistics')
    args = parser.parse_args()
    if (args.imagery_summary is None) != (args.execution_summary is None):
        parser.error('Supply both --imagery-summary and --execution-summary.')
    if args.imagery_summary is not None:
        primary = assemble_primary(pd.read_csv(args.imagery_summary), pd.read_csv(args.execution_summary))
    else:
        primary = pd.read_csv(args.primary_summary)
    result = summarize_primary(primary, pd.read_csv(args.surrogate_summary))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    primary.to_csv(args.out_dir / 'primary_subject_summary.csv', index=False)
    result.to_csv(args.out_dir / 'primary_statistics.csv', index=False)
    (args.out_dir / 'primary_statistics_protocol.json').write_text(json.dumps(dict(
        seed=SEED, resamples=RESAMPLES, interval='95% participant-percentile bootstrap',
        rng='numpy.default_rng; reset per contrast; participant order ascending',
        holm_family=result.contrast.tolist(), wilcoxon='two-sided; wilcox zeros; approx; no correction'), indent=2) + '\n')
    print(result.to_string(index=False))


if __name__ == '__main__':
    main()
