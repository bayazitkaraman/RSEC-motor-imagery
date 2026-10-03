"""Compute amplitude-sensitivity inputs and surface-Laplacian features."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import time
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
import mne
import numpy as np
import pandas as pd
import scipy
from scipy.signal import hilbert
from threadpoolctl import threadpool_limits
from run_feature_analysis import SUBJECTS, PUBLIC, read_condition, coincidence

CS_PARAMS = {'sphere': 'auto', 'lambda2': 1e-5, 'stiffness': 4, 'n_legendre_terms': 50}


def mapped_values(epoch):
    analytic = hilbert(epoch, axis=-1)
    q = np.abs(analytic / (np.abs(analytic).max(axis=-1, keepdims=True) + 1e-12))**2
    return {'rsec':-np.log(np.sqrt(1. / (q + 1.)) + 1e-12),
            'linear':q * np.log(2.) / 2.}, q


def marginal_expectation(values):
    bins = np.rint(values * 1000).astype(np.int64)
    bins = bins - bins.min()
    n_channels, n_times = bins.shape
    width = int(bins.max()) + 1
    keys = bins + np.arange(n_channels)[:, None] * width
    hist = np.bincount(keys.ravel(), minlength=n_channels * width).reshape(n_channels, width)
    probability = hist.astype(float) / n_times
    expected = ((probability.sum(axis=0)**2).sum() - (probability**2).sum()) / (n_channels*(n_channels-1))
    return float(expected)


def direct_observed(values):
    rounded = np.round(values, decimals=3)
    pairs = [float(np.mean(rounded[i] == rounded[j]))
             for i in range(len(values)) for j in range(i+1, len(values))]
    return float(np.mean(pairs))


def direct_expectation(values):
    rounded = np.round(values, decimals=3)
    pairs = [float(np.mean(rounded[i, :, None] == rounded[j, None, :]))
             for i in range(len(values)) for j in range(i+1, len(values))]
    return float(np.mean(pairs))


def self_test():
    rng = np.random.default_rng(20261002)
    for values in [np.zeros((3, 8)), rng.integers(0, 8, (4, 9))/1000,
                   np.tile(np.arange(8)/1000, (3, 1)), rng.random((5, 13))]:
        np.testing.assert_allclose(marginal_expectation(values), direct_expectation(values), atol=1e-14)
        np.testing.assert_allclose(coincidence(values)[0], direct_observed(values), atol=1e-14)
        permuted = np.stack([row[rng.permutation(len(row))] for row in values])
        np.testing.assert_allclose(marginal_expectation(values), marginal_expectation(permuted), atol=1e-14)
    synchronized = np.tile(np.arange(8)/1000, (3, 1))
    assert coincidence(synchronized)[0] == 1.
    assert marginal_expectation(synchronized) == .125
    print('PASS: marginal expectation, independent observed counter, permutation invariance.', flush=True)


def process_subject(subject, data_dir, output):
    threadpool_limits(limits=1)
    mne.set_log_level('ERROR')
    start = time.monotonic()
    rows, audits, sources = [], [], []
    for condition in ('rest', 'imagery', 'execution'):
        data, names, sfreq, provenance = read_condition(subject, condition, Path(data_dir), (1., 31.))
        sources.extend(provenance)
        info = mne.create_info(names, sfreq, ch_types='eeg')
        info.set_montage(mne.channels.make_standard_montage('standard_1005'), match_case=False)
        epochs = mne.EpochsArray(data, info, baseline=None, verbose='ERROR')
        csd = mne.preprocessing.compute_current_source_density(epochs, **CS_PARAMS, verbose='ERROR').get_data(copy=True)
        assert csd.shape == data.shape and np.isfinite(csd).all()
        for index, (native, transformed) in enumerate(zip(data, csd)):
            ptp = np.ptp(native, axis=1) * 1e6
            front = [i for i, name in enumerate(names) if name in ('Fp1', 'Fpz', 'Fp2')]
            mapping, q = mapped_values(native)
            transformed_mapping, _ = mapped_values(transformed)
            row = {'subject':subject, 'condition':condition, 'epoch':index,
                   'max_ptp_uv':float(ptp.max()), 'frontopolar_max_ptp_uv':float(ptp[front].max()),
                   'any_channel_over_150uv':int(ptp.max()>150), 'frontopolar_over_150uv':int(ptp[front].max()>150),
                   'mean_q':float(q.mean()), 'fraction_q_below_001':float((q<.01).mean())}
            for metric, values in mapping.items():
                observed = coincidence(values)[0]
                expected = marginal_expectation(values)
                row.update({metric:observed, metric+'_expected':expected,
                            metric+'_excess':observed-expected,
                            metric+'_csd':coincidence(transformed_mapping[metric])[0]})
                if condition=='rest' and index==0:
                    direct_error = abs(observed-direct_observed(values))
                    assert direct_error < 1e-12
                    audits.append({'metric':metric, 'direct_counter_error':direct_error})
            rows.append(row)
    folder = Path(output) / f'S{subject:03d}'
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(folder/'epochs.csv', index=False)
    (folder/'audit.json').write_text(json.dumps(audits, indent=2))
    (folder/'sources.json').write_text(json.dumps(sources, indent=2))
    (folder/'complete.json').write_text(json.dumps({'subject':subject, 'seconds':time.monotonic()-start}))
    return subject, time.monotonic()-start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('mne_data'))
    parser.add_argument('--out-dir', type=Path, default=PUBLIC / 'outputs_artifacts')
    parser.add_argument('--subjects', type=int, nargs='+', default=SUBJECTS)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--test-only', action='store_true')
    args = parser.parse_args()
    self_test()
    if args.test_only:
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(process_subject, s, args.data_dir, args.out_dir) for s in args.subjects]
        for job in as_completed(jobs):
            subject, seconds = job.result()
            print(f'Completed participant {subject} in {seconds:.1f}s', flush=True)
    frame = pd.concat([pd.read_csv(args.out_dir / f'S{s:03d}/epochs.csv')
                       for s in sorted(args.subjects)], ignore_index=True)
    frame.sort_values(['subject', 'condition', 'epoch']).to_csv(
        args.out_dir / 'artifact_epoch_features.csv', index=False)
    (args.out_dir / 'protocol.json').write_text(json.dumps(dict(
        subjects=args.subjects, filter_band=[1., 31.], epoch_seconds=4., csd=CS_PARAMS,
        mne=mne.__version__, numpy=np.__version__, scipy=scipy.__version__,
        python=platform.python_version(), code_sha256=sha256(Path(__file__).read_bytes()).hexdigest()), indent=2))


if __name__ == '__main__':
    main()
