"""Compute epoch-level RSEC, precision, reference, band and sensor features."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import sys
import time
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[variable] = '1'
import mne
import numpy as np
import pandas as pd
import scipy
from scipy.signal import hilbert
from threadpoolctl import threadpool_limits
from study_common import PUBLIC, SUBJECTS, RUNS, BANDS, source_path, coincidence, rnt_energy, rnt_shared_energy_events
from run_physionet_eegmmi import _extract_baseline_windows, _extract_task_epochs
FP = {'Fp1', 'Fpz', 'Fp2'}


def feature_values(epoch, precision=False):
    analytic = hilbert(epoch, axis=-1)
    eta = analytic / (np.abs(analytic).max(axis=-1, keepdims=True) + 1e-12)
    q = np.abs(eta)**2
    energy = -np.log(np.sqrt(1. / (q + 1.)) + 1e-12)
    density, nodes = coincidence(energy)
    row = {'rsec_d3':density, 'power_uv2':float(np.mean(epoch**2) * 1e12),
           'normalized_power':float(q.mean()),
           'linear_q_coincidence':coincidence(q * np.log(2.) / 2.)[0]}
    phase = np.exp(1j * np.angle(analytic))
    i, j = np.triu_indices(epoch.shape[0], 1)
    row['plv'] = float(np.abs((phase @ phase.conj().T) / epoch.shape[1])[i, j].mean())
    # Scale each channel for numerical stability; positive scaling cancels in wPLI.
    scaled = analytic / np.maximum(np.abs(analytic).max(axis=1, keepdims=True), np.finfo(float).tiny)
    cross_imag = np.imag(scaled[i] * scaled[j].conj())
    denominator = np.abs(cross_imag).sum(axis=1)
    pair_wpli = np.divide(np.abs(cross_imag.sum(axis=1)), denominator,
                          out=np.zeros_like(denominator), where=denominator > 0)
    row['wpli'] = float(pair_wpli.mean())
    if precision:
        for d in (2, 4, 5):
            row[f'rsec_d{d}'] = coincidence(energy, d)[0]
    return row, nodes


def read_condition(subject, condition, data_dir, band):
    raws = []
    source_files = []
    for run in RUNS[condition]:
        path = source_path(data_dir, subject, run)
        if not path.is_file():
            raise FileNotFoundError(path)
        raw = mne.io.read_raw_edf(path, preload=True, verbose='ERROR')
        mne.datasets.eegbci.standardize(raw)
        raw.pick_types(eeg=True, verbose='ERROR')
        raw.set_montage(mne.channels.make_standard_montage('standard_1005'), match_case=False, on_missing='ignore')
        raw.filter(*band, fir_design='firwin', verbose='ERROR')
        raws.append(raw)
        source_files.append({'path':str(path), 'sha256':sha256(path.read_bytes()).hexdigest()})
    raw = raws[0] if len(raws)==1 else mne.concatenate_raws(raws, verbose='ERROR')
    extract = _extract_baseline_windows if condition=='rest' else _extract_task_epochs
    epochs, names, sfreq = extract(raw, 4., None)
    return epochs, names, sfreq, source_files


def process_subject(subject, data_dir, out_dir):
    threadpool_limits(limits=1)
    mne.set_log_level('ERROR')
    started = time.monotonic()
    rows, nodes, counts, quality, provenance = [], [], [], [], []
    reference_names = None
    reference_rate = None
    for band_name, band in BANDS.items():
        for condition in RUNS:
            epochs, names, sfreq, sources = read_condition(subject, condition, Path(data_dir), band)
            if reference_names is None:
                reference_names, reference_rate = names, sfreq
            if names != reference_names or sfreq != reference_rate:
                raise ValueError(f'Subject {subject}: inconsistent channels or sampling rate')
            assert len(names)==64 and np.isfinite(epochs).all()
            assert len(epochs)>0
            if condition=='rest':
                assert len(epochs)==15
            counts.append({'subject':subject, 'band':band_name, 'condition':condition,
                           'n_epochs':len(epochs), 'sfreq':sfreq, 'samples_per_epoch':epochs.shape[-1]})
            if band_name=='broadband':
                provenance.extend(sources)
            keep = [i for i, name in enumerate(names) if name not in FP]
            front = [i for i, name in enumerate(names) if name in FP]
            for index, epoch in enumerate(epochs):
                values, node = feature_values(epoch, precision=band_name=='broadband')
                base = {'subject':subject, 'condition':condition, 'band':band_name,
                        'reference':'native', 'epoch':index}
                rows.append({**base, **values})
                if band_name=='broadband':
                    rows.append({**base, 'reference':'average',
                                 'rsec_d3':coincidence(rnt_energy(epoch - epoch.mean(axis=0, keepdims=True)))[0]})
                    rows.append({**base, 'reference':'no_frontopolar',
                                 'rsec_d3':coincidence(rnt_energy(epoch[keep]))[0]})
                    for channel, score in zip(names, node):
                        nodes.append({'subject':subject, 'condition':condition, 'epoch':index,
                                      'channel':channel, 'participation':float(score)})
                    ptp = np.ptp(epoch, axis=1) * 1e6
                    quality.append({'subject':subject, 'condition':condition, 'epoch':index,
                                    'max_ptp_uv':float(ptp.max()), 'frontopolar_max_ptp_uv':float(ptp[front].max()),
                                    'any_channel_over_150uv':int(ptp.max()>150),
                                    'frontopolar_over_150uv':int(ptp[front].max()>150)})
    folder = Path(out_dir) / f'S{subject:03d}'
    folder.mkdir(parents=True, exist_ok=True)
    for name, data in [('epochs',rows), ('nodes',nodes), ('epoch_counts',counts), ('quality',quality)]:
        pd.DataFrame(data).to_csv(folder / f'{name}.csv', index=False)
    (folder / 'sources.json').write_text(json.dumps(provenance, indent=2))
    (folder / 'complete.json').write_text(json.dumps({'subject':subject, 'seconds':time.monotonic()-started}))
    return subject, time.monotonic()-started


def self_test():
    rng = np.random.default_rng(824)
    for epoch in [rng.normal(size=(8, 640)), np.zeros((8, 640)), rng.normal(size=(64, 640))*1e-5]:
        for d in (2, 3, 4, 5):
            expected, _, enodes = rnt_shared_energy_events(epoch, decimals=d)
            got, gnodes = coincidence(rnt_energy(epoch), d)
            np.testing.assert_allclose(got, expected, rtol=0, atol=1e-15)
            np.testing.assert_allclose(gnodes, enodes, rtol=0, atol=1e-15)
    t = np.arange(640)/160
    locked = np.stack([np.sin(2*np.pi*10*t), np.sin(2*np.pi*10*t+.7)])
    values, _ = feature_values(locked)
    assert np.isclose(values['plv'], 1.) and np.isclose(values['wpli'], 1.)
    scaled, _ = feature_values(locked*np.array([1e-6, 1e-4])[:,None])
    np.testing.assert_allclose([scaled['plv'], scaled['wpli']], [values['plv'], values['wpli']], atol=1e-12)
    zero, _ = feature_values(np.zeros((3, 640)))
    assert zero['wpli']==0 and zero['rsec_d3']==1
    print('PASS: optimized RSEC matches reference for four precisions; PLV/wPLI synthetic controls and scaling.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path('mne_data'))
    parser.add_argument('--out-dir', type=Path, default=PUBLIC/'outputs_features')
    parser.add_argument('--subjects', nargs='+', type=int, default=SUBJECTS)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--test-only', action='store_true')
    args = parser.parse_args()
    self_test()
    if args.test_only:
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)
    code_hash = sha256(Path(__file__).read_bytes()).hexdigest()
    manifest = {'protocol_date':'2026-10-01', 'python':platform.python_version(), 'mne':mne.__version__,
                'numpy':np.__version__, 'scipy':scipy.__version__, 'pandas':pd.__version__,
                'code_sha256':code_hash, 'subjects':args.subjects, 'data_dir':str(args.data_dir),
                'bands':BANDS, 'runs':RUNS, 'workers':args.workers, 'max_epochs':None,
                'public_feature_sha256':sha256((PUBLIC/'rsec_eeg/features.py').read_bytes()).hexdigest()}
    existing = args.out_dir/'manifest.json'
    if existing.exists():
        old = json.loads(existing.read_text())
        assert old['code_sha256']==code_hash, 'Code changed; use a new output directory'
    existing.write_text(json.dumps(manifest, indent=2))
    todo = [s for s in args.subjects if not (args.out_dir/f'S{s:03d}/complete.json').exists()]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_subject, s, str(args.data_dir), str(args.out_dir)):s for s in todo}
        for i, future in enumerate(as_completed(futures), 1):
            subject, seconds = future.result()
            print(f'[{i}/{len(todo)}] S{subject:03d} complete, {seconds:.1f}s', flush=True)
    print('COMPLETE', args.out_dir, flush=True)


if __name__ == '__main__':
    main()
