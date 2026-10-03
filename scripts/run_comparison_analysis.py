"""Matched RSEC and native spectral estimators with fixed participant-level selections."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
from hashlib import sha256
import json
import os
import platform
import time
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import mne
import mne_connectivity
import numpy as np
import pandas as pd
import scipy
from threadpoolctl import threadpool_limits
from study_common import PUBLIC, SUBJECTS, BANDS, RUNS, independent_epochs, connectivity, coincidence, rnt_energy, self_test

FEATURES = ['rsec_d3', 'power_uv2', 'coh', 'plv', 'pli', 'wpli']
SEED = 20261001


def process_subject(subject, data_dir, out_dir, bands, low_frequency, extra_data_dir):
    threadpool_limits(limits=1)
    mne.set_log_level('ERROR')
    rows, selections, counts = [], [], []
    for band_name in bands:
        for cindex, condition in enumerate(RUNS):
            epochs, names, sfreq, mapping = independent_epochs(
                subject, condition, BANDS[band_name], data_dir, extra_data_dir=extra_data_dir)
            if sfreq != 160. or epochs.shape[1:] != (64, 640) or len(epochs) < 15:
                raise ValueError('The matched protocol requires 64 channels, 160 Hz and at least 15 epochs.')
            rsec = np.array([coincidence(rnt_energy(epoch))[0] for epoch in epochs])
            power = (epochs ** 2).mean(axis=(1, 2)) * 1e12
            rng = np.random.default_rng(np.random.SeedSequence([SEED, subject, cindex]))
            repetitions = 1 if condition == 'rest' else 20
            counts.append(dict(subject=subject, band=band_name, condition=condition,
                               available_epochs=len(epochs), selected_epochs=15, selections=repetitions))
            for repetition in range(repetitions):
                indices = np.sort(rng.choice(len(epochs), 15, replace=False))
                spectral_band = (1.25, 31.) if low_frequency else BANDS[band_name]
                values = connectivity(epochs[indices], sfreq, spectral_band)
                values.update(rsec_d3=float(rsec[indices].mean()), power_uv2=float(power[indices].mean()))
                rows.append(dict(subject=subject, band=band_name, condition=condition,
                                 selection=repetition, **values))
                selections.append(dict(subject=subject, band=band_name, condition=condition,
                                       selection=repetition, indices=indices.tolist()))
    folder = Path(out_dir) / f'S{subject:03d}'
    folder.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(folder / 'selection_features.csv', index=False)
    frame.groupby(['subject', 'band', 'condition'])[FEATURES].mean().reset_index().to_csv(
        folder / 'subject_features.csv', index=False)
    pd.DataFrame(counts).to_csv(folder / 'epoch_counts.csv', index=False)
    (folder / 'selections.json').write_text(json.dumps(selections, indent=2))
    return subject


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('mne_data'))
    parser.add_argument('--extra-data-dir', type=Path)
    parser.add_argument('--out-dir', type=Path, default=PUBLIC / 'outputs_comparisons')
    parser.add_argument('--subjects', type=int, nargs='+', default=SUBJECTS)
    parser.add_argument('--bands', choices=list(BANDS), nargs='+', default=list(BANDS))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--low-frequency-check', action='store_true')
    parser.add_argument('--test-only', action='store_true')
    args = parser.parse_args()
    self_test()
    if args.test_only:
        return
    if not set(args.subjects) <= set(SUBJECTS) or len(set(args.subjects)) != len(args.subjects):
        parser.error('Use unique participants from the 106-participant primary cohort.')
    bands = ['broadband'] if args.low_frequency_check else args.bands
    args.out_dir.mkdir(parents=True, exist_ok=True)
    protocol = dict(seed=SEED, selected_epochs=15, task_repeats=20, subjects=args.subjects,
                    bands=bands, spectral_low_check=args.low_frequency_check,
                    python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                    pandas=pd.__version__, mne=mne.__version__, mne_connectivity=mne_connectivity.__version__,
                    code_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    config = args.out_dir / 'protocol.json'
    if config.exists() and json.loads(config.read_text()) != protocol:
        raise ValueError('Output directory contains a different analysis protocol.')
    config.write_text(json.dumps(protocol, indent=2))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(process_subject, s, args.data_dir, args.out_dir, bands,
                            args.low_frequency_check, args.extra_data_dir) for s in args.subjects]
        for job in as_completed(jobs):
            print(f'Completed participant {job.result()}', flush=True)
    frame = pd.concat([pd.read_csv(args.out_dir / f'S{s:03d}/subject_features.csv')
                       for s in sorted(args.subjects)], ignore_index=True)
    filename = 'low_frequency_subject_features.csv' if args.low_frequency_check else 'comparison_subject_features.csv'
    if args.low_frequency_check:
        frame = frame[['subject', 'condition', 'coh', 'plv', 'pli', 'wpli']]
    frame.to_csv(args.out_dir / filename, index=False)


if __name__ == '__main__':
    main()
