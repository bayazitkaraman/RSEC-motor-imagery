"""Compute all-participant broadband RSEC at common 160-Hz and 128-Hz rates."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from study_common import PUBLIC, RUNS, independent_epochs, coincidence, rnt_energy


def process_subject(subject, rate, data_dir, extra_data_dir):
    threadpool_limits(limits=1)
    rows = []
    for condition in RUNS:
        epochs, names, sfreq, mapping = independent_epochs(
            subject, condition, (1., 31.), data_dir, target_sfreq=rate, extra_data_dir=extra_data_dir)
        if sfreq != rate or len(names) != 64 or not len(epochs):
            raise ValueError('Incomplete common-rate participant input.')
        density = [coincidence(rnt_energy(epoch))[0] for epoch in epochs]
        rows.append(dict(subject=subject, condition=condition, rsec_d3=float(np.mean(density))))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('mne_data'))
    parser.add_argument('--extra-data-dir', type=Path)
    parser.add_argument('--out-dir', type=Path, default=PUBLIC / 'outputs_cohort')
    parser.add_argument('--subjects', type=int, nargs='+', default=list(range(1, 110)))
    parser.add_argument('--rates', type=int, nargs='+', choices=[128, 160], default=[160, 128])
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for rate in args.rates:
        rows = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs = [pool.submit(process_subject, s, rate, args.data_dir, args.extra_data_dir) for s in args.subjects]
            for job in as_completed(jobs):
                records = job.result()
                rows.extend(records)
                print(f'Completed {rate}-Hz participant {records[0]["subject"]}', flush=True)
        pd.DataFrame(rows).sort_values(['subject', 'condition']).to_csv(
            args.out_dir / f'cohort_subject_means_{rate}hz.csv', index=False)
    (args.out_dir / 'protocol.json').write_text(json.dumps(dict(
        subjects=args.subjects, rates=args.rates, filter_band=[1., 31.], epoch_seconds=4.,
        resampling='MNE FFT on continuous recordings with jointly mapped events, before FIR filtering'), indent=2))


if __name__ == '__main__':
    main()
