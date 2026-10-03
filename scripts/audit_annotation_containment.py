"""Audit four-second task windows in local EEGMMI EDF annotation intervals."""
from pathlib import Path
import argparse
from hashlib import sha256
import json

import mne
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('mne_data'))
    parser.add_argument('--extra-data-dir', type=Path)
    parser.add_argument('--subjects', type=int, nargs='+', default=list(range(1, 110)))
    parser.add_argument('--out-dir', type=Path, default=Path('results/summary'))
    args = parser.parse_args()
    rows = []
    for subject in args.subjects:
        for run in (3, 4, 7, 8, 11, 12):
            relative = Path(f'S{subject:03d}/S{subject:03d}R{run:02d}.edf')
            candidates = [args.data_dir / 'MNE-eegbci-data/files/eegmmidb/1.0.0' / relative,
                          args.data_dir / relative]
            if args.extra_data_dir is not None:
                candidates.append(args.extra_data_dir / relative)
            path = next((p for p in candidates if p.is_file()), None)
            if path is None:
                raise FileNotFoundError(relative)
            raw = mne.io.read_raw_edf(path, preload=False, verbose='ERROR')
            sfreq = float(raw.info['sfreq'])
            count = int(round(4. * sfreq))
            annotations = raw.annotations
            samples = raw.time_as_index(annotations.onset, use_rounding=True,
                                        origin=annotations.orig_time)
            if annotations.orig_time is not None:
                samples += raw.first_samp
            checks = []
            for index, annotation in enumerate(annotations):
                if annotation['description'].upper() not in ('T1', 'T2'):
                    continue
                start = int(samples[index]) - raw.first_samp
                stop = start + count
                end_time = (stop + raw.first_samp) / sfreq
                label_end = annotation['onset'] + annotation['duration']
                next_onset = (annotations.onset[index + 1] if index + 1 < len(annotations)
                              else raw.first_time + raw.n_times / sfreq)
                tolerance = .5 / sfreq + 1e-10
                checks.append(dict(in_record=0 <= start and stop <= raw.n_times,
                                   in_annotation=end_time <= label_end + tolerance,
                                   before_next=end_time <= next_onset + tolerance,
                                   duration=float(annotation['duration'])))
            if not checks:
                raise ValueError(f'No task annotations: {relative}')
            rows.append(dict(subject=subject, run=run, primary_cohort=subject not in (88, 92, 100),
                             file=relative.as_posix(), sha256=sha256(path.read_bytes()).hexdigest(),
                             sfreq=sfreq, task_epochs=len(checks),
                             within_record=sum(c['in_record'] for c in checks),
                             within_annotation=sum(c['in_record'] and c['in_annotation'] for c in checks),
                             before_next_annotation=sum(c['in_record'] and c['before_next'] for c in checks),
                             minimum_task_duration=min(c['duration'] for c in checks)))
    result = pd.DataFrame(rows)
    # Incomplete end-of-record candidates are not retained by the epoch extractor.
    valid = result[['within_annotation', 'before_next_annotation']].eq(result.within_record, axis=0)
    primary = result[result.primary_cohort]
    report = dict(recordings=len(result), subjects=len(args.subjects),
                  task_candidates=int(result.task_epochs.sum()), primary_recordings=len(primary),
                  retained_task_epochs=int(result.within_record.sum()),
                  primary_retained_task_epochs=int(primary.within_record.sum()),
                  incomplete_candidates=int((result.task_epochs - result.within_record).sum()),
                  incomplete_records=result[result.task_epochs != result.within_record][
                      ['subject', 'run', 'task_epochs', 'within_record']].to_dict('records'),
                  all_retained_windows_contained=bool(valid.all().all()),
                  interval_tolerance='half a sample for annotation-onset rounding',
                  mne=mne.__version__, code_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out_dir / 'annotation_containment.csv', index=False)
    (args.out_dir / 'annotation_containment.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if not report['all_retained_windows_contained']:
        raise RuntimeError('Some task windows are not contained; inspect the audit before rerunning analyses.')


if __name__ == '__main__':
    main()
