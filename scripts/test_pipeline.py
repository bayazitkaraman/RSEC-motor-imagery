"""Bounded regression tests; no downloads or full-cohort processing."""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mne
import numpy as np
import pandas as pd
from rsec_eeg.features import rnt_energy, rnt_shared_energy_events
from run_physionet_eegmmi import _extract_baseline_windows, _extract_task_epochs
import run_physionet_surrogate as surrogate
from summarize_primary_statistics import assemble_primary, subject_table, summarize_primary


def raw_fixture(duration=4.0, first_samp=0):
    data = np.random.default_rng(42).normal(size=(3, 9600)) * 1e-5
    raw = mne.io.RawArray(data, mne.create_info(['C3', 'Cz', 'C4'], 160., 'eeg'),
                         first_samp=first_samp, verbose='ERROR')
    raw.set_annotations(mne.Annotations([4., 12., 20.], [duration] * 3, ['T1', 'T2', 'T1']))
    return raw


class PipelineTests(unittest.TestCase):
    def test_finite_input(self):
        epoch = np.ones((3, 640)) * 1e-5
        self.assertTrue(np.isfinite(rnt_energy(epoch)).all())
        for invalid in (np.nan, np.inf, -np.inf):
            for function in (rnt_energy, rnt_shared_energy_events):
                with self.subTest(value=invalid, function=function.__name__):
                    bad = epoch.copy()
                    bad[1, 5] = invalid
                    with self.assertRaisesRegex(ValueError, 'NaN or infinite'):
                        function(bad)
        with self.assertRaisesRegex(ValueError, 'NaN or infinite'):
            rnt_shared_energy_events(np.full_like(epoch, np.nan))

    def test_epoch_caps(self):
        raw = raw_fixture()
        for cap in (None, 0, -1):
            self.assertEqual(_extract_baseline_windows(raw, 4., cap)[0].shape, (15, 3, 640))
            self.assertEqual(_extract_task_epochs(raw, 4., cap)[0].shape, (3, 3, 640))
        self.assertEqual(len(_extract_baseline_windows(raw, 4., 2)[0]), 2)
        self.assertEqual(len(_extract_task_epochs(raw, 4., 2)[0]), 2)

    def test_annotation_containment(self):
        with self.assertRaisesRegex(ValueError, 'annotation interval'):
            _extract_task_epochs(raw_fixture(duration=1.), 4., None)
        self.assertEqual(len(_extract_task_epochs(raw_fixture(first_samp=1600), 4., None)[0]), 3)
        cropped = raw_fixture().crop(tmin=2., tmax=30.)
        self.assertEqual(len(_extract_task_epochs(cropped, 4., None)[0]), 3)
        joined = mne.concatenate_raws([raw_fixture(), raw_fixture()], verbose='ERROR')
        self.assertEqual(len(_extract_task_epochs(joined, 4., None)[0]), 6)
        from datetime import datetime, timezone
        dated = raw_fixture(first_samp=1600)
        dated.set_meas_date(datetime(2020, 1, 1, tzinfo=timezone.utc))
        self.assertEqual(len(_extract_task_epochs(dated, 4., None)[0]), 3)

    def test_primary_input_validation(self):
        imagery = pd.DataFrame({'subject': [2, 1], 'rest': [.2, .3], 'task': [.1, .2]})
        execution = imagery.copy()
        assembled = assemble_primary(imagery, execution)
        self.assertEqual(assembled.subject.tolist(), [1, 2])
        with self.assertRaisesRegex(ValueError, 'unique'):
            subject_table(pd.concat([imagery, imagery]), ['rest', 'task'])
        with self.assertRaisesRegex(ValueError, 'sets differ'):
            assemble_primary(imagery, execution.assign(subject=[2, 3]))
        with self.assertRaisesRegex(ValueError, 'same participant rest'):
            assemble_primary(imagery, execution.assign(rest=[.4, .3]))
        with self.assertRaisesRegex(ValueError, 'finite'):
            subject_table(imagery.assign(task=[np.nan, .2]), ['rest', 'task'])
        control = pd.concat([imagery.assign(data_type=kind, delta_task_minus_rest=-.1)
                             for kind in ('real', 'phase_randomized_surrogate')])
        bad = control.copy()
        bad.loc[bad.data_type.eq('real'), 'rest'] += .01
        bad['delta_task_minus_rest'] = bad.task - bad.rest
        with self.assertRaisesRegex(ValueError, 'do not match'):
            summarize_primary(assembled, bad)
        with self.assertRaisesRegex(ValueError, 'task minus rest'):
            summarize_primary(assembled, control.assign(delta_task_minus_rest=0.))

    def test_empty_surrogate_condition(self):
        with self.assertRaisesRegex(ValueError, 'At least one EEG epoch'):
            surrogate._make_surrogate_segments(np.empty((0, 3, 640)), np.random.default_rng(42))

    def test_surrogate_cli_caps(self):
        class StopBeforeAnalysis(Exception):
            pass

        for cap, expected in ((0, 15), (2, 2)):
            with self.subTest(cap=cap), TemporaryDirectory() as temporary:
                argv = ['surrogate', '--max-epochs', str(cap), '--out-dir', temporary,
                        '--data-dir', temporary]
                with patch.object(sys, 'argv', argv), \
                     patch.object(surrogate, '_read_raws', side_effect=lambda *args: [raw_fixture()]), \
                     patch.object(surrogate, '_make_surrogate_segments', side_effect=StopBeforeAnalysis) as make:
                    with self.assertRaises(StopBeforeAnalysis):
                        surrogate.main()
                    self.assertEqual(len(make.call_args.args[0]), expected)


if __name__ == '__main__':
    unittest.main()
