"""Regenerate every reported statistical table from the supplied participant/epoch inputs."""
from pathlib import Path
import argparse
import subprocess
import sys
from tempfile import TemporaryDirectory

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CHECKS = {
    'primary': {'primary_statistics.csv': ['contrast']},
    'comparison': {'statistics_54.csv': ['band', 'metric', 'contrast'],
                   'paired_effect_comparisons_45.csv': ['band', 'contrast', 'baseline'],
                   'low_frequency_sensitivity_12.csv': ['metric', 'contrast']},
    'feature': {'sensitivity_statistics.csv': ['setting', 'contrast'],
                'imagery_channel_statistics.csv': ['channel'], 'imagery_roi_statistics.csv': ['region']},
    'cohort': {'cohort_sensitivity.csv': ['cohort', 'contrast']},
    'artifact': {'amplitude_aggregation_statistics.csv': ['family', 'setting', 'metric', 'contrast'],
                 'csd_statistics.csv': ['family', 'setting', 'metric', 'contrast']},
}


def main():
    with TemporaryDirectory(prefix='rsec-reproduction-') as temporary:
        count = 0
        for group, tables in CHECKS.items():
            output = Path(temporary) / group
            subprocess.run([sys.executable, str(ROOT / f'scripts/summarize_{group}_statistics.py'),
                            '--out-dir', str(output)], check=True)
            for name, keys in tables.items():
                actual = pd.read_csv(output / name).set_index(keys).sort_index()
                expected = pd.read_csv(ROOT / 'results/summary' / name).set_index(keys).sort_index()
                if set(actual.columns) != set(expected.columns):
                    raise AssertionError(f'Columns differ: {name}')
                pd.testing.assert_frame_equal(actual[expected.columns], expected,
                                              check_exact=False, atol=1e-11, rtol=1e-10)
                count += 1
                print(f'PASS {name}', flush=True)
        print(f'All {count} reported statistical tables reproduced, including bootstrap interval endpoints.')


if __name__ == '__main__':
    main()
