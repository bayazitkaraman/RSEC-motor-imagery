"""Reproduce seven-setting RSEC sensitivity and participant-level spatial statistics."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from summarize_primary_statistics import describe, holm_adjust

ROOT = Path(__file__).resolve().parents[1]
SUBJECTS = [s for s in range(1, 110) if s not in (88, 92, 100)]
FEATURES = ['rsec_d3', 'rsec_d2', 'rsec_d4', 'rsec_d5', 'power_uv2',
            'normalized_power', 'linear_q_coincidence', 'plv', 'wpli']


def contrasts(wide):
    imagery, execution = wide.imagery - wide.rest, wide.execution - wide.rest
    return {'imagery-rest': imagery, 'execution-rest': execution,
            'imagery-execution': imagery - execution}


def adjust(rows):
    result = pd.DataFrame(rows)
    result['p_holm'] = holm_adjust(result.p.to_numpy())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--input', type=Path, default=ROOT / 'results/summary/feature_subject_summary.csv')
    parser.add_argument('--nodes', type=Path, default=ROOT / 'results/summary/imagery_node_deltas_by_subject.csv')
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'outputs_feature_statistics')
    args = parser.parse_args()
    if args.run_dir is not None:
        epochs = pd.concat([pd.read_csv(args.run_dir / f'S{s:03d}/epochs.csv') for s in SUBJECTS], ignore_index=True)
        means = epochs.groupby(['subject', 'band', 'reference', 'condition'])[FEATURES].mean().reset_index()
        nodes = pd.concat([pd.read_csv(args.run_dir / f'S{s:03d}/nodes.csv') for s in SUBJECTS], ignore_index=True)
        node_means = nodes.groupby(['subject', 'condition', 'channel']).participation.mean().unstack('condition')
        node_delta = (node_means.imagery - node_means.rest).unstack('channel').loc[SUBJECTS]
    else:
        means = pd.read_csv(args.input)
        # Preserve float64 rank ties from the original participant-level calculation.
        node_delta = pd.read_csv(args.nodes, float_precision='round_trip').set_index('subject').loc[SUBJECTS]
    if means.duplicated(['subject', 'band', 'reference', 'condition']).any():
        raise ValueError('Duplicate feature-summary keys.')

    def wide(metric='rsec_d3', band='broadband', reference='native'):
        part = means[(means.band == band) & (means.reference == reference)]
        result = part.pivot(index='subject', columns='condition', values=metric).loc[SUBJECTS]
        if result.shape != (106, 3) or not np.isfinite(result.to_numpy()).all():
            raise ValueError('Incomplete participant/condition feature input.')
        return result

    settings = [('d2', 'rsec_d2', 'broadband', 'native'), ('d4', 'rsec_d4', 'broadband', 'native'),
                ('d5', 'rsec_d5', 'broadband', 'native'), ('average-reference', 'rsec_d3', 'broadband', 'average'),
                ('no-frontopolar', 'rsec_d3', 'broadband', 'no_frontopolar'),
                ('mu-8-13', 'rsec_d3', 'mu', 'native'), ('beta-13-30', 'rsec_d3', 'beta', 'native')]
    sensitivity = adjust([dict(setting=setting, contrast=label, **describe(delta))
                          for setting, metric, band, reference in settings
                          for label, delta in contrasts(wide(metric, band, reference)).items()])
    if node_delta.shape != (106, 64) or not np.isfinite(node_delta.to_numpy()).all():
        raise ValueError('Expected finite 106-participant, 64-channel spatial inputs.')
    channels = adjust([dict(channel=channel, **describe(node_delta[channel])) for channel in node_delta])
    fp = node_delta[['Fp1', 'Fpz', 'Fp2']].mean(axis=1)
    sm = node_delta[['C3', 'Cz', 'C4']].mean(axis=1)
    rois = adjust([dict(region='frontopolar', **describe(fp)), dict(region='sensorimotor', **describe(sm)),
                   dict(region='frontopolar-minus-sensorimotor', **describe(fp - sm))])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    wide().to_csv(args.out_dir / 'primary_subject_summary.csv')
    means.to_csv(args.out_dir / 'feature_subject_summary.csv', index=False)
    sensitivity.to_csv(args.out_dir / 'sensitivity_statistics.csv', index=False)
    node_delta.to_csv(args.out_dir / 'imagery_node_deltas_by_subject.csv')
    channels.sort_values('mean_delta').to_csv(args.out_dir / 'imagery_channel_statistics.csv', index=False)
    rois.to_csv(args.out_dir / 'imagery_roi_statistics.csv', index=False)
    print('Recomputed 21 sensitivity, 64 channel and 3 regional tests.')


if __name__ == '__main__':
    main()
