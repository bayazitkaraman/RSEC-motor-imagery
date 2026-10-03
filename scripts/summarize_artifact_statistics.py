"""Reconstruct amplitude and Laplacian sensitivity, preserving all correction families."""
from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rsec_eeg.stats import holm_adjust
SUBJECTS = [s for s in range(1, 110) if s not in (88, 92, 100)]
SEED = 20261002
METRICS = ['rsec', 'linear']
CONDITIONS = ['rest', 'imagery', 'execution']
KEYS = ['subject', 'condition', 'epoch']


def effects(wide):
    return {'imagery-rest':wide.imagery-wide.rest,
            'execution-rest':wide.execution-wide.rest,
            'imagery-execution':wide.imagery-wide.execution}


def basic(values):
    x = np.asarray(values, dtype=float)
    assert np.isfinite(x).all()
    n = len(x)
    return {'n':n, 'mean_delta':float(x.mean()) if n else np.nan,
            'median_delta':float(np.median(x)) if n else np.nan,
            'dz':float(x.mean()/x.std(ddof=1)) if n>1 and x.std(ddof=1)>0 else np.nan,
            'n_decrease':int((x<0).sum()), 'n_increase':int((x>0).sum()),
            'n_equal':int((x==0).sum()),
            'p':float(stats.wilcoxon(x, zero_method='wilcox', correction=False, method='approx').pvalue)
            if n>=10 and np.any(x!=0) else (1. if n>=10 else np.nan)}


def describe(values):
    row = basic(values)
    x = np.asarray(values, dtype=float)
    if len(x)>1:
        rng = np.random.default_rng(SEED)
        boot = x[rng.integers(0, len(x), (10000, len(x)))].mean(axis=1)
        row['ci_low'], row['ci_high'] = map(float, np.quantile(boot, [.025, .975]))
    else:
        row['ci_low'] = row['ci_high'] = np.nan
    row['inference_status'] = 'estimable' if len(x)>=10 else 'insufficient_cohort'
    return row


def correction(frame):
    frame = frame.copy()
    frame['p_holm_family'] = holm_adjust(frame.p.fillna(1.).to_numpy())
    frame.loc[frame.p.isna(), 'p_holm_family'] = np.nan
    return frame


def wide(frame, metric, aggregation='mean'):
    result = frame.groupby(['subject','condition'])[metric].agg(aggregation).unstack()
    return result.reindex(columns=CONDITIONS)


def correlation(x, y):
    if len(x)<3 or np.std(x)==0 or np.std(y)==0:
        return {'spearman_r':np.nan, 'pearson_r':np.nan}
    return {'spearman_r':float(stats.spearmanr(x,y).statistic),
            'pearson_r':float(stats.pearsonr(x,y).statistic)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'results/summary/artifact_epoch_features.csv')
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'outputs_artifact_statistics')
    args = parser.parse_args()
    OUT = args.out_dir
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.input).sort_values(KEYS).reset_index(drop=True)
    if data.duplicated(KEYS).any() or len(data) != 11134 or data.isna().any().any():
        raise ValueError('Expected the complete finite 11134-epoch input with unique participant/condition/epoch keys.')
    original = {m:wide(data,m).loc[SUBJECTS] for m in METRICS}
    reference = {m:effects(original[m]) for m in METRICS}

    masks = {}
    for region,column in [('all','max_ptp_uv'),('fp','frontopolar_max_ptp_uv')]:
        for threshold in (150,250,500):
            masks[f'{region}-ptp-{threshold}'] = (data[column]<=threshold, 'mean')
        ranks = data.groupby(['subject','condition'])[column].rank(method='first')
        limits = np.floor(.8*data.groupby(['subject','condition'])[column].transform('size'))
        masks[f'{region}-trim-top20'] = (ranks<=limits, 'mean')
    masks['median-epochs'] = (pd.Series(True, index=data.index), 'median')
    assert len(masks)==9
    artifact_rows, retention_rows, cohort_rows, correlation_rows = [], [], [], []
    subject_profiles = []
    all_counts = data.groupby(['subject','condition']).size().unstack().reindex(index=SUBJECTS,columns=CONDITIONS)
    for setting,(mask, aggregation) in masks.items():
        kept = data.loc[mask]
        counts = kept.groupby(['subject','condition']).size().unstack().reindex(index=SUBJECTS,columns=CONDITIONS).fillna(0)
        eligible = counts.index[(counts>=5).all(axis=1)].tolist()
        cohort_rows.append({'setting':setting, 'n_eligible':len(eligible),
                            'excluded_subjects':','.join(str(s) for s in SUBJECTS if s not in eligible)})
        for subject in SUBJECTS:
            for condition in CONDITIONS:
                retention_rows.append({'setting':setting, 'subject':subject, 'condition':condition,
                                       'original_epochs':int(all_counts.loc[subject,condition]),
                                       'retained_epochs':int(counts.loc[subject,condition]),
                                       'eligible':subject in eligible})
        setting_effects = {}
        for metric in METRICS:
            means = wide(kept, metric, aggregation).reindex(eligible)
            assert not means.isna().any().any()
            setting_effects[metric] = effects(means)
            for subject,row in means.iterrows():
                subject_profiles.append({'setting':setting,'metric':metric,'subject':subject,**row.to_dict()})
            for contrast,delta in setting_effects[metric].items():
                matched = reference[metric][contrast].loc[eligible]
                old_result = basic(matched)
                artifact_rows.append({'family':'amplitude_aggregation','setting':setting,'metric':metric,
                                      'contrast':contrast, **describe(delta),
                                      'matched_original_mean_delta':old_result['mean_delta'],
                                      'matched_original_dz':old_result['dz'],
                                      'mean_change_from_epoch_processing':float((delta-matched).mean()) if len(delta) else np.nan})
        for contrast in reference['rsec']:
            correlation_rows.append({'setting':setting,'contrast':contrast,'n':len(eligible),
                                     **correlation(setting_effects['rsec'][contrast],setting_effects['linear'][contrast])})
    artifacts = correction(pd.DataFrame(artifact_rows))
    assert len(artifacts)==54
    pd.DataFrame(retention_rows).to_csv(OUT/'retention_by_subject.csv', index=False)
    pd.DataFrame(cohort_rows).to_csv(OUT/'cohort_selection.csv', index=False)
    pd.DataFrame(subject_profiles).to_csv(OUT/'masked_subject_means.csv', index=False)

    csd_rows, mechanism_rows, decomposition_rows, condition_rows = [], [], [], []
    for metric in METRICS:
        transformed = wide(data,metric+'_csd').loc[SUBJECTS]
        for contrast,delta in effects(transformed).items():
            csd_rows.append({'family':'csd','setting':'csd','metric':metric,'contrast':contrast,**describe(delta)})
        for kind in ('expected','excess'):
            summary = wide(data,metric+'_'+kind).loc[SUBJECTS]
            for contrast,delta in effects(summary).items():
                mechanism_rows.append({'family':'marginal_control','setting':kind,'metric':metric,
                                       'contrast':contrast,**describe(delta)})
        expected = wide(data,metric+'_expected').loc[SUBJECTS]
        excess = wide(data,metric+'_excess').loc[SUBJECTS]
        np.testing.assert_allclose(original[metric],expected+excess,rtol=0,atol=1e-14)
        for condition in CONDITIONS:
            condition_rows.append({'metric':metric,'condition':condition,
                                   'observed_mean':float(original[metric][condition].mean()),
                                   'expected_mean':float(expected[condition].mean()),
                                   'excess_mean':float(excess[condition].mean())})
        for contrast,delta in reference[metric].items():
            e_delta = effects(expected)[contrast]
            x_delta = effects(excess)[contrast]
            decomposition_rows.append({'metric':metric,'contrast':contrast,'observed_mean_delta':float(delta.mean()),
                                       'expected_mean_delta':float(e_delta.mean()),'excess_mean_delta':float(x_delta.mean()),
                                       'expected_to_observed_mean_ratio':float(e_delta.mean()/delta.mean()),
                                       **correlation(delta,e_delta)})
    csd = correction(pd.DataFrame(csd_rows))
    mechanism = correction(pd.DataFrame(mechanism_rows))
    assert len(csd)==6 and len(mechanism)==12
    combined = pd.concat([artifacts,csd,mechanism],ignore_index=True)
    assert len(combined)==72
    combined['p_holm_all72'] = holm_adjust(combined.p.fillna(1.).to_numpy())
    combined.loc[combined.p.isna(),'p_holm_all72'] = np.nan
    combined.to_csv(OUT/'all_followup_tests.csv', index=False)
    for family,name in [('amplitude_aggregation','amplitude_aggregation_statistics.csv'),
                        ('csd','csd_statistics.csv'),('marginal_control','marginal_control_statistics.csv')]:
        combined[combined.family==family].to_csv(OUT/name,index=False)
    pd.DataFrame(decomposition_rows).to_csv(OUT/'marginal_decomposition.csv',index=False)
    pd.DataFrame(condition_rows).to_csv(OUT/'marginal_condition_means.csv',index=False)

    print('Recomputed nine amplitude/aggregation settings and surface-Laplacian contrasts.')


if __name__ == '__main__':
    main()
