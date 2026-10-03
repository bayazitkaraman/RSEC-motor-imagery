"""Shared study extraction and estimators, retaining the audited numerical procedures."""
from pathlib import Path
import sys
import mne
import numpy as np

PUBLIC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PUBLIC))
from rsec_eeg.features import rnt_energy, rnt_shared_energy_events

SUBJECTS = [s for s in range(1, 110) if s not in (88, 92, 100)]
RUNS = {'rest': [1], 'imagery': [4, 8, 12], 'execution': [3, 7, 11]}
BANDS = {'broadband': (1., 31.), 'mu': (8., 13.), 'beta': (13., 30.)}
METHODS = ['coh', 'plv', 'pli', 'wpli']
SEED = 20261001


def source_path(data_dir, subject, run, extra_data_dir=None):
    data_dir = Path(data_dir)
    relative = Path(f'S{subject:03d}/S{subject:03d}R{run:02d}.edf')
    candidates = [data_dir / 'MNE-eegbci-data/files/eegmmidb/1.0.0' / relative,
                  data_dir / relative]
    if extra_data_dir is not None:
        candidates.append(Path(extra_data_dir) / relative)
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(relative)


def coincidence(values, decimals=3):
    """Count equal bins using disjoint per-time keys, including node counts."""
    if not np.isfinite(values).all():
        raise ValueError('Energy values must be finite.')
    quantized = np.rint(values * 10**decimals).astype(np.int64)
    nc, nt = quantized.shape
    span = int(quantized.max() - quantized.min() + 1)
    keys = (quantized.T - quantized.min() + np.arange(nt)[:, None] * span).ravel()
    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    nodes = (counts[inverse].reshape(nt, nc) - 1).mean(axis=0) / (nc - 1)
    return float(nodes.mean()), nodes


def independent_epochs(subject, condition, band, data_dir, target_sfreq=None, extra_data_dir=None):
    segments, provenance=[],[]
    reference_names=None
    reference_sfreq=None
    for run in RUNS[condition]:
        path=source_path(data_dir, subject, run, extra_data_dir)
        raw=mne.io.read_raw_edf(path,preload=True,verbose='ERROR')
        mne.datasets.eegbci.standardize(raw)
        raw.pick_types(eeg=True,verbose='ERROR')
        raw.set_montage(mne.channels.make_standard_montage('standard_1005'),match_case=False,on_missing='raise')
        original_rate=float(raw.info['sfreq'])
        original_duration=raw.n_times/original_rate
        events,event_id=mne.events_from_annotations(raw,verbose='ERROR')
        if target_sfreq is not None and original_rate != target_sfreq:
            raw,events=raw.resample(target_sfreq,events=events,n_jobs=1,verbose='ERROR')
            assert abs(raw.n_times/raw.info['sfreq']-original_duration)<1/target_sfreq
        raw.filter(*band,fir_design='firwin',verbose='ERROR')
        data=raw.get_data()
        rate=float(raw.info['sfreq'])
        count=int(round(4*rate))
        assert np.isfinite(data).all()
        if reference_names is None:
            reference_names=raw.ch_names
            reference_sfreq=rate
        assert reference_names==raw.ch_names and reference_sfreq==rate
        if condition=='rest':
            starts=list(range(0,raw.n_times-count+1,count))
            candidates=[(start,'R01',4.,float(start/rate)) for start in starts]
        else:
            selected={event_id[name]:name for name in ('T1','T2') if name in event_id}
            annotations=[a for a in raw.annotations if a['description'] in ('T1','T2')]
            task_events=[e for e in events if e[2] in selected]
            assert len(annotations)==len(task_events)
            candidates=[]
            for event,annotation in zip(task_events,annotations):
                start=int(event[0])-raw.first_samp
                assert abs(start/rate-annotation['onset'])<=1/original_rate+1/rate
                duration=float(annotation['duration'])
                safe=start/rate+4<=annotation['onset']+duration+.5/rate
                if not safe:
                    continue
                candidates.append((start,selected[event[2]],duration,float(annotation['onset'])))
        for start,label,duration,onset in candidates:
            if start<0 or start+count>raw.n_times:
                continue
            segments.append(data[:,start:start+count])
            provenance.append({'run':run,'start_sample':start,'label':label,'annotation_duration':duration,
                               'onset_seconds':onset,'native_sfreq':original_rate,'processed_sfreq':rate})
    if not segments:
        return np.empty((0,64,int(4*reference_sfreq))),reference_names,reference_sfreq,provenance
    return np.stack(segments),reference_names,reference_sfreq,provenance


def connectivity(epochs, sfreq, band):
    from mne_connectivity import spectral_connectivity_epochs
    pairs = np.triu_indices(epochs.shape[1], 1)
    results = spectral_connectivity_epochs(
        epochs, method=METHODS, indices=pairs, sfreq=sfreq, mode='fourier',
        fmin=band[0], fmax=band[1], faverage=True, block_size=256,
        n_jobs=1, verbose='ERROR')
    values = np.array([con.get_data().mean() for con in results])
    assert np.isfinite(values).all() and (values >= 0).all() and (values <= 1+1e-12).all()
    return dict(zip(METHODS, values))


def self_test():
    rng = np.random.default_rng(SEED)
    t = np.arange(640)/160.
    phases = rng.uniform(0, 2*np.pi, 15)
    locked = np.array([[np.sin(2*np.pi*10*t+p), np.sin(2*np.pi*10*t+p+.7)] for p in phases])
    got = connectivity(locked, 160., (10., 10.))
    np.testing.assert_allclose(list(got.values()), 1., atol=1e-6)
    scaled = connectivity(locked * np.array([1e-6, 1e-4])[None, :, None], 160., (10., 10.))
    np.testing.assert_allclose(list(scaled.values()), list(got.values()), atol=1e-12)
    epochs = rng.normal(size=(60, 3, 640))
    got = connectivity(epochs, 160., (8., 13.))
    ft = np.fft.rfft((epochs-epochs.mean(axis=-1, keepdims=True))*np.hanning(640), axis=-1)
    freqs = np.fft.rfftfreq(640, 1/160.)
    ft = ft[:, :, (freqs >= 8) & (freqs <= 13)]
    i,j = np.triu_indices(3, 1)
    cross = ft[:, i]*ft[:, j].conj()
    imag = cross.imag
    direct = {
        'coh': np.mean(abs(cross.mean(axis=0))/np.sqrt((abs(ft[:,i])**2).mean(axis=0)*(abs(ft[:,j])**2).mean(axis=0))),
        'plv': np.mean(abs((cross/abs(cross)).mean(axis=0))),
        'pli': np.mean(abs(np.sign(imag).mean(axis=0))),
        'wpli': np.mean(abs(imag.mean(axis=0))/abs(imag).mean(axis=0))}
    np.testing.assert_allclose(list(got.values()), list(direct.values()), atol=1e-12)
    assert max(got.values()) < .25
    print('PASS: MNE results match independent Fourier formulas; locked, independent and scaling tests.', flush=True)
