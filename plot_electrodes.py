import os
import os.path as op
import numpy as np
import json
import pandas as pd

import mne
import mne_bids
import matplotlib.pyplot as plt

sub = 'P3'
task = None  # 'numbers'
run = None  # '1'
aseg = 'wmparc'  # use for automatic freesurfer labels
template = 'MNI152NLin2009cAsym'

bids_root = 'EMU_test_BIDS'

# get plotting information
subjects_dir = op.join(bids_root, "derivatives", "freesurfer")
brain_kwargs = dict(
    cortex="low_contrast",
    alpha=0.2,
    background="white",
    subjects_dir=subjects_dir,
)
lut, colors = mne._freesurfer.read_freesurfer_lut()
lut_r = {v: k for k, v in lut.items()}
cmap = plt.get_cmap("viridis")

path = mne_bids.BIDSPath(subject=sub, root=bids_root, run=run, task=task)
if path.fpath.exists():
    raw = mne_bids.read_raw_bids(path)
    brain_kwargs.update(units='m')
else:
    df = pd.read_csv(path.copy().update(suffix='electrodes',
                                        datatype='ieeg',
                                        extension='.tsv'), sep='\t')
    info = mne.create_info(list(df['name']), 1000., 'seeg')
    raw = mne.io.RawArray(np.zeros((len(df), 1)), info)
    montage = mne.channels.make_dig_montage(
        {name: [x, y, z] for name, x, y, z in zip(df.name, df.x, df.y, df.z)},
        coord_frame="ras"
    )
    raw.set_montage(montage)
    with open(path.copy().update(suffix='coordsystem',
                                 datatype='ieeg',
                                 extension='.json').fpath, 'r') as fid:
        brain_kwargs.update(units=json.load(fid)['iEEGCoordinateUnits'])


# plot on individual
montage = raw.get_montage()
mne_bids.convert_montage_to_mri(montage, f'sub-{sub}', subjects_dir)
brain = mne.viz.Brain(f'sub-{sub}', **brain_kwargs)
ch_pos = montage.get_positions()['ch_pos']
scale = 0.0025
if brain_kwargs['units'] == 'mm':
    scale *= 1000
for name, pos in ch_pos.items():
    brain._renderer.sphere(center=pos, color='yellow', scale=scale)


# plot on template
raw = mne_bids.read_raw_bids(path.copy().update(space=template))
montage = raw.get_montage()
mne_bids.convert_montage_to_mri(montage, f'sub-{sub}', subjects_dir)
brain = mne.viz.Brain(template, **dict(
    brain_kwargs, subjects_dir=op.join(os.environ["FREESURFER_HOME"], 'subjects')))
ch_pos = montage.get_positions()['ch_pos']
for name, pos in ch_pos.items():
    brain._renderer.sphere(center=pos, color='yellow', scale=0.0025)
