import os
import os.path as op

import mne
import mne_bids
import matplotlib.pyplot as plt

sub = '72'
task = 'numbers'
aseg = 'wmparc'  # use for automatic freesurfer labels
template = 'MNI152NLin2009cAsym'

bids_root = '../numbers_BIDS'

# get plotting information
subjects_dir = op.join(bids_root, "derivatives", "freesurfer")
brain_kwargs = dict(
    cortex="low_contrast",
    alpha=0.2,
    background="white",
    subjects_dir=subjects_dir,
    units="m",  # could get from sidecar
)
lut, colors = mne._freesurfer.read_freesurfer_lut()
lut_r = {v: k for k, v in lut.items()}
cmap = plt.get_cmap("viridis")

path = mne_bids.BIDSPath(subject=sub, root=bids_root, run='1', task=task)
raw = mne_bids.read_raw_bids(path)

# plot on individual
brain = mne.viz.Brain(f'sub-{sub}', **brain_kwargs)
montage = raw.get_montage()
mne_bids.convert_montage_to_mri(montage, f'sub-{sub}', subjects_dir)
ch_pos = montage.get_positions()['ch_pos']
for name, pos in ch_pos.items():
    brain._renderer.sphere(center=pos, color='yellow', scale=0.0025)


# plot on template
raw = mne_bids.read_raw_bids(path.copy().update(space=template))
montage = raw.get_montage()
mne_bids.convert_montage_to_mri(montage, f'sub-{sub}', subjects_dir)
brain = mne.viz.Brain(template, **dict(
    brain_kwargs, subjects_dir=op.join(os.environ["FREESURFER_HOME"], 'subjects')))
ch_pos = montage.get_positions()['ch_pos']
for name, pos in ch_pos.items():
    brain._renderer.sphere(center=pos, color='yellow', scale=0.0025)
