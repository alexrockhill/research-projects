# to run, first run in a terminal (one time only):
# pip install mne mne_bids nibabel dipy pd_parser
# then run in a terminal:
# python /path/to/this/file/ieeg_preprocessing.py

import sys
import os
import os.path as op
import json
import numpy as np
from collections import OrderedDict
import time
from shutil import which, copyfile
from subprocess import run, Popen, PIPE
from multiprocessing import Pool
from dipy.align import affine_registration
import mne
import mne_gui_addons as mne_gui
import mne_bids
import nibabel as nib
import pd_parser
from pd_parser.parse_pd import _read_tsv, _to_tsv, _load_data
from pandas import read_csv, DataFrame
import matplotlib.pyplot as plt
from templateflow import api as tflow

TEMPLATE = "MNI152NLin2009cAsym"
INSET = 30  # this inset and theta for defacing was chosen to work well
THETA = 30  # for this specific template and it's estimated fiducials

ACPC_COORDSYS = {
    "iEEGCoordinateSystem": "ACPC",
    "iEEGCoordinateSystemDescription":
        "The origin of the coordinate system is at the Anterior Commissure and "
        "the negative y-axis is passing through the Posterior Commissure. "
        "The positive z-axis is passing through a mid-hemispheric point in "
        "the superior direction.",
    "iEEGCoordinateUnits": "m"
}

TEMPLATE_COORDSYS = {
    "iEEGCoordinateSystem": TEMPLATE,
    "iEEGCoordinateUnits": "m",
}


def print_status(sub, root, work_dir):
    for events_fname in [f for f in os.listdir(op.join(root, 'ieeg')) if
                         f.endswith('events.tsv')]:
        name_dict = dict([kv.split('-') for kv in events_fname.split('_')[:-1]])
        print('Task data file complete: {}'.format(name_dict['task']))
    for name, fname in zip(
        [
            "Import MR",
            "Import CT",
            "Find Contacts",
            "Warp to Template",
        ],
        [
            op.join(work_dir, 'anat' "T1.mgz"),
            op.join(work_dir, 'anat', "CT.mgz"),
            op.join(work_dir, "ieeg", "ch_pos.fif"),
            op.join(work_dir, "ieeg", f"{TEMPLATE}_ch_pos.fif"),
        ],
    ):
        status = "done" if op.isfile(fname) else "incomplete"
        print(f"{name}: {status}")


def do_step(prompt, step, *args):
    answer = input(f"{prompt}? (Y/n)\t").lower()
    while answer not in ("y", "n"):
        answer = input(f"{prompt}? (Y/n)\t").lower()
    if answer == "y":
        step(*args)


def get_file(prompt, required=True):
    path = None
    while path is None:
        path = input(f'{prompt}\t').strip()
        if not op.isfile(path):
            if required:
                print('File not found')
                path = None
            elif path:
                keep_going = None
                while keep_going is None:
                    keep_going = input('File not found, try again?\t (Y/n)')
                    if not (keep_going.lower().startswith('y') or
                            keep_going.lower().startswith('n')):
                        keep_going = None
                path = None if keep_going.lower().startswith('y') else ''
    return path


# modified from mne-bids
def deface(image, landmarks, inset=5, theta=15.0):
    # get image data, make a copy
    image_data = image.get_fdata().copy()

    # make indices to move around so that the image doesn't have to
    idxs = np.meshgrid(
        np.arange(image_data.shape[0]),
        np.arange(image_data.shape[1]),
        np.arange(image_data.shape[2]),
        indexing="ij",
    )
    idxs = np.array(idxs)  # (3, *image_data.shape)
    idxs = np.transpose(idxs, [1, 2, 3, 0])  # (*image_data.shape, 3)
    idxs = idxs.reshape(-1, 3)  # (n_voxels, 3)

    # convert to RAS by applying affine
    idxs = nib.affines.apply_affine(image.affine, idxs)

    # now comes the actual defacing
    # 1. move center of voxels to (nasion - inset)
    # 2. rotate the head by theta from vertical
    x, y, z = landmarks[1].copy()
    idxs = mne.transforms.apply_trans(
        mne.transforms.translation(x=-x, y=-y + inset, z=-z), idxs)
    idxs = mne.transforms.apply_trans(
        mne.transforms.rotation(x=-np.pi / 2 + np.deg2rad(theta)), idxs)
    idxs = idxs.reshape(image_data.shape + (3,))
    mask = idxs[..., 2] < 0  # z < middle
    image_data[mask] = 0

    image = nib.Nifti1Image(image_data, image.affine, image.header)
    return image


def import_dicom(sub, root, work_dir, subjects_dir, fs_subjects_dir):
    tmp_dir = op.join(work_dir, "tmp")
    fpath = input('DICOM folder (searches all subfolders)?\t').strip()
    run(f"dcm2niix -o {tmp_dir} -z y {fpath}", shell=True)


def print_imported_nii():
    tmp_dir = op.join(work_dir, "tmp")
    nii_files = [op.join(tmp_dir, f) for f in os.listdir(tmp_dir)
                 if f.endswith('nii.gz')]
    print('\n' * 2)
    for i, fname in enumerate(nii_files):
        proc = Popen(['mri_info', fname], stdout=PIPE, stderr=PIPE)
        mri_info, err = proc.communicate()
        mri_info = mri_info.decode("utf-8").split('  ')
        vox_row = [row.strip() for row in mri_info if
                   row.strip().startswith('voxel sizes')][0]
        print(f"({i + 1}) {op.basename(fname)} "
              f"size={sys.getsizeof(fname) / 1e6} MB {vox_row} mm")
    print('\n' * 2)
    return nii_files


def choose_file(ftype, nii_files):
    if ftype == 'T2':
        print('T2 not required, press enter to skip')
    idx = None
    while idx is None or idx < 0 or idx > len(nii_files) - 1:
        out = input(f'{ftype} file (1 - {len(nii_files)} or file path)?\t')
        if op.isfile(out):
            if op.isfile(out.replace('nii.gz', 'json')):
                return out, out.replace('nii.gz', 'json')
            return out, input(f'{ftype} json file path? (enter to skip)\t')
        try:
            idx = int(out) - 1
        except Exception:
            if ftype == 'T2' and out == '':
                return None, None
            else:
                idx = None
        if idx is None or idx < 0 or idx > len(nii_files) - 1:
            print(f'Choose (1 - {len(nii_files)})\n')
    return nii_files[idx], nii_files[idx].replace('nii.gz', 'json')


def import_mr(sub, root, work_dir, subjects_dir, fs_subjects_dir):
    tmp_dir = op.join(work_dir, "tmp")
    nii_files = print_imported_nii()
    t1_fname, t1_json_fname = choose_file('T1', nii_files)
    t2_fname, t2_json_fname = choose_file('T2', nii_files)

    anat_dir = op.join(root, f'sub-{sub}', 'anat')
    _ensure_recon(TEMPLATE, 'brain', fs_subjects_dir)

    t1 = nib.load(t1_fname)
    brain_template = nib.load(op.join(fs_subjects_dir, TEMPLATE, 'mri', 'brain.mgz'))

    print('Aligning to ACPC and defacing')
    # this works the best rather than trying to align the reverse or with the full T1
    reg_affine = mne.transforms.compute_volume_registration(
        brain_template, t1, pipeline='rigids')[0]
    template_t1_t_fname = op.join(work_dir, 'anat', f'{TEMPLATE}_T1-trans.fif')
    if op.isfile(template_t1_t_fname):
        os.remove(template_t1_t_fname)
    mne.transforms.Transform(fro='ras', to='ras', trans=reg_affine).save(
        template_t1_t_fname)

    landmarks = mne._freesurfer.get_mni_fiducials(TEMPLATE, fs_subjects_dir)
    landmarks = np.asarray([landmark['r'] for landmark in landmarks])  # in surface RAS
    landmarks = mne.transforms.apply_trans(  # surface RAS to voxels
        np.linalg.inv(brain_template.header.get_vox2ras_tkr()), landmarks * 1000)
    landmarks = mne.transforms.apply_trans(brain_template.affine, landmarks)  # RAS
    # transform to individual T1 space
    landmarks = mne.transforms.apply_trans(np.linalg.inv(reg_affine), landmarks)

    t1_defaced = deface(t1, landmarks, inset=INSET, theta=THETA)

    # save to BIDS
    t1_acpc_fname = op.join(work_dir, 'anat', 'T1.mgz')
    nib.save(t1_defaced, op.join(anat_dir, f'sub-{sub}_T1w.nii.gz'))
    if t1_json_fname:
        copyfile(t1_json_fname, op.join(anat_dir, f'sub-{sub}_T1w.json'))
    t1_acpc = mne.transforms.apply_volume_registration(
        t1_defaced, brain_template, np.linalg.inv(reg_affine))
    nib.save(t1_acpc, t1_acpc_fname)

    t2_acpc_fname = op.join(work_dir, 'anat', 'T2.mgz')
    if t2_fname:
        t2 = nib.load(t2_fname)
        t2_defaced = deface(t2, landmarks, reg_affine)
        nib.save(t2_defaced, op.join(anat_dir, f'sub-{sub}_T2w.nii.gz'))
        if t2_json_fname:
            copyfile(t2_json_fname, op.join(anat_dir, f'sub-{sub}_T2w.json'))
        t2_acpc = mne.transforms.apply_volume_registration(
            t2_defaced, brain_template, np.linalg.inv(reg_affine))
        nib.save(t2_acpc, t2_acpc_fname)

    renderer = mne.viz.create_3d_figure((600, 600), scene=False, show=True)
    verts, tris = mne.surface._marching_cubes(
        np.array(t1_defaced.dataobj) > 50, [1])[0]
    renderer.mesh(*verts.T, tris, 'gray')
    renderer.screenshot(filename=op.join(work_dir, 'figures', 'T1_deface.png'))

    print(
        "Running Freesurfer recon-all, do not put the computer to sleep until "
        "it finishes. This may take as long as 12 hours. You can "
        "move on to the next step in the meantime. The progress will be "
        f"output to {tmp_dir}/sub-{sub}_recon_output.txt"
    )
    # run recon-all
    os.environ["SUBJECTS"] = f"sub-{sub}"
    os.environ["SUBJECTS_DIR"] = subjects_dir
    recon_cmd = which("recon-all")
    cmd = f"{recon_cmd} -subjid sub-{sub} -i {t1_acpc_fname} "
    if op.isfile(t2_acpc_fname):
        cmd += f"-T2 {t2_acpc_fname} "
    cmd += "-all -deface -cw256"
    with open(op.join(tmp_dir, f"sub-{sub}_recon_output.txt"), "w") as fid:
        Popen(cmd.split(" "), stdout=fid, stderr=fid, env=os.environ)

    # make head surfaces
    pool = Pool(processes=1)
    pool.apply_async(make_head_surface, [sub, subjects_dir, t2_acpc_fname])


def import_ct(sub, root, work_dir, subjects_dir, fs_subjects_dir):
    nii_files = print_imported_nii()
    ct_fname, ct_json_fname = choose_file('CT', nii_files)

    if not op.isfile(op.join(work_dir, 'anat', 'T1.mgz')):
        raise RuntimeError('Import MR must be done first')

    t1 = nib.load(op.join(work_dir, 'anat', 'T1.mgz'))

    # rough CT alignment for defacing
    print('Pre-registering CT to T1 and defacing CT')
    ct = nib.load(ct_fname)
    ct_reg_affine = mne.transforms.compute_volume_registration(
        ct, t1, pipeline='rigids')[0]
    ct_trans = mne.transforms.Transform(fro='ras', to='ras', trans=ct_reg_affine)
    ct_trans_fname = op.join(work_dir, 'anat', 'CT_MR-trans.fif')
    if op.isfile(ct_trans_fname):
        os.remove(ct_trans_fname)
    ct_trans.save(ct_trans_fname)

    anat_dir = op.join(root, f'sub-{sub}', 'anat')

    landmarks = mne._freesurfer.get_mni_fiducials(TEMPLATE, fs_subjects_dir)
    landmarks = np.asarray([landmark['r'] for landmark in landmarks])  # in surface RAS
    landmarks = mne.transforms.apply_trans(  # surface RAS to voxels
        np.linalg.inv(t1.header.get_vox2ras_tkr()), landmarks * 1000)
    landmarks = mne.transforms.apply_trans(t1.affine, landmarks)  # RAS
    # transform to individual CT space
    landmarks = mne.transforms.apply_trans(ct_reg_affine, landmarks)
    ct_defaced = deface(ct, landmarks, inset=INSET, theta=THETA)
    nib.save(ct_defaced, op.join(anat_dir, f'sub-{sub}_ct.nii.gz'))

    renderer = mne.viz.create_3d_figure((600, 600), scene=False, show=True)
    verts, tris = mne.surface._marching_cubes(
        np.logical_and(50 < np.array(ct_defaced.dataobj),
                       np.array(ct_defaced.dataobj) < 500), [1])[0]
    verts = mne.transforms.apply_trans(ct_defaced.affine, verts)
    renderer.mesh(*verts.T, tris, 'gray')
    renderer.screenshot(filename=op.join(work_dir, 'figures', 'CT_deface.png'))

    ct_acpc = mne.transforms.apply_volume_registration(
        ct_defaced, t1, ct_reg_affine)

    nib.save(ct_acpc, op.join(work_dir, 'anat', 'CT.mgz'))
    if ct_json_fname:
        copyfile(ct_json_fname, op.join(anat_dir, f'sub-{sub}_ct.json'))


def make_head_surface(sub, subjects_dir, t2_acpc_fname):
    mne.bem.make_scalp_surfaces(
        f"sub-{sub}", subjects_dir, force=True, overwrite=True
    )
    if t2_acpc_fname:
        mne.bem.make_flash_bem(
            f"sub-{sub}",
            subjects_dir=subjects_dir,
            flash5_img=t2_acpc_fname,
            register=False,
            overwrite=True,
        )
    else:
        mne.bem.make_watershed_bem(
            f"sub-{sub}", subjects_dir=subjects_dir, overwrite=True
        )


def _ensure_recon(sub, ftype, subjects_dir):
    assert ftype in ("trans", "T1", "brain")
    if ftype == "trans":
        fname = op.join(
            subjects_dir, f"sub-{sub}", "mri", "transforms", "talairach.xfm"
        )
    else:
        fname = op.join(subjects_dir, f"sub-{sub}", "mri", f"{ftype}.mgz")
    if not op.isfile(fname):
        print(
            "The recon has not gotten far enough yet, pausing here until it "
            f"does ({fname} does not exist yet)"
        )
    while not op.isfile(fname):
        time.sleep(1)
        print(".", end="", flush=True)


def align_CT(sub, work_dir, subjects_dir):
    ct_fname = op.join(work_dir, 'anat', 'CT.mgz')
    if not op.isfile(ct_fname):
        raise RuntimeError('Import CT must be done first')
    ct = nib.load(ct_fname)
    t1_fname = op.join(work_dir, 'anat', 'T1.mgz')
    t1 = nib.load(t1_fname)
    ct_orig = nib.load(op.join(root, f'sub-{sub}', 'anat', f'sub-{sub}_ct.nii.gz'))

    # manual pre-align
    print(
        "Check alignment, think you can do better? perform a manual alignment "
        "with Tools>>Transform Volume.. Save Volume As.. anywhere when finished"
    )
    run(
        "freeview {} {}:colormap=heat:opacity=0.6".format(
            t1_fname, ct_fname
        ).split(" ")
    )
    ct_manual_fname = None
    while ct_manual_fname is None:
        ct_manual_fname = input("CT manual alignment path? [enter to skip]\t").strip()
        if ct_manual_fname == '':
            break
        elif not op.isfile(ct_manual_fname):
            print('File not found')
            ct_manual_fname = None
    # save as backup after manual input
    if ct_manual_fname == '':
        return

    if ct_manual_fname != ct_fname.replace(".mgz", "_manual.mgz"):
        copyfile(ct_manual_fname, ct_fname.replace(".mgz", "_manual.mgz"))
        copyfile(ct_manual_fname + ".lta", ct_fname.replace(".mgz", "_manual.mgz.lta"))
    manual_reg_affine_vox = mne.read_lta(ct_fname.replace(".mgz", "_manual.mgz.lta"))
    # convert from vox->vox to ras->ras
    manual_reg_affine = (
        ct.affine
        @ np.linalg.inv(manual_reg_affine_vox)
        @ np.linalg.inv(ct.affine)
    )
    orig_reg_affine = mne.read_trans(op.join(work_dir, 'anat', 'ct_mr-trans.fif'))
    ct_aligned_fix_img, reg_affine_fix = affine_registration(
        moving=np.array(ct_orig.dataobj),
        static=np.array(t1.dataobj),
        moving_affine=ct_orig.affine,
        static_affine=t1.affine,
        pipeline=["rigid"],
        starting_affine=manual_reg_affine.dot(orig_reg_affine),
        level_iters=[100],
        sigmas=[0],
        factors=[1],
    )
    nib.save(nib.MGHImage(ct_aligned_fix_img.astype(np.float32), t1.affine), ct_fname)
    print("Check the final alignment, if it fails, try restarting or ask for help")
    run(
        "freeview {} {}:colormap=heat:opacity=0.6".format(t1_fname, ct_fname).split(" ")
    )


# modified from mne-bids
def electrodes_tsv(info):
    x, y, z, names = list(), list(), list(), list()
    for ch in info["chs"]:
        if ch["kind"] in (mne.constants.FIFF.FIFFV_STIM_CH,
                          mne.constants.FIFF.FIFFV_MISC_CH):
            print(f"Not writing stim chan {ch['ch_name']} to electrodes.tsv")
            continue
        elif np.isnan(ch["loc"][:3]).any() or np.allclose(ch["loc"][:3], 0):
            x.append("n/a")
            y.append("n/a")
            z.append("n/a")
        else:
            x.append(ch["loc"][0])
            y.append(ch["loc"][1])
            z.append(ch["loc"][2])
        names.append(ch["ch_name"])

    data = OrderedDict([("name", names), ("x", x), ("y", y), ("z", z),
                        ("size", ["n/a"] * len(names))])
    return DataFrame(data)


def find_contacts(sub, root, work_dir, subjects_dir):
    if not op.isfile(op.join(work_dir, "anat", "CT.mgz")):
        raise RuntimeError('CT Align must be done first')
    mne.viz.set_3d_backend("pyvistaqt")
    ct = nib.load(op.join(work_dir, "anat", "CT.mgz"))
    raw_fnames = [f for f in os.listdir(op.join(root, f'sub-{sub}', 'ieeg'))
                  if f.endswith('edf') or f.endswith('vmrk')]
    if raw_fnames:
        raw_fname = raw_fnames[0]
    else:
        raw_fname = input("Intracranial recording file path?\t").strip()
    info_fname = op.join(work_dir, "ieeg", "ch_pos.fif")
    _ensure_recon(sub, "trans")
    trans = mne.coreg.estimate_head_mri_t(f"sub-{sub}", subjects_dir)
    raw = mne.io.read_raw(raw_fname)
    raw = normalize_channel_names(raw)
    if op.isfile(info_fname):
        info = mne.io.read_info(info_fname)
        raw.set_montage(
            mne.channels.make_dig_montage(
                {ch["ch_name"]: ch["loc"][:3] for ch in info["chs"]}, coord_frame="head"
            )
        )
    mne_gui.locate_ieeg(raw.info, trans, ct, subject=f"sub-{sub}",
                        subjects_dir=subjects_dir)
    while input('Press "s" to save when finished\t').lower() != "s":
        pass
    mne.io.write_info(info_fname, raw.info)
    coordsys_fname = op.join(root, f'sub-{sub}', 'ieeg',
                             f'sub-{sub}_space-ACPC_coordsystem.tsv')
    df = electrodes_tsv(raw.info)
    df.to_csv(coordsys_fname, sep='\t', index=False)
    if not op.isfile(op.splitext(coordsys_fname)[0] + '.json'):
        t1_fname = op.relpath(op.join(subjects_dir, f'sub-{sub}', 'mri', 'T1.mgz'),
                              root)
        with open(op.splitext(coordsys_fname)[0] + '.json', 'w') as fid:
            fid.write(json.dumps(dict(ACPC_COORDSYS, IntendedFor=t1_fname), indent=4))


def warp_to_template(sub, root, work_dir, subjects_dir, fs_subjects_dir):
    info_fname = op.join(work_dir, "ieeg", "ch_pos.fif")
    if not op.isfile(info_fname):
        raise RuntimeError('Find Contacts must be done first')

    raw_fnames = [f for f in os.listdir(op.join(root, f'sub-{sub}', 'ieeg'))
                  if f.endswith('edf') or f.endswith('vmrk')]
    if raw_fnames:
        raw_fname = raw_fnames[0]
    else:
        raw_fname = input("Intracranial recording file path?\t").strip()
    raw = mne.io.read_raw(raw_fname)

    info = mne.io.read_info(info_fname)
    template_info_fname = op.join(work_dir, "ieeg", f"{TEMPLATE}_ch_pos.fif")
    if op.isfile(template_info_fname):
        print("Warning already exists, overwriting")
    print(f"Warping to {TEMPLATE} template brain, this will take about 15 minutes")
    # template data
    template_brain = nib.load(op.join(fs_subjects_dir, TEMPLATE, "mri", "brain.mgz"))
    template_trans = mne.coreg.estimate_head_mri_t(TEMPLATE, fs_subjects_dir)
    # subject data
    trans = mne.coreg.estimate_head_mri_t(f"sub-{sub}", subjects_dir)
    subject_brain = nib.load(op.join(subjects_dir, f"sub-{sub}", "mri", "brain.mgz"))
    reg_affine, sdr_morph = mne.transforms.compute_volume_registration(
        subject_brain, template_brain, verbose=True
    )
    ch_pos = {ch["ch_name"]: ch["loc"][:3] for ch in info["chs"]}
    montage = mne.channels.make_dig_montage(ch_pos, coord_frame="head")
    # montage = raw.get_montage()
    montage.apply_trans(trans)
    CT_aligned = nib.load(op.join(subjects_dir, f"sub-{sub}", "CT", "CT.mgz"))
    montage_warped = mne.preprocessing.ieeg.warp_montage(
        montage, subject_brain, template_brain, reg_affine, sdr_morph
    )
    elec_image = mne.preprocessing.ieeg.make_montage_volume(montage, CT_aligned)
    # now go back to "head" coordinates to save to raw
    montage_warped.apply_trans(mne.transforms.invert_transform(template_trans))
    raw.set_montage(montage_warped, on_missing='warn')
    mne.io.write_info(template_info_fname, raw.info)
    nib.save(elec_image, op.join(work_dir, "ieeg", "elec_image.mgz"))
    coordsys_fname = op.join(root, f'sub-{sub}', 'ieeg',
                             f'sub-{sub}_space-{TEMPLATE}_coordsystem.tsv')
    df = electrodes_tsv(raw.info)
    df.to_csv(coordsys_fname, sep='\t', index=False)
    if not op.isfile(op.splitext(coordsys_fname)[0] + '.json'):
        with open(op.splitext(coordsys_fname)[0] + '.json', 'w') as fid:
            fid.write(json.dumps(TEMPLATE_COORDSYS, indent=4))


def _get_pd_ch_names(raw):
    pd_chs = list()
    while len(pd_chs) < 2:
        ch = None
        while ch not in raw.ch_names:
            ch = input(f"PD channel {len(pd_chs) + 1}?\t")
            if ch not in raw.ch_names:
                print(f"{ch} not in {raw.ch_names}")
        pd_chs.append(ch)
    return pd_chs


def _fix_pd(raw):
    pd_chs = _get_pd_ch_names(raw)
    pd = (
        raw._data[raw.ch_names.index(pd_chs[0])]
        - raw._data[raw.ch_names.index(pd_chs[1])]
    )
    pd_diff = abs(np.diff(pd))
    pd_diff_filter = np.concatenate(
        [
            np.array(
                [
                    max([pd_diff[j + i] for i in range(-10, 11)])
                    for j in range(10, pd_diff.size - 10)
                ]
            ),
            pd_diff[-21:],
        ]
    )
    fig, ax = plt.subplots()
    ax.plot(pd_diff_filter)
    fig.show()
    thresh = float(input(
        "Zoom in on plot to find where deflections are "
        "higher than the baseline. Threshold?\t"
    ))
    # assign pd deflections based on where there is no noise
    pd_fix = np.where(pd_diff_filter < float(thresh), 1, 0)
    raw._data[raw.ch_names.index(pd_chs[0])] = pd_fix
    rng = np.random.default_rng(11)
    raw._data[raw.ch_names.index(pd_chs[1])] = rng.random(pd.size) / 1e8
    return pd_chs


def _fix_pd2(raw, width=20):
    pd_chs = _get_pd_ch_names(raw)
    pd = (
        raw._data[raw.ch_names.index(pd_chs[0])]
        - raw._data[raw.ch_names.index(pd_chs[1])]
    )
    pd1, pd2 = [
        np.concatenate(
            [
                pd[:width],
                np.array(
                    [
                        func([pd[j + i] for i in range(-width, width + 1)])
                        for j in range(width, pd.size - width)
                    ]
                ),
                pd[-width:],
            ]
        )
        for func in (min, max)
    ]
    fig, ax = plt.subplots()
    ax.plot(pd, color="black", alpha=0.25)
    ax.plot(pd1, color="red", alpha=0.75)
    ax.plot(pd2, color="blue", alpha=0.75)
    ax.set_xlim((pd.size // 2 - 10000, pd.size // 2 + 10000))
    fig.show()
    choice = None
    while choice not in ("r", "b", "o"):
        choice = input(
            "Which has the correct photodiode shape? " "(red/blue/original)\t"
        ).lower()
        if choice:
            choice = choice[0]
    pd = pd if choice == "o" else (pd1 if choice == "r" else pd2)
    pd -= np.median(pd)
    if choice == "b":
        pd = -pd
    fig, ax = plt.subplots()
    ax.plot(pd)
    fig.show()
    thresh = float(input(
        "Zoom in on plot to find where deflections are "
        "higher than the baseline. Threshold?\t"
    ))
    pd = pd > thresh
    raw._data[raw.ch_names.index(pd_chs[0])] = pd
    rng = np.random.default_rng(11)
    raw._data[raw.ch_names.index(pd_chs[1])] = rng.random(pd.size) / 1e8
    return pd_chs


def normalize_channel_names(raw):
    raw.rename_channels(
        {
            ch: ch.replace("-20000", "").replace("-2000", "").replace("-200", "")
            for ch in raw.ch_names
        }
    )
    raw.set_channel_types({ch: ('ecog' if ch.lower().startswith('g') else 'seeg')
                           for ch in raw.ch_names})
    raw.set_channel_types({ch: 'misc' for ch in raw.ch_names if
                           'ekg' in ch.lower() or 'res' in ch.lower()})
    return raw


# modified from mne-bids
def events_tsv(raw):
    annot = raw.annotations
    events, event_id = mne.event_from_annotations(raw)

    # Onset column needs to be specified in seconds
    data = OrderedDict(
        [
            ("onset", annot.onset),
            ("duration", annot.duration),
            ("trial_type", annot.description),
            ("value", events[:, 2]),
            ("sample", events[:, 0]),
        ]
    )
    return DataFrame(data)


def find_events(sub, task, root):
    raw_fname = input("Intracranial recording file path?\t").strip()
    beh_fname = input("Behavior tsv file path?\t").strip()
    run = input('Run?\t')
    raw = mne.io.read_raw(raw_fname, preload=True)
    check = input("Preprocess to fix noise? (Y/n/2)\t").lower()
    if check == "y":
        pd_ch_names = _fix_pd(raw)
    elif check == "2":
        pd_ch_names = _fix_pd2(raw)
    else:
        pd_ch_names = None
    if task == "mirror":
        df = _read_tsv(beh_fname)
        for col in df:
            if col.endswith("onset"):
                df.loc[:, col] = [t / 1000 for t in df[col]]
        _to_tsv("tmp.tsv", df)
        annot, samples = pd_parser.parse_pd(
            raw=raw,
            pd_ch_names=pd_ch_names,
            pd_event_name="WatchOnset",
            beh="tmp.tsv",
            beh_key="watch_onset",
            max_len=20,
            recover=True,
            overwrite=True,
        )
        pd_parser.add_relative_events(
            raw=raw_fname,
            beh=beh_fname,
            relative_event_keys=["fix_duration", "go_time", "response_time"],
            relative_event_names=["ISI Onset", "Go Cue", "Response"],
        )
        annot, pd_ch_names, df = _load_data(raw)
    elif task in ("numbers", "food"):
        df = read_csv(beh_fname, sep="\t")
        df = df[
            ["number", "trial_type"]
            if task == "numbers"
            else ["stimuli"]
            + [
                "response_trial",
                "correct",
                "response_time",
                "fixation_start",
                "stimulus_start",
                "iti_start",
            ]
        ]
        df = df.fillna("n/a")
        for col in ("fixation_start", "stimulus_start", "iti_start"):
            df.loc[:, col] = [t / 1000 for t in df[col]]
        df.loc[:, "stimulus_start"] -= df["fixation_start"]
        df.loc[:, "iti_start"] -= df["fixation_start"]
        df.loc[:, "response_time"] = [
            "n/a" if t == "n/a" else int(t) / 1000 for t in df["response_time"]
        ]
        df.to_csv("tmp.tsv", sep="\t", index=False)
        annot, samples = pd_parser.parse_pd(
            raw=raw,
            pd_ch_names=pd_ch_names,
            pd_event_name="FixationOnset",
            zscore=25,
            max_len=2.5,
            beh="tmp.tsv",
            beh_key="fixation_start",
            recover=True,
            overwrite=True,
        )
        # EKG+RES3, EKG+RES4
        # EKG+RES6, EKG+RES6
        pd_parser.add_relative_events(
            raw=raw,
            beh="tmp.tsv",
            relative_event_keys=["iti_start", "stimulus_start", "response_time"],
            relative_event_names=["ISI Onset", "Cue", "Response"],
        )
        annot, pd_ch_names, beh = _load_data(raw)
        df = read_csv(beh_fname, sep="\t")
        df.loc[:, "pd_parser_sample"] = beh["pd_parser_sample"]
    if "iat" in task:
        df = read_csv(beh_fname, sep="\t")
        df = DataFrame(
            dict(stimulus_start=[t / 1000 for t in df["stimulus_start"][::2]])
        )
        df.to_csv("tmp.tsv", sep="\t", index=False)
        annot, samples = pd_parser.parse_pd(
            raw=raw,
            pd_ch_names=pd_ch_names,
            pd_event_name="Cue1",
            zscore=25,
            max_len=2,
            beh="tmp.tsv",
            beh_key="stimulus_start",
            recover=True,
            overwrite=True,
        )
        pd_parser.add_pd_off_events(
            raw=raw,
            off_event_name="Cue2",
            zscore=25,
            max_len=2,
            overwrite=True,
        )
        annot, pd_ch_names, beh = _load_data(raw)
        annot_samples = (annot.onset * raw.info['sfreq']).round().astype(int)
        pd_samples = list()
        for i, entry in enumerate(beh["pd_parser_sample"]):
            pd_samples.append(entry)
            if entry == 'n/a':
                pd_samples.append('n/a')
            else:
                entry2 = beh["pd_parser_sample"][i + 1] if \
                    i + 1 < len(beh["pd_parser_sample"]) - 1 else np.inf
                entry2 = entry + 2 * raw.info['sfreq'] if entry2 == 'n/a' else entry2
                offset_samples = np.logical_and(annot_samples > entry, annot_samples < entry2)
                if offset_samples.sum() == 1:
                    pd_samples.append(annot_samples[offset_samples][0])
                else:
                    pd_samples.append('n/a')
        df = read_csv(beh_fname, sep="\t")
        df = df[["stimuli", "block_i", "correct", "response_time", "stimulus_start"]]
        df.loc[:, "pd_parser_sample"] = [np.nan if i == 'n/a' else i for i in pd_samples]
        df.loc[:, "stimulus_start"] = [t / 1000 for t in df["stimulus_start"]]
        df.loc[:, "response_time"] = [
            int(t) / 1000 if isinstance(t, str) and t.isdigit() else "n/a"
            for t in df["response_time"]
        ]
    bids_path = mne_bids.BIDSPath(subject=str(sub), task=task, run=run, root=root)
    raw = normalize_channel_names(raw)
    print('Please mark bad channels by clicking on them')
    raw.compute_psd().plot(show=False)
    raw.plot()
    mne_bids.write_raw_bids(raw, bids_path, anonymize=dict(daysback=40000),
                            overwrite=True)
    raw_json_fname = op.join(root, f'sub-{sub}', 'ieeg',
                             f'sub-{sub}_task-{task}_run-{run}_ieeg.json')
    # add JSON fields
    with open(raw_json_fname, 'r') as fid:
        data = json.load(fid)
    manufacturer = input('Manufacturer? [Cadwell]\t')
    data['Manufacturer'] = 'Cadwell' if manufacturer == '' else manufacturer
    powerline = input('Powerline frequency (60 US, 50 EU)? [60]\t')
    data['PowerLineFrequency'] = 60 if powerline == '' else powerline
    with open(raw_json_fname, 'w') as fid:
        fid.write(json.dumps(data, indent=4))
    # save behavior
    beh_dir = op.join(root, f"sub-{sub}", "beh")
    os.makedirs(beh_dir, exist_ok=True)
    df.to_csv(op.join(beh_dir, f"sub-{sub}_task-{task}_run-{run}_beh.tsv"),
              sep="\t", index=False)
    raw.set_annotations(annot)
    # save events
    events_df = events_tsv(raw)
    events_df.to_csv(op.join(root, f'sub-{sub}', 'ieeg',
                             f'sub-{sub}_task-{task}_run-{run}_events.tsv'),
                     sep='\t', index=False)


if __name__ == "__main__":
    fs_subjects_dir = op.join(os.environ['FREESURFER_HOME'], 'subjects')
    template_t1_fname = op.join(os.environ['FREESURFER_HOME'], 'subjects',
                                TEMPLATE, 'mri', 'T1.mgz')
    if not op.isfile(template_t1_fname):
        print('Running recon-all for template, this only has to be done '
              'once but takes 8 hours or so')
        os.environ["SUBJECTS"] = TEMPLATE
        os.environ["SUBJECTS_DIR"] = fs_subjects_dir
        recon_cmd = which("recon-all")
        tpath = tflow.get(TEMPLATE)
        template_t1_input_fname = \
            [f for f in tpath if 'T1' in str(f) and 'brain' not in str(f) and
             str(f).endswith('nii.gz')][0]
        template_t2_input_fnames = \
            [f for f in tpath if 'T2' in str(f) and 'brain' not in str(f) and
             str(f).endswith('nii.gz')]
        cmd = f"{recon_cmd} -subjid {TEMPLATE} -i {template_t1_input_fname} "
        for template_t2_input_fname in template_t2_input_fnames[:1]:
            cmd += f"-T2 {template_t1_input_fname} "
        cmd += "-all -cw256"
        with open(f"{TEMPLATE}_recon_output.txt", "w") as fid:
            Popen(
                cmd.split(" "),
                stdout=fid,
                stderr=fid,
                env=os.environ,
            )
    root = input('BIDS directory?\t')
    sub = input("Subject ID number?\t")
    task = input("Task?\t")
    subjects_dir = op.join(root, "derivatives", "freesurfer")
    work_dir = op.join(subjects_dir, "ieeg-preprocessing", f"sub-{sub}")
    for dtype in ('anat', 'ieeg', 'figures', 'tmp'):
        os.makedirs(op.join(work_dir, dtype), exist_ok=True)
    print_status(sub, task, subjects_dir, work_dir)
    do_step("Find events", find_events, sub, task, root)
    do_step('Convert DICOMs', import_dicom, sub, root, work_dir,
            subjects_dir, fs_subjects_dir)
    do_step('Import MR', import_mr, sub, root, work_dir,
            subjects_dir, fs_subjects_dir)
    do_step('Import CT', import_ct, sub, root, work_dir,
            subjects_dir, fs_subjects_dir)
    do_step("Align CT", align_CT, sub, work_dir, subjects_dir)
    do_step("Find contacts", find_contacts, sub, root, work_dir, subjects_dir)
    do_step("Warp to template", warp_to_template, sub, root, work_dir,
            subjects_dir, fs_subjects_dir)
