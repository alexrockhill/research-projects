# to run, first run in a terminal (one time only):
# pip install mne mne_bids nibabel dipy pd_parser
# then run in a terminal:
# sudo python /path/to/this/file/ieeg_preprocessing.py

import os
import os.path as op
import json
import numpy as np
from collections import OrderedDict
import time
from shutil import which, copyfile
from contextlib import redirect_stdout
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
from AFQ.api.group import GroupAFQ

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

PROCS = dict()


def print_status(sub, root, work_dir):
    ieeg_dir = op.join(root, f'sub-{sub}', 'ieeg')
    if op.isfile(ieeg_dir):
        for events_fname in [f for f in os.listdir(ieeg_dir) if
                             f.endswith('events.tsv')]:
            name_dict = dict([kv.split('-') for kv in events_fname.split('_')[:-1]])
            print('Task data file complete: {}'.format(name_dict['task']))
    for name, fname in zip(
        [
            "Import MR",
            "Import DWI",
            "Import CT",
            "Find Contacts",
            "Warp to Template",
        ],
        [
            op.join(work_dir, 'anat' "T1.mgz"),
            op.join(root, f'sub-{sub}', 'dwi', f'sub-{sub}_dwi.nii.gz'),
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


def import_dicom(sub, work_dir):
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
              f"size={op.getsize(fname) / 1e6} MB {vox_row} mm")
    print('\n' * 2)
    return nii_files


def choose_file(ftype, nii_files):
    if ftype == 'T2':
        print('T2 not required, press enter to skip')
    inputs = f'1 - {len(nii_files)} or file path' if nii_files else 'file path'
    idx = None
    while idx is None or idx < 0 or idx > len(nii_files) - 1:
        out = input(f'{ftype} file ({inputs})?\t')
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
    # landmarks = mne.transforms.apply_trans(np.linalg.inv(reg_affine), landmarks)
    landmarks_vox = mne.transforms.apply_trans(np.linalg.inv(t1.affine), landmarks)
    landmarks_dict = dict(LPA=list(landmarks_vox[0]),
                          NAS=list(landmarks_vox[1]),
                          RPA=list(landmarks_vox[2]))

    t1_defaced = deface(t1, landmarks, inset=INSET, theta=THETA)

    # save to BIDS
    t1_acpc_fname = op.join(work_dir, 'anat', 'T1.mgz')
    os.makedirs(anat_dir, exist_ok=True)
    nib.save(t1_defaced, op.join(anat_dir, f'sub-{sub}_T1w.nii.gz'))
    if t1_json_fname:
        with open(t1_json_fname, "r+") as fid:
            scan_info = json.load(fid)
        scan_info['AnatomicalLandmarks'] = landmarks_dict
        with open(op.join(anat_dir, f'sub-{sub}_T1w.json'), 'w') as fid:
            fid.write(json.dumps(scan_info, indent=4))
    t1_acpc = mne.transforms.apply_volume_registration(
        t1_defaced, brain_template, np.linalg.inv(reg_affine))
    nib.save(t1_acpc, t1_acpc_fname)

    t2_acpc_fname = op.join(work_dir, 'anat', 'T2.mgz')
    if t2_fname:
        t2 = nib.load(t2_fname)
        t2_defaced = deface(t2, landmarks, inset=INSET, theta=THETA)
        nib.save(t2_defaced, op.join(anat_dir, f'sub-{sub}_T2w.nii.gz'))
        if t2_json_fname:
            with open(t2_json_fname, "r") as fid:
                scan_info = json.load(fid)
            scan_info['AnatomicalLandmarks'] = landmarks_dict
            with open(op.join(anat_dir, f'sub-{sub}_T2w.json'), 'w') as fid:
                fid.write(json.dumps(scan_info, indent=4))
        t2_acpc = mne.transforms.apply_volume_registration(
            t2_defaced, brain_template, np.linalg.inv(reg_affine))
        nib.save(t2_acpc, t2_acpc_fname)

    renderer = mne.viz.create_3d_figure((600, 600), scene=False, show=True)
    t1_defaced_data = np.array(t1_defaced.dataobj)
    verts, tris = mne.surface._marching_cubes(
        t1_defaced_data > np.quantile(t1_defaced_data[t1_defaced_data > 0], 0.5),
        [1])[0]
    verts = mne.transforms.apply_trans(t1.affine, verts)
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
    cmd += "-all -cw256"
    with open(op.join(tmp_dir, f"sub-{sub}_recon_output.txt"), "w") as fid:
        proc = Popen(cmd.split(" "), stdout=fid, stderr=fid, env=os.environ)
        PROCS['recon-all'] = proc

    # make head surfaces
    pool = Pool(processes=1)
    proc = pool.apply_async(
        make_head_surface, [sub, work_dir, subjects_dir, t2_acpc_fname])
    PROCS['make-head-surface'] = proc


def import_dwi(sub, root, work_dir, subjects_dir):
    my_run = 1
    check = True
    tmp_dir = op.join(work_dir, "tmp")
    while check:
        nii_files = print_imported_nii()
        dwi_fname, dwi_json_fname = choose_file('DWI', nii_files)
        dwi_dir = op.join(root, f'sub-{sub}', 'dwi')
        os.makedirs(dwi_dir, exist_ok=True)
        copyfile(dwi_fname, op.join(dwi_dir, f'sub-{sub}_run-{my_run}_dwi.nii.gz'))
        for ext in ('bval', 'bvec'):
            copyfile(dwi_fname.split('.')[0] + f'.{ext}',
                     op.join(dwi_dir, f'sub-{sub}_run-{my_run}_dwi.{ext}'))
        if op.isfile(dwi_json_fname):
            with open(dwi_json_fname, "r") as fid:
                scan_info = json.load(fid)
            if scan_info['Manufacturer'] == 'Philips':  # fix known issues
                scan_info['PhaseEncodingDirection'] = "j-"
                scan_info['TotalReadoutTime'] = scan_info['EstimatedTotalReadoutTime']
            with open(op.join(dwi_dir, f'sub-{sub}_run-{my_run}_dwi.json'), 'w') as fid:
                fid.write(json.dumps(scan_info, indent=4))
        check = input('Add another? (Y/n)\t').lower() == 'y'
        if check:
            my_run += 1
    # qsiprep
    derivatives_dir = op.join(root, 'derivatives')
    license_fname = op.join(os.environ['FREESURFER_HOME'], 'license.txt')
    work_dir_qsi = op.join(derivatives_dir, 'temp_qsi')
    cmd = f"qsiprep-docker {root} {derivatives_dir} participant \
    --participant_label sub-{sub} \
    --fs-license-file {license_fname} \
    --work-dir {work_dir_qsi} \
    --output-resolution 1.25 \
    --write-graph \
    -vv"
    with open(op.join(tmp_dir, f"sub-{sub}_qsiprep_output.txt"), "w") as fid:
        proc = Popen(cmd.split(" "), stdout=fid, stderr=fid, env=os.environ)
        PROCS['qsiprep'] = proc
    # pyafq
    pool = Pool(processes=1)
    proc = pool.apply_async(run_pyafq, [proc, op.join(derivatives_dir, 'qsiprep'),
                                        work_dir, subjects_dir])
    PROCS['pyafq'] = proc


def run_pyafq(proc, qsiprep_root, work_dir, subjects_dir):
    tmp_dir = op.join(work_dir, 'tmp')
    with open(op.join(tmp_dir, f"sub-{sub}_pyafq_output.txt"), "w") as fid:
        with redirect_stdout(fid):
            poll = None
            print('Waiting for qsiprep to finish')
            while poll is None:
                time.sleep(60)
                poll = proc.poll()
                print(".", end="", flush=True)
            run(f'sudo chown -R {os.getlogin()} {qsiprep_root}', shell=True)
            myafq = GroupAFQ(bids_path=qsiprep_root)
            myafq.export_all()


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
    landmarks_vox = mne.transforms.apply_trans(np.linalg.inv(ct.affine), landmarks)

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
        with open(ct_json_fname, "r") as fid:
            scan_info = json.load(fid)
        scan_info['AnatomicalLandmarks'] = dict(LPA=list(landmarks_vox[0]),
                                                NAS=list(landmarks_vox[1]),
                                                RPA=list(landmarks_vox[2]))
        with open(op.join(anat_dir, f'sub-{sub}_ct.json'), 'w') as fid:
            fid.write(json.dumps(scan_info, indent=4))


def make_head_surface(sub, work_dir, subjects_dir, t2_acpc_fname):
    tmp_dir = op.join(work_dir, 'tmp')
    with open(op.join(tmp_dir, f"sub-{sub}_head_surf_output.txt"), "w") as fid:
        with redirect_stdout(fid):
            _ensure_recon(f'sub-{sub}', 'T1', subjects_dir)
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
            subjects_dir, sub, "mri", "transforms", "talairach.xfm"
        )
    else:
        fname = op.join(subjects_dir, sub, "mri", f"{ftype}.mgz")
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
def electrodes_tsv(montage):
    pos = montage.get_positions()
    assert pos['coord_frame'] == 'ras'
    x, y, z, names = list(), list(), list(), list()
    for ch in pos['ch_pos']:
        loc = pos['ch_pos'][ch]
        if np.isnan(loc).any():
            x.append("n/a")
            y.append("n/a")
            z.append("n/a")
        else:
            x.append(loc[0])
            y.append(loc[1])
            z.append(loc[2])
        names.append(ch)

    data = OrderedDict([("name", names), ("x", x), ("y", y), ("z", z),
                        ("size", ["n/a"] * len(names))])
    return DataFrame(data)


def read_montage_from_electrodes_tsv(electrodes_fname, sub, subjects_dir):
    df = read_csv(electrodes_fname, sep='\t')
    montage = mne.channels.make_dig_montage(
        {name: [x, y, z] for name, x, y, z in zip(df.name, df.x, df.y, df.z)},
        coord_frame="ras"
    )
    mne_bids.convert_montage_to_mri(
        montage, subject=sub, subjects_dir=subjects_dir)
    return montage


def find_contacts(sub, root, work_dir, subjects_dir):
    if not op.isfile(op.join(work_dir, "anat", "CT.mgz")):
        raise RuntimeError('CT Align must be done first')
    mne.viz.set_3d_backend("pyvistaqt")
    ct = nib.load(op.join(work_dir, "anat", "CT.mgz"))
    ieeg_dir = op.join(root, f'sub-{sub}', 'ieeg')
    raw_fnames = [op.join(ieeg_dir, f) for f in os.listdir(ieeg_dir)
                  if f.endswith('edf') or f.endswith('vmrk')]
    if raw_fnames:
        raw_fname = raw_fnames[0]
    else:
        raw_fname = input("Intracranial recording file path?\t").strip()
    trans = mne.coreg.estimate_head_mri_t(f"sub-{sub}", subjects_dir)
    electrodes_fname = op.join(root, f'sub-{sub}', 'ieeg',
                               f'sub-{sub}_space-ACPC_electrodes.tsv')
    _ensure_recon(f'sub-{sub}', "T1", subjects_dir)
    _ensure_recon(f'sub-{sub}', "trans", subjects_dir)

    raw = mne.io.read_raw(raw_fname)
    raw = normalize_channel_names(raw)
    if op.isfile(electrodes_fname):
        montage = read_montage_from_electrodes_tsv(
            electrodes_fname, f'sub-{sub}', subjects_dir)
        montage.apply_trans(mne.transforms.invert_transform(trans))
        raw.set_montage(montage)
    mne_gui.locate_ieeg(raw.info, trans, ct, subject=f"sub-{sub}",
                        subjects_dir=subjects_dir)
    answer = None
    while answer != 'q':
        answer = input('Press "s" to save and "q" to quit\t').lower()
        if answer == "s":
            montage = raw.get_montage()
            montage.apply_trans(trans)
            mne_bids.convert_montage_to_ras(
                montage, subject=f'sub-{sub}', subjects_dir=subjects_dir)
            df = electrodes_tsv(montage)
            df.to_csv(electrodes_fname, sep='\t', index=False)

    # final save
    montage = raw.get_montage()
    montage.apply_trans(trans)
    mne_bids.convert_montage_to_ras(
        montage, subject=f'sub-{sub}', subjects_dir=subjects_dir)
    df = electrodes_tsv(montage)
    df.to_csv(electrodes_fname, sep='\t', index=False)
    coordsys_fname = op.join(root, f'sub-{sub}', 'ieeg',
                             f'sub-{sub}_space-ACPC_coordsystem.json')
    if not op.isfile(coordsys_fname):
        t1_fname = op.relpath(op.join(subjects_dir, f'sub-{sub}', 'mri', 'T1.mgz'),
                              root)
        with open(coordsys_fname, 'w') as fid:
            fid.write(json.dumps(dict(ACPC_COORDSYS, IntendedFor=t1_fname), indent=4))


def warp_to_template(sub, root, work_dir, subjects_dir, fs_subjects_dir):
    electrodes_fname = op.join(root, f'sub-{sub}', 'ieeg',
                               f'sub-{sub}_space-ACPC_electrodes.tsv')
    electrodes_fname_template = op.join(root, f'sub-{sub}', 'ieeg',
                                        f'sub-{sub}_space-{TEMPLATE}_coordsystem.tsv')
    if not op.isfile(electrodes_fname):
        raise RuntimeError('Find Contacts must be done first')

    montage = read_montage_from_electrodes_tsv(electrodes_fname, sub, subjects_dir)
    if op.isfile(electrodes_fname_template):
        print("Warning already exists, overwriting")
    print(f"Warping to {TEMPLATE} template brain, this will take about 15 minutes")
    # template data
    template_brain = nib.load(op.join(fs_subjects_dir, TEMPLATE, "mri", "brain.mgz"))
    # subject data
    subject_brain = nib.load(op.join(subjects_dir, f"sub-{sub}", "mri", "brain.mgz"))
    reg_affine, sdr_morph = mne.transforms.compute_volume_registration(
        subject_brain, template_brain, verbose=True
    )
    CT_aligned = nib.load(op.join(subjects_dir, f"sub-{sub}", "CT", "CT.mgz"))
    montage_warped = mne.preprocessing.ieeg.warp_montage(
        montage, subject_brain, template_brain, reg_affine, sdr_morph
    )
    # save elec image for sizes
    elec_image = mne.preprocessing.ieeg.make_montage_volume(montage, CT_aligned)
    nib.save(elec_image, op.join(work_dir, "ieeg", "elec_image.mgz"))
    # save electrode locations
    mne_bids.convert_montage_to_ras(
        montage_warped, subject=TEMPLATE, subjects_dir=fs_subjects_dir)
    df = electrodes_tsv(montage_warped)
    df.to_csv(electrodes_fname_template, sep='\t', index=False)
    coordsys_fname = op.join(root, f'sub-{sub}', 'ieeg',
                             f'sub-{sub}_space-{TEMPLATE}_coordsystem.json')
    if not op.isfile(coordsys_fname):
        with open(coordsys_fname, 'w') as fid:
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
            ch: ch.replace("-20000", "").replace("-2000", "").replace(
                "-200", "").replace('-Ref1', '')
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
    events, event_id = mne.events_from_annotations(raw)

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
    task = input("Task?\t").strip()
    beh_fname = input("Behavior tsv file path?\t").strip()
    run = input('Run?\t')
    raw = mne.io.read_raw(raw_fname)
    check = input("Preprocess to fix noise? (Y/n/2)\t").lower()
    if check in ("y", "2"):
        raw.load_data()
        pd_ch_names = _fix_pd(raw) if check == "y" else _fix_pd2(raw)
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
    elif "iat" in task:
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
    else:
        df = None
    bids_path = mne_bids.BIDSPath(subject=str(sub), task=task, run=run, root=root)
    raw = mne.io.read_raw(raw_fname)
    raw = normalize_channel_names(raw)
    print('Please mark bad channels by clicking on them')
    fig = raw.compute_psd().plot(show=False)
    fig.show()
    raw.plot(block=True)
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
    ref = input('Reference? [n/a]\t')
    data['iEEGReference'] = 'n/a' if ref == '' else ref
    with open(raw_json_fname, 'w') as fid:
        fid.write(json.dumps(data, indent=4))
    # save behavior
    if df is not None:
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
            proc = Popen(
                cmd.split(" "),
                stdout=fid,
                stderr=fid,
                env=os.environ,
            )
            PROCS['template-recon-all'] = proc
    root = input('BIDS directory?\t')
    bids_name = op.basename(root)
    sub = input("Subject ID number?\t")
    subjects_dir = op.join(root, "derivatives", "freesurfer")
    work_dir = op.join(root, '..', f"{bids_name}-ieeg-preprocessing", f"sub-{sub}")
    for dtype in ('anat', 'figures', 'tmp'):
        os.makedirs(op.join(work_dir, dtype), exist_ok=True)
    with open(op.join(root, '.bidsignore'), 'a+') as fid:
        if '\n*_ct.json\n*_ct.nii.gz' not in fid.read():
            fid.write('*_ct.json\n*_ct.nii.gz')
    print_status(sub, root, work_dir)
    do_step("Import ieeg data", find_events, sub, root)
    do_step('Convert DICOMs', import_dicom, sub, work_dir)
    do_step('Import MR', import_mr, sub, root, work_dir,
            subjects_dir, fs_subjects_dir)
    do_step('Import DWI', import_dwi, sub, root, work_dir, subjects_dir)
    do_step('Import CT', import_ct, sub, root, work_dir,
            subjects_dir, fs_subjects_dir)
    do_step("Align CT", align_CT, sub, work_dir, subjects_dir)
    do_step("Find contacts", find_contacts, sub, root, work_dir, subjects_dir)
    do_step("Warp to template", warp_to_template, sub, root, work_dir,
            subjects_dir, fs_subjects_dir)
    one_finished = True
    while PROCS:
        for name, proc in PROCS.copy().items():
            if (hasattr(proc, 'ready') and proc.ready()) or proc.poll() is not None:
                PROCS.pop(name)
                one_finished = True
        if PROCS:
            if one_finished:
                print(f'Waiting for {", ".join(PROCS.keys())} to finish, please '
                      'don\'t close the console')
                one_finished = False
            else:
                print('.', end='', flush=True)
            time.sleep(60)
