import os.path as op
import numpy as np
import nibabel as nib
from mne.transforms import apply_trans, translation, rotation


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
    idxs = apply_trans(translation(x=-x, y=-y + inset, z=-z), idxs)
    idxs = apply_trans(rotation(x=-np.pi / 2 + np.deg2rad(theta)), idxs)
    idxs = idxs.reshape(image_data.shape + (3,))
    mask = idxs[..., 2] < 0  # z < middle
    image_data[mask] = 0

    image = nib.Nifti1Image(image_data, image.affine, image.header)
    return image


if __name__ == '__main__':
    t1_fname = input('T1 fname?\t').strip()
    t1 = nib.load(t1_fname)
    lpa = np.array([float(c.strip()) for c in input(
        'LPA? (RAS, comma separated)\t').split(',')])
    nas = np.array([float(c.strip()) for c in input(
        'Nasion? (RAS, comma separated)\t').split(',')])
    rpa = np.array([float(c.strip()) for c in input(
        'RPA? (RAS, comma separated)\t').split(',')])
    t1_defaced = deface(t1, [lpa, nas, rpa])
    nib.save(t1_defaced, op.join(op.dirname(
        t1_fname), op.basename(t1_fname).split('.')[0] + '_defaced.nii.gz'))
