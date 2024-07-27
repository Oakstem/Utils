import nibabel as nib
from nilearn import datasets
from nilearn.input_data import NiftiMasker
import numpy as np
from nilearn.image import new_img_like

def build_masker(fmri_img, voxel_coords, radius=2):
    """ Create a spherical ROI mask around each voxel coordinate in MNI space
    radius is set in voxels"""

    # Initialize an empty mask with the same shape as the fMRI data
    mask_data = np.zeros(fmri_img.shape[:3], dtype=bool)

    # Create a spherical ROI mask around each voxel coordinate
    for coord in voxel_coords:
        for x in range(coord[0] - radius, coord[0] + radius + 1):
            for y in range(coord[1] - radius, coord[1] + radius + 1):
                for z in range(coord[2] - radius, coord[2] + radius + 1):
                    if np.linalg.norm(coord - np.array([x, y, z])) <= radius:
                        if 0 <= x < mask_data.shape[0] and 0 <= y < mask_data.shape[1] and 0 <= z < mask_data.shape[2]:
                            mask_data[x, y, z] = True

    # Create a Nifti image for the mask
    mask_img = new_img_like(fmri_img, mask_data.astype(int))
    return mask_img

def get_mask_from_atlas(atlas_filename, atlas_labels, roi_label):
    """ Create a mask for a specific region of interest (ROI) in an atlas"""
    roi_label_index = atlas_labels.index(roi_label)
    atlas_img = nib.load(atlas_filename)
    atlas_data = atlas_img.get_fdata()
    pSTS_mask_data = np.zeros(atlas_data.shape, dtype=np.int)
    pSTS_mask_data[atlas_data == roi_label_index] = 1

    # Create a Nifti image for the pSTS mask
    pSTS_mask_img = new_img_like(atlas_img, pSTS_mask_data)
    return pSTS_mask_img

def get_atlas_mask_routine():
    # Load the atlas
    atlas_type = 'harvard'  # or 'hcp'
    if atlas_type == 'harvard':
        atlas_data = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-1mm')
        atlas_filename = atlas_data.filename
        # found only 'Superior Temporal Gyrus, posterior division'
    elif atlas_type == 'hcp':
        atlas = nib.load('/Users/alonz/Downloads/3501911/HCP-MMP1_on_MNI152_ICBM2009a_nlin_hd.nii.gz')
        atlas_data = atlas.get_fdata()
        L_pSTS_indices = [162, 161]  # Left Posterior Superior Temporal Sulcus TE1p + TE2p
        # R_pSTS_indices = [362, 361]
        atlas_filename = atlas['maps']
        # atlas_labels = atlas['labels']

    # Load the atlas labels
    atlas_labels = atlas_data.labels

    # Find the index for the pSTS label
    pSTS_label = 'Superior Temporal Gyrus, posterior division'
    mask = get_mask_from_atlas(atlas_filename, atlas_labels, pSTS_label)

    return mask

# Convert MNI coordinates to voxel indices
def mni_to_voxel(mni_coords, affine):
    inv_affine = np.linalg.inv(affine)
    voxel_coords = nib.affines.apply_affine(inv_affine, mni_coords)
    return np.round(voxel_coords).astype(int)

# Load your fMRI data
localization_method = 'mni'  #  atlas | mni
fmri_img = nib.load('/Users/alonz/PycharmProjects/HAD-fmri/validation/results/brain_map_individual/reliability/sub-01_reliability.dtseries.nii')

if 'mni' in localization_method:
    mni_coords = [50, -48, 15]
    voxel_coords = mni_to_voxel(mni_coords, fmri_img.affine)
    psts_mask = build_masker(fmri_img, [voxel_coords], radius=2)
elif 'atlas' in localization_method:
    psts_mask = get_atlas_mask_routine()

# Initialize the NiftiMasker with the pSTS mask
masker = NiftiMasker(mask_img=psts_mask)

# Extract the time series from the fMRI data within the pSTS ROI
pSTS_time_series = masker.fit_transform(fmri_img)
pass
# pSTS_time_series is now a 2D array with shape (n_volumes, n_voxels)
