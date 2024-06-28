import nibabel as nib
#%%
# Load the fMRI file
file_path = '/Volumes/swagbox/moments/fmri/osfstorage-archive/localizers/sub-01/sub-01_task-FBOS_space-T1w_roi-face-pSTS_hemi-rh_roi-mask.nii.gz'
img = nib.load(file_path)

# Access the data array
data = img.get_fdata()

# Check the shape of the data array
print("Data shape:", data.shape)

# Optionally, you can also check other information about the image
print("Voxel dimensions:", img.header.get_zooms())
print("Affine matrix:", img.affine)

#%% Sociality envelope
from scipy.io import loadmat
mat_path = '/Users/alonz/Downloads/Sociality and Interaction Envelope Organize Visual Action Representations/1-Stimuli/Stimulus Ratings/FeatureRatings.mat'
mat = loadmat(mat_path)
vals = mat['Ratings']