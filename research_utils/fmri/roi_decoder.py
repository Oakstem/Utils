import os
import numpy as np
import nibabel as nib
from pprint import pprint
from nilearn.plotting import plot_roi
from nilearn.image import resample_to_img
from nimare.dataset import Dataset
from nimare.decode import discrete
from nimare.utils import get_resource_path
from nilearn.plotting import find_xyz_cut_coords
from nimare.io import convert_neurosynth_to_dataset
from nimare.extract import download_abstracts, fetch_neuroquery, fetch_neurosynth


class ROIDecoder:
    def __init__(self, dataset_path):
        self._load_dataset(dataset_path)

    def _load_dataset(self, dataset_path=None):
        if dataset_path is None:
            dataset_path = r"./neurosynth/neurosynth_dataset.pkl.gz"
        if not os.path.exists(dataset_path):
            self.dataset = self.download_neurosynth()
        else:
            self.dataset = Dataset.load(dataset_path)

    def download_neurosynth(self):
        out_dir = os.path.abspath("./neurosynth")
        os.makedirs(out_dir, exist_ok=True)

        files = fetch_neurosynth(
            data_dir=out_dir,
            version="7",
            overwrite=False,
            source="abstract",
            vocab="terms",
        )
        # Note that the files are saved to a new folder within "out_dir" named "neurosynth".
        pprint(files)
        neurosynth_db = files[0]

        ###############################################################################
        # Convert Neurosynth database to NiMARE dataset file
        # -----------------------------------------------------------------------------
        print("\nConverting Neurosynth database to NiMARE format dataset file...")
        neurosynth_dset = convert_neurosynth_to_dataset(
            coordinates_file=neurosynth_db["coordinates"],
            metadata_file=neurosynth_db["metadata"],
            annotations_files=neurosynth_db["features"],
        )
        neurosynth_dset.save(os.path.join(out_dir, "neurosynth_dataset.pkl.gz"))
        print(neurosynth_dset)
        return neurosynth_dset

    def prepare_roi(self, mask_img):
        self.mask_img = resample_to_img(mask_img, self.dataset.masker.mask_img, interpolation='nearest')
        self.xyz_coords = find_xyz_cut_coords(self.mask_img)
        self.study_ids = self.dataset.get_studies_by_mask(self.mask_img)


    def brain_map_decoder(self, mask_img, study_ids):
        decoder = discrete.BrainMapDecoder(correction=None)
        decoder.fit(self.dataset)
        decoded_df = decoder.transform(ids=study_ids)
        top_results = decoded_df.sort_values(by="probReverse", ascending=False).head()
        return top_results, decoded_df

    def neurosynth_chi_decoder(self, mask_img, study_ids):
        decoder = discrete.NeurosynthDecoder(correction=None)
        decoder.fit(self.dataset)
        decoded_df = decoder.transform(ids=study_ids)
        top_results = decoded_df.sort_values(by="probReverse", ascending=False).head()
        return top_results, decoded_df

    def neurosynth_roi_decoder(self, mask_img):
        decoder = discrete.ROIAssociationDecoder(mask_img)
        decoder.fit(self.dataset)
        decoded_df = decoder.transform()
        top_results = decoded_df.sort_values(by="r", ascending=False).head()
        return top_results, decoded_df

    def get_roi_name(self, mask_img, decoder_type='neurosynth'):
        if self.dataset is None:
            return {'top_name': 'No dataset', 'top5': None, 'all_results': None, 'mask_img': None, 'center_coords': None}

        self.prepare_roi(mask_img)

        if decoder_type == 'brain_map':
            top_results, decoded_df = self.brain_map_decoder(self.mask_img, self.study_ids)
        elif decoder_type == 'neurosynth-chi':
            top_results, decoded_df = self.neurosynth_chi_decoder(self.mask_img, self.study_ids)
        elif decoder_type == 'neurosynth-roi':
            top_results, decoded_df = self.neurosynth_roi_decoder(self.mask_img)
        else:
            raise ValueError(f'Unknown decoder type: {decoder_type}')

        top_name = top_results.index[0].split('__')[-1]
        results = {'top_name': top_name, 'top5': top_results, 'all_results': decoded_df,
                   'mask_img': self.mask_img, 'center_coords': self.xyz_coords}
        return results



if __name__ == '__main__':
    local_dataset_file = '/Users/alonz/PycharmProjects/pSTS_DB/psts_db/NiMARE/neurosynth_dataset_with_abstracts.pkl.gz'
    mask_img = nib.load(
        r"E:\NND\ds002837-download\derivatives\sub-1\func\sub-1_task-500daysofsummer_bold_blur_censor_ica.nii.gz")

    roi_decoder = ROIDecoder(None)
    results = roi_decoder.get_roi_name(mask_img, decoder_type='neurosynth-roi')

    print(f'Top ROI name: {results["top_name"]}')
    print(f'ROI XYZ: {results["center_coords"]}')
