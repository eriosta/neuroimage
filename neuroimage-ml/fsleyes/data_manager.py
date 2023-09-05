import numpy as np
import nibabel as nib

class DataManager:
    def __init__(self):
        self.data = None
        self.metadata = None

    def load_data(self, nifti_file):
        # Load the NIFTI file
        img = nib.load(nifti_file)

        # Extract the data as a numpy array
        self.data = img.get_fdata()

        # Normalize the data for visualization
        self.data = self.normalize_data(self.data)

        # Extract the metadata
        self.metadata = img.header

    def normalize_data(self, data):
        # Normalize the data to the range [0, 1]
        data_min = np.min(data)
        data_max = np.max(data)
        return (data - data_min) / (data_max - data_min)
