import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class MedReconDataset(Dataset):
    """ 3D Reconstruction Dataset."""
    def __init__(self, file_list, data_root, num_views, input_size, output_size, transform=None):
        """
        Args:
            file_list (string): Path to the csv file with annotations.
            data_root (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.df = pd.read_csv(file_list)
        self.data_root = data_root
        self.transform = transform

        self.num_views = num_views
        self.input_size = input_size
        self.output_size = output_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        projs = np.zeros((self.input_size, self.input_size, self.num_views), dtype=np.uint8)  # input image size (H, W, C)

        # load 2D projections
        for view_idx in range(self.num_views):
            proj_path = self.df.iloc[idx]['view_%d' %(view_idx)]
            proj_path = os.path.join(self.data_root, proj_path[8:])

            # resize 2D images
            proj = Image.open(proj_path).resize((self.input_size, self.input_size))
            projs[:, :, view_idx] = np.array(proj)

        if self.transform:
            projs = self.transform(projs)

        # load 3D images
        image_path = self.df.iloc[idx]['3d_model']
        image_path = os.path.join(self.data_root, image_path[8:])
        image = np.fromfile(image_path, dtype=np.float32)
        image = np.reshape(image, (-1, self.output_size, self.output_size))

        # scaling normalize for 3D images
        image = image - np.min(image)
        image = image / np.max(image)
        assert((np.max(image)-1.0 < 1e-3) and (np.min(image) < 1e-3))

        image = torch.from_numpy(image)

        return (projs, image)

