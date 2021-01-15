
import pandas as pd
from skimage import io
import torch
from torch.utils.data import Dataset


class ExtraDataset(Dataset):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.dataset_df = pd.read_csv(dataset_dir + '/tags.csv', names=['id', 'tag'])

    def __len__(self):
        return len(self.dataset_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.dataset_dir + '/images/' + str(self.dataset_df['id'][idx]) + '.jpg'
        image = io.imread(img_name)

        img_tag = self.dataset_df['tag'][idx]

        return { 'image': image, 'tag': img_tag }
