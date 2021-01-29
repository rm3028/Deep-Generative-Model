
import re
import pandas as pd
from skimage import io
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from enum import Enum, auto


class ActivationType(Enum):
    AT_SIGMOID = auto()
    AT_TANH = auto()


class AnimeDataset(Dataset):
    def __init__(self, dataset_dir, activation_type=ActivationType.AT_SIGMOID):
        self.dataset_dir = dataset_dir
        self.dataset_df = pd.read_csv(dataset_dir + '/tags_clean.csv', names=['id', 'tags'])
        self.transform = T.Compose([T.ToTensor()])
        self.activation_type = activation_type

    def __len__(self):
        return len(self.dataset_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.dataset_dir + '/faces/' + str(self.dataset_df['id'][idx]) + '.jpg'
        image = io.imread(img_name)
        image = self.transform(image)

        if self.activation_type == ActivationType.AT_TANH:
            image = image * 2 - 1

        img_tags = self.dataset_df['tags'][idx]
        img_tags = re.split('\t', img_tags)
        img_tags = [string for string in img_tags if string != '']
        img_tags = {sub_tags.split(':')[0]: float(sub_tags.split(':')[1]) for sub_tags in img_tags}

        return { 'image': image, 'tags': torch.zeros_like(image) }

if __name__ == '__main__':
    animeDataset = AnimeDataset('data/AnimeDataset')
    for i in range(len(animeDataset)):
        try:
            animeDataset[i]
        except:
            print(i)
    pass
