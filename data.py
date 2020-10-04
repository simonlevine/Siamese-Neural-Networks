from torch.utils.data import Dataset
from torchvision.datasets.vision import VisionDataset
import warnings
from PIL import Image
import os
import os.path
import numpy as np
import torch
import codecs
import torch.utils.data as data
import torch
from torchvision.datasets import ImageFolder

class FaceData(VisionDataset):
    '''
    Args:
        root (string): Root directory of dataset where ``MNIST/processed/training.pt``
            and  ``MNIST/processed/test.pt`` exist.
        train (bool, optional): If True, creates train dataset from ,
            otherwise test.

        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    '''


    training_file = 'training.pt'
    test_file = 'test.pt'
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    hparams = {
            'train_data_path' : '../data/classification_data/train_data/',
            'test_data_path'  : '../data/classification_data/test_data/',
            'val_data_path'   : '../data/classification_data/val_data/', 
    }


    def __init__(self, root, train=True, transform=None, target_transform=None):
        super(FaceData, self).__init__(root, transform=transform,
                                    target_transform=target_transform)

        self.train_data_path = hparams.get('train_data_path', None)
        self.test_data_path = hparams.get('test_data_path', None)
        self.val_data_path = hparams.get('val_data_path', None)

        self.train = train  # training set or test set


        if self.train:
            self.dataset = ImageFolder(root=self.train_data_path)
            # self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))
        else: #test
            self.dataset = ImageFolder(root=self.test_data_path)
            # self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file)) 
            indices = [3, 2, 1, 18, 6, 8, 11, 17, 61, 16] # Manually picked support set
            self.support_set = np.stack([self.transform(self.dataset[x][0]) for x in indices])   

    def __getitem__(self, index):
        image1_index = np.random.randint(0, len(self.data))
        image1,_ = self.dataset[image1_index]
        target1 = int(self.dataset.targets[image1_index])

        image2,_ = self.dataset[index]
        target2 = int(self.dataset.targets[index])

        target = 1.0 if target1 == target2 else 0.0

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # image1 = Image.fromarray(image1.numpy(), mode='L')
        # image2 = Image.fromarray(image2.numpy(), mode='L')

        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
            
        return image1, image2, target, target1

    def __len__(self):
        return len(self.dataset)

    def class_to_idx(self):
        return self.dataset.class_to_idx

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")