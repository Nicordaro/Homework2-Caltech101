from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys
import numpy

train_idx_dir = 'Caltech101/train.txt'
test_idx_dir = 'Caltech101/test.txt'


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split  # This defines the split you are going to use
        # (split files are called 'train.txt' and 'test.txt'



        #self.categories = {i: train[i] for i in range(0, len(listOfStr))}
        self.indexes = []



        if split=='train':
            ##read all idxs for the train
            with open(train_idx_dir) as idx:
                data = idx.read().splitlines()

                self.imgs = []


                #prendo righe con percorsi img
                for i in data:
                    #divido in categ e img relativa a categoria
                    categ = i.split("/")[0]
                    #scarto bg google
                    if categ != 'BACKGROUND_Google':
                        #lista di categorie con duplicati, è ordinata in modo da tenere traccia di categoria
                        # in base a posizione
                        self.imgs.append(categ)
                self.categories = list(numpy.unique(self.imgs))

            for ind, imag in enumerate(self.imgs):
                    self.indexes.append(ind)

        if split == 'test':
            ##read all idxs for the train
            with open(test_idx_dir) as idx:
                data = idx.read().splitlines()

                self.imgs = []

                # prendo righe con percorsi img
                for i in data:
                    # divido in categ e img relativa a categoria
                    categ = i.split("/")[0]
                    # scarto bg google
                    if categ != 'BACKGROUND_Google':
                        # lista di categorie con duplicati, è ordinata in modo da tenere traccia di categoria
                        # in base a posizione
                        self.imgs.append(categ)
                self.categories = list(numpy.unique(self.imgs))

            for ind, imag in enumerate(self.imgs):
                self.indexes.append(ind)



        # self.categories = sorted(os.listdir('Caltech101/'))
        # #self.categories.remove('BACKGROUND_Google')
        #
        # # self.index = []
        # # self.y = []
        # # for(i,c) in enumerate(self.categories):
        # #    self.index.append(i)
        # #    self.y.append(c)
        # self.index = []
        # self.y = []
        # for (i, c) in enumerate(self.categories):
        #     print(c)
        #     n = len(os.listdir('Caltech101/'))
        #     self.index.extend(range(1, n + 1))
        #     self.y.extend(n * [i])
        # print(self.y)
        # print(self.index)
        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class) 
        '''

    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''

        def getimg(idx):
            lab = self.imgs[idx]
            return lab

        image, label = pil_loader(index), getimg(index)  # Provide a way to access image and label via index
        # Image should be a PIL Image
        # label can be int

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = len(self.index)
        return length

Caltech('Caltech101').__init__('')
img, lab = Caltech('Caltech').__getitem__(2)
