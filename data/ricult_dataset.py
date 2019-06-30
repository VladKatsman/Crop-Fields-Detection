import os
import collections
import torch
import numpy as np

from PIL import Image
from torch.utils import data
from glob import glob


def get_class_weights(root, num_classes):
    """ Find weights for disbalanced class problem"""
    train_path = "{}/masks/train".format(root)
    val_path = "{}/masks/val".format(root)
    train_files = glob("{}/*.png".format(train_path))
    val_files = glob("{}/*.png".format(val_path))
    files = train_files + val_files
    data = np.zeros(num_classes)
    for file in files:
        img = Image.open(file)
        img = np.array(img)
        stats = np.unique(img, return_counts=True)
        for i, key in enumerate(stats[0]):
            data[key] += stats[1][i]
    total_count = np.sum(data)
    weights = 1/(data/total_count)
    weights = weights/max(weights)

    return weights


class Ricult(data.Dataset):
    """Data loader for the CelebA Aligned benchmark face-hair semantic segmentation dataset.

        Args:
         root (str): path to dataset folder
         pil_transforms (obj): PIL transformations with both input and mask
         tensor_transforms (obj): Tensor transformations with input
    """

    def __init__(self, root, split, pil_transforms=None, tensor_transforms=None, visual_augmentations=None):
        self.root = os.path.expanduser(root)
        if split == 0:
            self.split = 'train'
        else:
            self.split = 'val'
        self.n_classes = 2
        self.files = collections.defaultdict(list)

        # list of images
        path = os.path.join(self.root, self.split + "_img.txt")
        file_list = tuple(open(path, "r"))
        file_list = [id_.rstrip() for id_ in file_list]
        self.images = file_list

        # list of segmentations
        path = os.path.join(self.root, self.split + "_seg.txt")
        file_list = tuple(open(path, "r"))
        file_list = [id_.rstrip() for id_ in file_list]
        self.segs = file_list
        self.pil_transforms = pil_transforms
        self.tensor_transforms = tensor_transforms
        self.visual_augmentations = visual_augmentations
        self.weights = torch.tensor(get_class_weights(root, self.n_classes)).float()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        idx = index
        im_name = self.images[idx]
        seg_name = self.segs[idx]
        im_path = os.path.join(self.root, im_name)
        seg_path = os.path.join(self.root, seg_name)
        image = Image.open(im_path)
        seg = Image.open(seg_path)

        if self.visual_augmentations is not None:
            im = self.visual_augmentations(image)
            im = Image.fromarray(im)

        if self.pil_transforms is not None:
            im, seg = self.pil_transforms(image, seg)
            # for val
            image = torch.from_numpy(np.array(im)).float()
            image = image.permute(2,0,1)
        if self.tensor_transforms is not None:
            im = self.tensor_transforms(im)

        seg = torch.from_numpy(np.array(seg)).long()
        if self.split == 'train':
            return im, seg

        elif self.split == 'val':
            return im, seg, image
