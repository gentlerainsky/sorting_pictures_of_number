# This file is a copy (with minor modification) from https://github.com/Felix-Petersen/diffsort
#
# Copyright (c) Felix Petersen.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#@title Data set definition

import torch
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader


from torchvision.datasets.vision import VisionDataset
from PIL import Image
import os
import os.path
import numpy as np
from torchvision.datasets.utils import download_url, check_integrity, verify_str_arg
import torch
import torchvision.transforms as transforms
import clrs


class MultiDigitMNISTDataset(Dataset):
    def __init__(
            self,
            images,
            labels,
            num_digits,
            num_compare,
            num_list,
            seed=0,
            determinism=True,
    ):
        super(MultiDigitMNISTDataset, self).__init__()
        self.images = images
        self.labels = labels
        self.num_digits = num_digits
        self.num_compare = num_compare
        self.seed = seed
        self.rand_state = None
        self.num_list = num_list
        self.determinism = determinism

        if determinism:
            self.reset_rand_state()

    # def __len__(self):
        # return self.images.shape[0]

    def __len__(self):
        return self.num_list

    def __getitem__(self, idx):

        if self.determinism:
            prev_state = torch.random.get_rng_state()
            torch.random.set_rng_state(self.rand_state)

        labels = []
        images = []
        labels_ = None
        for digit_idx in range(self.num_digits):
            # id = torch.randint(len(self), (self.num_compare, ))
            id = torch.randint(self.images.shape[0], (self.num_compare, ))
            labels.append(self.labels[id])
            images.append(self.images[id].type(torch.float32) / 255.)
            if labels_ is None:
                labels_ = torch.zeros_like(labels[0] * 1.)
            labels_ = labels_ + 10.**(self.num_digits - 1 - digit_idx) * self.labels[id]

        images = torch.cat(images, dim=-1)

        if self.determinism:
            self.rand_state = torch.random.get_rng_state()
            torch.random.set_rng_state(prev_state)
        return images, labels_

    def reset_rand_state(self):
        prev_state = torch.random.get_rng_state()
        torch.random.manual_seed(self.seed)
        self.rand_state = torch.random.get_rng_state()
        torch.random.set_rng_state(prev_state)


class SVHNMultiDigit(VisionDataset):
    """`Preprocessed SVHN-Multi <>`_ Dataset.
    Note: The preprocessed SVHN dataset is based on the the `Format 1` official dataset.
    By cropping the numbers from the images, adding a margin of :math:`30\%` , and resizing to :math:`64\times64` ,
    the dataset has been preprocessed.
    The data split is as follows:

        * ``train``: (30402 of 33402 original ``train``) + (200353 of 202353 original ``extra``)
        * ``val``: (3000 of 33402 original ``train``) + (2000 of 202353 original ``extra``)
        * ``test``: (all of 13068 original ``test``)

    Each ```train / val`` split has been performed using
    ``sklearn.model_selection import train_test_split(data_X_y_tuples, test_size=3000 / 2000, random_state=0)`` .
    This is the closest that we could come to the
    `work by Goodfellow et al. 2013 <https://arxiv.org/pdf/1312.6082.pdf>`_ .

    Args:
        root (string): Root directory of dataset where directory
            ``SVHNMultiDigit`` exists.
        split (string): One of {'train', 'val', 'test'}.
            Accordingly dataset is selected.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop`` .
            (default = random 54x54 crop + normalization)
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    split_list = {
        'train': ["https://nyc3.digitaloceanspaces.com/publicdata1/svhn-multi-digit-3x64x64_train.p",
                  "svhn-multi-digit-3x64x64_train.p", "25df8732e1f16fef945c3d9a47c99c1a"],
        'val': ["https://nyc3.digitaloceanspaces.com/publicdata1/svhn-multi-digit-3x64x64_val.p",
                "svhn-multi-digit-3x64x64_val.p", "fe5a3b450ce09481b68d7505d00715b3"],
        'test': ["https://nyc3.digitaloceanspaces.com/publicdata1/svhn-multi-digit-3x64x64_test.p",
                 "svhn-multi-digit-3x64x64_test.p", "332977317a21e9f1f5afe7ef47729c5c"]
    }

    def __init__(self, root, split='train',
                 transform=transforms.Compose([
                     transforms.RandomCrop([54, 54]),
                     transforms.ToTensor(),
                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                 ]),
                 target_transform=None, download=False):
        super(SVHNMultiDigit, self).__init__(root, transform=transform, target_transform=target_transform)
        self.split = verify_str_arg(split, "split", tuple(self.split_list.keys()))
        self.url = self.split_list[split][0]
        self.filename = self.split_list[split][1]
        self.file_md5 = self.split_list[split][2]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        data = torch.load(os.path.join(self.root, self.filename))

        self.data = data[0]
        # loading gives an np array of type np.uint8
        # converting to np.int64, so that we have a LongTensor after
        # the conversion from the numpy array
        # the squeeze is needed to obtain a 1D tensor
        self.labels = data[1].type(torch.LongTensor)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.labels[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img.numpy(), (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        md5 = self.split_list[self.split][2]
        fpath = os.path.join(root, self.filename)
        return check_integrity(fpath, md5)

    def download(self):
        md5 = self.split_list[self.split][2]
        download_url(self.url, self.root, self.filename, md5)

    def extra_repr(self):
        return "Split: {split}".format(**self.__dict__)


class MultiDigitMNISTSplits(object):
    def __init__(
            self,
            dataset,
            num_digits=4,
            num_compare=None,
            num_train_list=1000,
            num_val_list=1000,
            num_test_list=1000,
            seed=0,
            deterministic_data_loader=True
        ):

        self.deterministic_data_loader = deterministic_data_loader

        if dataset == 'mnist':
            trva_real = datasets.MNIST(root='./data-mnist', download=True)
            xtr_real = trva_real.data[:55000].view(-1, 1, 28, 28)
            ytr_real = trva_real.targets[:55000]
            xva_real = trva_real.data[55000:].view(-1, 1, 28, 28)
            yva_real = trva_real.targets[55000:]

            te_real = datasets.MNIST(root='./data-mnist', train=False, download=True)
            xte_real = te_real.data.view(-1, 1, 28, 28)
            yte_real = te_real.targets

            self.train_dataset = MultiDigitMNISTDataset(
                images=xtr_real, labels=ytr_real,
                num_digits=num_digits, num_compare=num_compare,
                num_list=num_train_list, seed=seed,
                determinism=deterministic_data_loader)
            self.valid_dataset = MultiDigitMNISTDataset(
                images=xva_real, labels=yva_real, num_digits=num_digits, num_compare=num_compare, num_list=num_val_list, seed=seed)
            self.test_dataset = MultiDigitMNISTDataset(
                images=xte_real, labels=yte_real, num_digits=num_digits, num_compare=num_compare, num_list=num_test_list, seed=seed)

        elif dataset == 'svhn':
            self.train_dataset = SVHNMultiDigit(root='./data-svhn', split='train', download=True)
            self.valid_dataset = SVHNMultiDigit(root='./data-svhn', split='val', download=True)
            self.test_dataset = SVHNMultiDigit(root='./data-svhn', split='test', download=True)
        else:
            raise NotImplementedError()

    def get_train_loader(self, batch_size, **kwargs):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            num_workers=4 if not self.deterministic_data_loader else 0,
            shuffle=True, **kwargs
        )
        return train_loader

    def get_valid_loader(self, batch_size, **kwargs):
        valid_loader = DataLoader(self.valid_dataset,
                                  batch_size=batch_size, shuffle=False, **kwargs)
        return valid_loader

    def get_test_loader(self, batch_size, **kwargs):
        test_loader = DataLoader(self.test_dataset,
                                 batch_size=batch_size, shuffle=False, **kwargs)
        return test_loader
