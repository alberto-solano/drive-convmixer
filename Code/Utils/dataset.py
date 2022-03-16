import os
from torch.utils.data import Dataset
from PIL import Image, ImageOps
from torchvision.transforms import ToTensor, Compose, RandomAffine, RandomApply
import torchvision.transforms.functional as TF
import torch
from random import random
import numpy as np

class DRIVE_dataset (Dataset):
    def __init__(self, image_dir, mask_dir, rotation, hflip_prob,
                 brightness, contrast, gamma, affine_prob, affine_translate,
                 affine_scale, affine_shears, noise, transform):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.rotation = rotation
        self.hflip_prob = hflip_prob
        self.brightness = brightness
        self.contrast = contrast
        self.gamma = gamma
        self.affine_prob = affine_prob
        self.affine_translate = affine_translate
        self.affine_scale = affine_scale
        self.affine_shears = affine_shears
        self.noise = noise
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def transform_train(self, image, mask):

        degree = np.random.randint(self.rotation[0], self.rotation[1])
        brightness = np.random.uniform(self.brightness[0], self.brightness[1])
        gamma = np.random.uniform(self.gamma[0], self.gamma[1])
        contrast = np.random.uniform(self.contrast[0], self.contrast[1])
        # Random rotation
        image = TF.rotate(image, degree)
        mask = TF.rotate(mask, degree)
        # Random horizontal flipping
        if random() < self.hflip_prob:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random affine
        if random() < self.affine_prob:
            affine_param = RandomAffine.get_params(
                degrees=[0, 0], translate=self.affine_translate,
                img_size=[584, 565], scale_ranges=self.affine_scale,
                shears=self.affine_shears)
            image = TF.affine(image,
                              affine_param[0], affine_param[1],
                              affine_param[2], affine_param[3])
            mask = TF.affine(mask,
                             affine_param[0], affine_param[1],
                             affine_param[2], affine_param[3])

        # Se elige aleatoriamente una transformación dentro de un 30% de posibilidades:
        aleat = random()

        if (aleat > 0.7) & (aleat < 0.8):
            image = TF.adjust_brightness(image, brightness)
        elif (aleat > 0.8) & (aleat < 0.9):
            image = TF.adjust_gamma(image, gamma)
        elif aleat > 0.9:
            image = TF.adjust_contrast(image, contrast)

        noisy_tensor = Compose([
                        ToTensor(),
                        RandomApply([AddGaussianNoise(self.noise[0], self.noise[1]), ], p=0.3)])
        image = noisy_tensor(image)
        tens = ToTensor()
        mask = tens(mask)

        return image, mask

    def transform_test(self, image, mask):
        # Transform to tensor
        tens = ToTensor()
        image = tens(image)
        mask = tens(mask)
        return image, mask

    def __getitem__(self, index):

        number = self.images[index][0:2]
        image = f'{number}_training.tif'
        mask = f'{number}_manual1.gif'

        img_path = os.path.join(self.image_dir, image)
        mask_path = os.path.join(self.mask_dir, mask)

        img = Image.open(img_path).convert("RGB")

        mask = Image.open(mask_path).convert("RGB")

        # Fuerzo a 3 canales aunque solo sea uno para poder
        # aplicarle las mismas transformaciones

        if self.transform == "train":
            img, mask = self.transform_train(img, mask)
        if self.transform == "test":
            img, mask = self.transform_test(img, mask)

        # Devuelvo solo 1 canal del label
        return img, mask[0, :, :], number


class AddGaussianNoise(object):
    def __init__(self, mean, std):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return torch.clamp(tensor + torch.randn(tensor.size()) * self.std +
                           self.mean, min=0, max=1)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class DRIVE_dataset_padding (Dataset):
    def __init__(self, image_dir, mask_dir, rotation, hflip_prob,
                 brightness, contrast, gamma, affine_prob, affine_translate,
                 affine_scale, affine_shears, noise, transform):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.rotation = rotation
        self.hflip_prob = hflip_prob
        self.brightness = brightness
        self.contrast = contrast
        self.gamma = gamma
        self.affine_prob = affine_prob
        self.affine_translate = affine_translate
        self.affine_scale = affine_scale
        self.affine_shears = affine_shears
        self.noise = noise
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def transform_train(self, image, mask):

        degree = np.random.randint(self.rotation[0], self.rotation[1])
        brightness = np.random.uniform(self.brightness[0], self.brightness[1])
        gamma = np.random.uniform(self.gamma[0], self.gamma[1])
        contrast = np.random.uniform(self.contrast[0], self.contrast[1])

        # Padding
        image = TF.pad(image, (5, 0, 6, 0), padding_mode="constant", fill=0)
        mask = TF.pad(mask, (5, 0, 6, 0), padding_mode="constant", fill=0)
        # Cropping
        image = ImageOps.crop(image, (0, 4, 0, 4))
        mask = ImageOps.crop(image, (0, 4, 0, 4))
        # Random rotation
        image = TF.rotate(image, degree)
        mask = TF.rotate(mask, degree)
        # Random horizontal flipping
        if random() < self.hflip_prob:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random affine
        if random() < self.affine_prob:
            affine_param = RandomAffine.get_params(
                degrees=[0, 0], translate=self.affine_translate,
                img_size=[584, 565], scale_ranges=self.affine_scale,
                shears=self.affine_shears)
            image = TF.affine(image,
                              affine_param[0], affine_param[1],
                              affine_param[2], affine_param[3])
            mask = TF.affine(mask,
                             affine_param[0], affine_param[1],
                             affine_param[2], affine_param[3])

        # Se elige aleatoriamente una transformación dentro de un 30% de posibilidades:
        aleat = random()

        if (aleat > 0.7) & (aleat < 0.8):
            image = TF.adjust_brightness(image, brightness)
        elif (aleat > 0.8) & (aleat < 0.9):
            image = TF.adjust_gamma(image, gamma)
        elif aleat > 0.9:
            image = TF.adjust_contrast(image, contrast)

        noisy_tensor = Compose([
                        ToTensor(),
                        RandomApply([AddGaussianNoise(self.noise[0], self.noise[1]), ], p=0.3)])
        image = noisy_tensor(image)
        tens = ToTensor()
        mask = tens(mask)

        return image, mask

    def transform_test(self, image, mask):
        # Transform to tensor
        tens = ToTensor()
        image = tens(image)
        mask = tens(mask)
        return image, mask

    def __getitem__(self, index):

        number = self.images[index][0:2]
        image = f'{number}_training.tif'
        mask = f'{number}_manual1.gif'

        img_path = os.path.join(self.image_dir, image)
        mask_path = os.path.join(self.mask_dir, mask)

        img = Image.open(img_path).convert("RGB")

        mask = Image.open(mask_path).convert("RGB")

        # Fuerzo a 3 canales aunque solo sea uno para poder
        # aplicarle las mismas transformaciones

        if self.transform == "train":
            img, mask = self.transform_train(img, mask)
        if self.transform == "test":
            img, mask = self.transform_test(img, mask)

        # Devuelvo solo 1 canal del label
        return img, mask[0, :, :], number
