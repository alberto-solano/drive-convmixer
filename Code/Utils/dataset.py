import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import ToTensor, Compose, RandomCrop
import torchvision.transforms.functional as TF
import torch
from random import random
import numpy as np


class DRIVE_dataset (Dataset):
    def __init__(self, image_dir, mask_dir, rotation, hflip_prob,
                 brightness, contrast, gamma, crop_size, p_crop, noise,
                 transform):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.rotation = rotation
        self.hflip_prob = hflip_prob
        self.brightness = brightness
        self.contrast = contrast
        self.gamma = gamma
        self.crop_size = crop_size
        self.p_crop = p_crop
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
        i, j, h, w = RandomCrop.get_params(
            image, output_size=self.crop_size)
        # Random rotation
        image = TF.rotate(image, degree)
        mask = TF.rotate(mask, degree)
        # Random horizontal flipping
        if random() < self.hflip_prob:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        # Crop
        if random() < self.p_crop:
            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)
            up_down_pad = 584-self.crop_size[0]
            left_right_pad = 565-self.crop_size[1]
            # Hago padding de ceros repartiendo dichos ceros de manera aleatoria
            # entre los 4 lados de la imagen siendo la dimesión final igual
            vertical = np.random.randint(0, up_down_pad)
            horizontal = np.random.randint(0, up_down_pad)
            top = vertical
            bottom = up_down_pad - vertical
            left = horizontal
            right = left_right_pad - horizontal
            image = TF.pad(image, (left, top, right, bottom))
            mask = TF.pad(mask, (left, top, right, bottom))

        # Se elige aleatoriamente una transformación:
        aleat = random()
        if aleat < 0.3333:
            image = TF.adjust_brightness(image, brightness)
        elif aleat < 0.6666:
            image = TF.adjust_gamma(image, gamma)
        else:
            image = TF.adjust_contrast(image, contrast)

        noisy_tensor = Compose([
                        ToTensor(),
                        AddGaussianNoise(self.noise[0], self.noise[1])])
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
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return torch.clamp(tensor + torch.randn(tensor.size()) * self.std + self.mean, min=0, max=1)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
