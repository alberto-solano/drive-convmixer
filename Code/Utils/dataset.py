import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch
from random import random
import numpy as np

class DRIVE_dataset (Dataset):
    def __init__(self, image_dir, mask_dir, resize, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.resize = resize

        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)
    
    def transform_train(self, image, mask):
        # Resize
        resize = transforms.Resize(size=self.resize)
        image = resize(image)
        mask = resize(mask)
        # Random rotation
        degree = np.random.randint(-10, 10)
        image = TF.rotate(image, degree)
        mask = TF.rotate(mask, degree)
        # Random horizontal flipping
        if random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        # Transform to tensor
        tens = transforms.ToTensor()
        image = tens(image)
        mask = tens(mask)
        # Normalize
        norm = transforms.Normalize(mean=(0.0, 0.0, 0.0),
                                    std=(1.0, 1.0, 1.0))
        image = norm(image)
        mask = norm(mask)
        return image, mask
    
    def transform_test(self, image, mask):
        # Transform to tensor
        tens = transforms.ToTensor()
        image = tens(image)
        mask = tens(mask)
        # Normalize
        norm = transforms.Normalize(mean=(0.0, 0.0, 0.0),
                                    std=(1.0, 1.0, 1.0))
        image = norm(image)
        mask = norm(mask)
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
