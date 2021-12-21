import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch
from random import random
import numpy as np

class DRIVE_dataset (Dataset):
    def __init__(self, image_dir, mask_dir, resize, rotation,
    hflip_prob, brightness, contrast, gamma, transform):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.resize = resize
        self.rotation = rotation
        self.hflip_prob = hflip_prob
        self.brightness = brightness
        self.contrast = contrast
        self.gamma = gamma
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)
    
    def transform_train(self, image, mask):
      # Resize
      if self.resize is not None:
        resize = transforms.Resize(size=self.resize)
        image = resize(image)
        mask = resize(mask)
      # Random rotation
      if self.rotation is not None:
        degree = np.random.randint(self.rotation[0], self.rotation[1])
        image = TF.rotate(image, degree)
        mask = TF.rotate(mask, degree)
      # Random horizontal flipping
      if self.hflip_prob is not None:
        if random() < self.hflip_prob:
          image = TF.hflip(image)
          mask = TF.hflip(mask)
      # Encadeno 3 transformaciones excluyentes entre si
      # Brightness
      if random() < 0.33333:
        image = TF.adjust_brightness(image, np.random.uniform(self.brightness[0], self.brightness[1]))
      # Gamma
      elif random() > 0.5:
        image = TF.adjust_gamma(image, np.random.uniform(self.gamma[0], self.gamma[1]))
      # Contrast
      else:
        image = TF.adjust_contrast(image, np.random.uniform(self.contrast[0], self.contrast[1]))
      # Transform to tensor
      tens = transforms.ToTensor()
      image = tens(image)
      mask = tens(mask)
        
      return image, mask
    
    def transform_test(self, image, mask):
      # Transform to tensor
      tens = transforms.ToTensor()
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
