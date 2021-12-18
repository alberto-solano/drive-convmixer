import numpy as np
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
import torchvision.transforms as TF

from typing import Sequence, Tuple, List, Union, Callable
from tqdm import tqdm
from glob import glob
import os
#import warnings


def to_numpy_image(img: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Casts an image torch tensor to numpy array format. If numpy array is 
    passed, returns it directly. If the input is a normalized tensor in the 
    range [0.0, 1.0], it will be coerced to a RGB uint8.

    Args:
        img (Union[np.ndarray, torch.Tensor]): image in numpy array or torch
            tensor format.

    Raises:
        TypeError: if the input image is not in numpy or torch format.

    Returns:
        [type]: image as numpy array.
    """

    if not isinstance(img, (np.ndarray, torch.Tensor)):
        raise TypeError("img must be a numpy array or a torch tensor.")

    if isinstance(img, torch.Tensor):
        if img.dtype == torch.float:
            img = np.uint8(((img*255).permute(1, 2, 0)
                            if img.ndim == 3 else img).numpy())
        else:
            img = (img.permute(1, 2, 0) if img.ndim == 3 else img).numpy()

    return img


def imshow(img: Union[np.ndarray, torch.Tensor]) -> None:
    """Shows an image.

    Args:
        img (Union[np.ndarray, torch.Tensor]): Either an image in numpy array 
            or in torch tensor format (normalized or not).
    """
    img = to_numpy_image(img)
    cmap = "gray" if img.ndim == 2 else None

    plt.imshow(img, cmap=cmap)
    plt.show()


def load_data(input_data: Union[Sequence[np.ndarray], str],
              label: str = "data") -> List[np.ndarray]:
    """Loads segmentation data. If a folder path is given, 
    an image reading process is launched, otherwise a sequence of
    numpy arrays must be passed as input data.

    Args:
        input_data (Union[Sequence[np.ndarray], str]): folder path (string) or
            sequence (list or tuple) of numpy arrays.
        label (str): label describing the data.

    Raises:
        TypeError: if any element of input sequence is not a numpy array.
        ValueError: if the provided image folder is empty.
        TypeError: if the input is neither a sequence nor a string.

    Returns:
        list(np.array): list of images as numpy arrays.
    """

    if isinstance(input_data, (list, tuple)):
        if all(isinstance(i, np.ndarray) for i in input_data):
            return input_data
        else:
            raise TypeError(
                "Not all elements of the sequence are numpy arrays.")

    elif isinstance(input_data, str):
        paths = sorted(glob(os.path.join(input_data, "*")))
        print("--- {} reading process ---".format(label), "\n",
              "{} files were found in {}.".format(len(paths), input_data),
              flush=True)
        if len(paths) > 0:
            images = []
            errors = 0
            for p in tqdm(paths):
                try:
                    if os.path.splitext(p)[1] == ".gif":
                        images.append(np.array(Image.open(p)))
                    else:
                        images.append(cv.cvtColor(
                            cv.imread(p), cv.COLOR_BGR2RGB))
                except:
                    errors += 1
            print(
                "{} images loaded, {} errors.\n-------------\n".format(len(images), errors))
            return images

        else:
            raise ValueError("The folder is empty.")

    else:
        raise TypeError(
            "No numpy array sequence or path to the images folder was entered.")


class ComposeTransforms(TF.Compose):
    """Class for composition of transformations. The resulting object is
    a callable which concatenates the list transformations to perform them
    over a sample (image, label).

    """

    def __init__(self, morpho_transforms: List[Callable],
                 quality_transforms: List[Callable] = None) -> None:
        """

        Args:
            morpho_transforms (List[Callable]): list of morphological 
                transformations. Such transformations are performed both on 
                the image and on the label (rotations, warpings, translations, 
                etc). ToTensor torch transform must be passed as the first list item.
            quality_transforms (List[Callable]): list of quality 
                transformations. These transformations are performed only on 
                the image (change of brightness, saturation, color jitter, etc).
        """
        transforms = morpho_transforms + \
            quality_transforms if quality_transforms else morpho_transforms
        super(ComposeTransforms, self).__init__(transforms)

        self.morpho_transforms = morpho_transforms
        self.quality_transforms = quality_transforms

    def __call__(self, sample) -> Tuple[torch.Tensor]:
        """Perfroms the transformations on a sample.

        Args:
            sample ((np.ndarray, np.ndarray)): (image, label) sample in numpy array format.

        Returns:
            (torch.Tensor, torch.Tensor): transformed sample in torch tensor format.
        """

        x, y = sample
        # Concatenates the image and the label to perform the same
        # random transformation to both
        sample = np.concatenate((x, y[:, :, None]), axis=2)

        for t in self.morpho_transforms:
            sample = t(sample)
        x, y = sample[:3], sample[3]
        if self.quality_transforms:
            for t in self.quality_transforms:
                x = t(x)

        return x, y


class SegmentationDataset(Dataset):
    """Torch Dataset tailored to data from segmentation problems. An easy-to-use 
    data augmentation functionality is implemented (deactivated by default).

    """

    def __init__(self, x: Union[Sequence[np.array], str],
                 y: Union[Sequence[np.array], str],
                 masks: Union[Sequence[np.array], str] = None,
                 transforms: Callable = None) -> None:
        """

        Args:
            x (Union[Sequence[np.array], str]): folder path (string) or
                sequence (list or tuple) of numpy arrays for image data.
            y (Union[Sequence[np.array], str]): folder path (string) or
                sequence (list or tuple) of numpy arrays for label data.
            masks (Union[Sequence[np.array], str]): folder path (string) or
                sequence (list or tuple) of numpy arrays for mask data (if exists).
            transforms (Callable, optional): ComposeTransformations object 
                gathering the different transformations to be performed in the 
                data augmentation. Defaults to None.
        """

        super(SegmentationDataset, self).__init__()

        self.x = load_data(x, "x data")
        self.y = load_data(y, "y data")
        if masks:
            masks = [np.uint8(m/m.max())
                     for m in load_data(masks, "mask data")]
        self.masks = masks

        self.transforms = transforms
        # Data augmentation flag
        self.dag = False

        if len(self.x) != len(self.y):
            print("Caution, x and y data do not have the same length")

    def __len__(self) -> int:
        """Returns the dataset length.

        Returns:
            int: number of samples in the dataset.
        """
        return len(self.x)

    def __getitem__(self, idx: Union[int, slice, Sequence[int]]
                    ) -> Tuple[Tuple[np.ndarray, np.ndarray]]:
        """Gets a sample or a sequence of samples from the dataset. 
        If the data augmentation mode is activated, the samples are 
        transformed before they are returned.

        Args:
            idx (Union[int, slice, Sequence[int]]): index/indexes to extract 
                from the dataset as integer, list or slice.

        Raises:
            TypeError: if index has bad type.

        Returns:
            Tuple[Tuple[np.ndarray, np.ndarray]]: Sequence of returned samples, 
            which can be transformed (if dag mode is enabled) or not.
        """

        if torch.is_tensor(idx):
            idx = idx.tolist()

        if isinstance(idx, int):
            idx = (idx, )
        elif isinstance(idx, slice):
            start = idx.start if idx.start else 0
            stop = idx.stop if idx.stop else len(self)
            step = idx.step if idx.step else 1
            idx = range(start, stop, step)
        elif isinstance(idx, (list, tuple)):
            if all(isinstance(i, int) for i in idx):
                pass
        else:
            raise TypeError("Index must be an integer, slice or sequence.")

        t = self.transforms if (self.transforms and self.dag) else lambda x: x

        return tuple(t((self.x[i], self.y[i])) for i in idx)

    def apply_masks(self) -> None:
        """Applies masks (if any) to the image data.
        """
        if self.masks:
            self.x = [x*m[:, :, None] for x, m in zip(self.x, self.masks)]

    def set_dag_mode(self, flag: bool = True):
        """Enables or disables the data augmentation mode.

        Args:
            flag (bool, optional): True to activate the dag mode, False to deactivate it. 
                Defaults to True.
        """

        self.dag = flag

# Pruebas
#compose = ComposeTransforms([TF.ToTensor(), TF.RandomRotation(180)])#

# dataset = SegmentationDataset(
#    "../../Data/dataset DRIVE/training/images",
#    "../../Data/dataset DRIVE/training/1st_manual",
#    #"../TFM/Data/dataset DRIVE/training/mask",
#    transforms=compose)#

#s = dataset[:2][1]
# imshow(s[0])
#imshow(s[1])#

# dataset.set_dag_mode(True)
#s = dataset[:2][1]
# imshow(s[0])
# imshow(s[1])
