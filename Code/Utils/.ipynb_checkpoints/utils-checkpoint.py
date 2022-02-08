import torch
from dataset import DRIVE_dataset
from torch.utils.data import DataLoader
import torchvision
from sklearn.metrics import roc_auc_score
import math
import pathlib
import warnings
from types import FunctionType
from typing import Any, Union, Optional, List, Tuple, BinaryIO

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    resize,
    rotation,
    hflip_prob,
    brightness,
    contrast,
    gamma,
    num_workers,
    pin_memory
):

    train_ds = DRIVE_dataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        resize = resize,
        transform="train",
        rotation = rotation,
        hflip_prob = hflip_prob,
        brightness = brightness,
        contrast = contrast,
        gamma = gamma
    )

    validation_ds = DRIVE_dataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        resize = resize,
        transform="test",
        rotation = rotation,
        hflip_prob = hflip_prob,
        brightness = brightness,
        contrast = contrast,
        gamma = gamma
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_loader = DataLoader(
        validation_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


def check_accuracy(loader, model, device):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    auc = 0
    model.eval()

    with torch.no_grad():
        for x, y, number in loader:
            x = x.to(device)
            y = y.to(device)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y.unsqueeze(1)).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y.unsqueeze(1)).sum()) / ((preds + y.unsqueeze(1)).sum() + 1e-8)
            auc += roc_auc_score(y.unsqueeze(1).view(y.numel()).cpu(), preds.view(preds.numel()).cpu())
    print(f"{num_correct} correct of {num_pixels} total pixels")
    print(f'Got {int(num_correct)/num_pixels*100}% accuracy')
    print(f'Dice score: {dice_score/len(loader)}')
    print(f'AUC: {auc/len(loader)}')
    model.train()

    return int(num_correct)/num_pixels, dice_score/len(loader), auc/len(loader)


def save_checkpoint(state, filename):
    print(" => Saving checkpoint ")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print(" => Loading checkpoint ")
    model.load_state_dict(checkpoint["state_dict"])


def train_fn(loader, model, optimizer, loss_fn, DEVICE):
    for batch_idx, (data, targets, numbers) in enumerate(loader):

        data = data.to(device=DEVICE)
        targets = targets.unsqueeze(1).to(device=DEVICE)

        # fwd
        optimizer.zero_grad()
        predictions = model(data)
        loss = loss_fn(predictions, targets)

        # backward
        loss.backward()
        optimizer.step()

        # Update tqdm loop
        # loop.set_postfix(loss=loss.item())
        print(f"Loss: {loss}")


def save_predictions_as_imgs(loader, model, folder, device):
    model.eval()
    for x, y, number in loader:
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{number}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{number}.png")
        
def make_grid(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    value_range: Optional[Tuple[int, int]] = None,
    scale_each: bool = False,
    pad_value: int = 0,
    **kwargs,
) -> torch.Tensor:
    """
    Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by ``value_range``. Default: ``False``.
        value_range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.

    Returns:
        grid (Tensor): the tensor containing grid of images.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(make_grid)
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f"tensor or list of tensors expected, got {type(tensor)}")

    if "range" in kwargs.keys():
        warning = "range will be deprecated, please use value_range instead."
        warnings.warn(warning)
        value_range = kwargs["range"]

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if value_range is not None:
            assert isinstance(
                value_range, tuple
            ), "value_range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, low, high):
            img.clamp_(min=low, max=high)
            img.sub_(low).div_(max(high - low, 1e-5))

        def norm_range(t, value_range):
            if value_range is not None:
                norm_ip(t, value_range[0], value_range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, value_range)
        else:
            norm_range(tensor, value_range)

    assert isinstance(tensor, torch.Tensor)
    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    num_channels = tensor.size(1)
    grid = tensor.new_full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            # Tensor.copy_() is a valid method but seems to be missing from the stubs
            # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.copy_
            grid.narrow(1, y * height + padding, height - padding).narrow(  # type: ignore[attr-defined]
                2, x * width + padding, width - padding
            ).copy_(tensor[k])
            k = k + 1
    return grid