import torch
from dataset import DRIVE_dataset
from torch.utils.data import DataLoader


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    num_workers=1,
    pin_memory=True,
):

    train_ds = DRIVE_dataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        resize = (565,584),
        transform="train",
    )

    validation_ds = DRIVE_dataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        resize = (565,584),
        transform="test",
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    val_loader = DataLoader(
        validation_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader
