import torch
from dataset import DRIVE_dataset
from torch.utils.data import DataLoader

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

def check_accuracy(loader, model, device):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
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
    print(f"{num_correct} correct of {num_pixels} total pixels")
    print(f'Got {int(num_correct)/num_pixels*100}% accuracy')
    print(f'Dice score: {dice_score/len(loader)}')
    model.train()

    return int(num_correct)/num_pixels, dice_score
    
def save_checkpoint(state, filename):
    print(" => Saving checkpoint ")
    torch.save(state, filename)
    
def load_checkpoint(checkpoint, model):
    print(" => Loading checkpoint ")
    model.load_state_dict(checkpoint["state_dict"])

def train_fn (loader, model, optimizer, loss_fn, DEVICE):
    
    for batch_idx, (data, targets, numbers) in enumerate(loader):
        
        data = data.to(device = DEVICE)
        targets = targets.unsqueeze(1).to(device = DEVICE)
        
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