import sys
sys.path.append("./Utils")
sys.path.append('./Models')
from utils import get_loaders, save_checkpoint, train_fn, check_accuracy, save_predictions_as_imgs, eval_model, epoch_logging
from U_net import UNET
import numpy as np
import os
import PIL
import torch
from tqdm import tqdm
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import time

# Model Hyperparams
LR = 5e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
NUM_EPOCHS = 2000
NUM_WORKERS = 0
PIN_MEMORY = True
LOAD_MODEL = False
LOSS_WEIGHTS = 10
CONTROL_METRIC = "dice"

# Training Loader params

TRAIN_IMG_DIR = "./Data/training/images/"
TRAIN_MASK_DIR = "./Data/training/1st_manual/"
VAL_IMG_DIR = "./Data/validation/images/"
VAL_MASK_DIR = "./Data/validation/1st_manual/"
RESIZE = None
ROTATION = [-180, 180]
HFLIP_PROB = 0.5
BRIGHTNESS = [0.8, 1.2]
CONTRAST = [0.8, 1.2]
GAMMA = [0.9, 1.1]

train_loader, val_loader = get_loaders(
     TRAIN_IMG_DIR,
     TRAIN_MASK_DIR,
     VAL_IMG_DIR,
     VAL_MASK_DIR,
     BATCH_SIZE,
     RESIZE,
     ROTATION,
     HFLIP_PROB,
     BRIGHTNESS,
     CONTRAST,
     GAMMA,
     NUM_WORKERS,
     PIN_MEMORY)

def metric_fn(logits, targets, thres = 0.5):
    probs = torch.sigmoid(logits)
    preds = probs > thres

    overlapping = preds + targets
    tp = (overlapping > 1.5).sum().item()
    tn = (overlapping < 0.5).sum().item()

    return {"dice": 2 * tp / overlapping.sum().item(), 
        "accuracy": (tp + tn) / overlapping.numel(), 
        "auc": roc_auc_score(targets, probs)}

# diccionario de metricas
training_logs = {
    "loss": [],
    "dice": [],
    "accuracy": [],
    "auc": [],
    "val_loss": [],
    "val_dice": [],
    "val_accuracy": [],
    "val_auc": [],
    "saved": [],
    "best_dice": 0,
    "best_accuracy": 0,
    "best_auc": 0
}

# Training

model = UNET(in_channels=3, out_channels=1).to(DEVICE)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([LOSS_WEIGHTS]).to(DEVICE))  # Crossentropy loss
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
# checkpoint loading
LOAD_CHECKPOINT = True
if LOAD_CHECKPOINT:
    state = torch.load("./Checkpoints/dev/my_check.pth.tar")
    model.load_state_dict(state["state_dict"])
    optimizer.load_state_dict(state["optimizer"])
    training_logs = torch.load("./Checkpoints/dev/training_logs.pt")
    
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor = 0.5, patience = 100, min_lr = 1e-5, verbose = True)


print('=======> Starting training....')
for epoch in range(NUM_EPOCHS):
    epoch_init = time.time()
    # train_fn devuelve los losses y posibles metricas en un dict de listas
    train_results = train_fn(train_loader, model, optimizer, loss_fn, DEVICE, metric_fn, epoch = epoch)

    training_logs["loss"].extend(train_results["loss"])
    training_logs["dice"].extend(train_results["dice"])
    training_logs["accuracy"].extend(train_results["accuracy"])
    training_logs["auc"].extend(train_results["auc"])

    # eval_model devuelve los losses y posibles metricas en un dict
    val_results = eval_model(val_loader, model, loss_fn, DEVICE, metric_fn)

    training_logs["val_loss"].append(val_results["loss"])
    training_logs["val_dice"].append(val_results["dice"])
    training_logs["val_accuracy"].append(val_results["accuracy"])
    training_logs["val_auc"].append(val_results["auc"])

    # imprime el log resumen de la epoch (medias para train)
    epoch_logging(epoch, np.mean(train_results["loss"]),
        val_results["loss"],  
        {k: np.mean(v) for k, v in train_results.items() if k != "loss"}, 
        {"val_"+k: v for k, v in val_results.items() if k != "loss"},
        time.time() - epoch_init
    )

    check = val_results[CONTROL_METRIC] >= training_logs["best_"+CONTROL_METRIC]
    if check:
        training_logs["best_dice"] = val_results["dice"]
        training_logs["best_accuracy"] = val_results["accuracy"]
        training_logs["best_auc"] = val_results["auc"]
        # Save model
        checkpoint = {"state_dict": model.state_dict(),
                      "optimizer": optimizer.state_dict()}
        save_checkpoint(checkpoint, "./Checkpoints/dev/my_check.pth.tar", best_metric = training_logs["best_"+CONTROL_METRIC])
        save_predictions_as_imgs(val_loader, model, folder="./Checkpoints/dev/image_predictions/", device=DEVICE)

    training_logs["saved"].append(check)

    if not epoch % 20 or epoch == (NUM_EPOCHS - 1):
        torch.save(training_logs, "./Checkpoints/dev/training_logs2.pt")

    lr_scheduler.step(np.mean(train_results["loss"]))

params = {
          'LR': LR,
          'BATCH_SIZE': BATCH_SIZE,
          'NUM_EPOCHS': NUM_EPOCHS,
          'LOSS_WEIGHTS': LOSS_WEIGHTS,
          'RESIZE': RESIZE,
          'ROTATION': ROTATION,
          'HFLIP_PROB': HFLIP_PROB,
          'BRIGHTNESS': BRIGHTNESS,
          'CONTRAST': CONTRAST,
          'GAMMA': GAMMA,
          'DICE': training_logs["best_dice"],
          'ACCURACY': training_logs["best_accuracy"],
          'AUC': training_logs["best_auc"]
}
torch.save(params, "./Checkpoints/dev/training_params.pt")