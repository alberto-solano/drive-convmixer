import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import time
import sys
sys.path.append("../Utils")
from utils import get_padded_loaders, save_checkpoint, train_fn, \
    save_predictions_as_imgs, eval_model, epoch_logging, TverskyLoss, DiceBCELoss
sys.path.append("../Models")
from ConvMixer import ConvMixer
import os

TRAINING_FOLDER = "2022_04_27_1"
if not os.path.exists(os.path.join("..", "Checkpoints_ConvMixer", TRAINING_FOLDER)):
    os.makedirs(os.path.join("..", "Checkpoints_ConvMixer", TRAINING_FOLDER))
    os.makedirs(os.path.join("..", "Checkpoints_ConvMixer", TRAINING_FOLDER, "image_predictions"))

# Model Hyperparams
LR = 0.002
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
NUM_EPOCHS = 1000
NUM_WORKERS = 0
PIN_MEMORY = True
LOAD_MODEL = False
LOSS_WEIGHTS = 3
CONTROL_METRIC = "dice"
RATIO = 0.5 # The higher the more weight to BCE
LOSS_METRIC = f"DICE+BCE_weights={LOSS_WEIGHTS}_ratio={RATIO}"
DEPTH = 32
KERNEL = 7
PATCH_SIZE = 18
H = 768

# Training Loader params

kwargs = {'train_dir': '../../Data/dataset_DRIVE/training/images/',
          'train_maskdir': '../../Data/dataset_DRIVE/training/1st_manual/',
          'val_dir': '../../Data/dataset_DRIVE/validation/images/',
          'val_maskdir': "../../Data/dataset_DRIVE/validation/1st_manual/",
          'batch_size': BATCH_SIZE,
          'rotation': [0, 1],
          'hflip_prob': 0.4,
          'brightness': [0.8, 1.2],
          'contrast': [0.8, 1.2],
          'gamma': [0.9, 1.1],
          'affine_prob': 0.3,
          'affine_translate': [0.0, 0.1],  # Horiz and vert translation
          'affine_scale': [1, 1.2],
          'affine_shears': [0, 0],
          'noise': (0, 0.1),  # (Mean,std)
          'num_workers': 0,
          'pin_memory': True}

train_loader, val_loader = get_padded_loaders(**kwargs)


def metric_fn(logits, targets, thres=0.5):
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

model = ConvMixer(h=H, depth=DEPTH, kernel_size=KERNEL, patch_size=PATCH_SIZE).to(DEVICE)

if LOSS_METRIC == "BCE_logits":
  loss_fn = nn.BCEWithLogitsLoss(weight=torch.Tensor([LOSS_WEIGHTS]).to(DEVICE))
elif LOSS_METRIC == "Tversky":
  loss_fn = TverskyLoss(**tversky).to(DEVICE)
else:
  loss_fn = DiceBCELoss(weight=torch.Tensor([LOSS_WEIGHTS]).
                               to(DEVICE), ratio=RATIO) # Crossentropy loss
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
# checkpoint loading
LOAD_CHECKPOINT = False
if LOAD_CHECKPOINT:
    state = torch.load("../Checkpoints_ConvMixer/{}/my_check.pth.tar".format(TRAINING_FOLDER))
    model.load_state_dict(state["state_dict"])
    optimizer.load_state_dict(state["optimizer"])
    training_logs = torch.load("../Checkpoints_ConvMixer/{}/training_logs.pt".format(TRAINING_FOLDER))

lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min",
                                                          factor=0.5,
                                                          patience=100,
                                                          min_lr=1e-5,
                                                          verbose=True)


print('=======> Starting training....')
for epoch in range(NUM_EPOCHS):
    epoch_init = time.time()
    # train_fn devuelve los losses y posibles metricas en un dict de listas
    train_results = train_fn(train_loader, model, optimizer, loss_fn, DEVICE,
                             metric_fn, epoch=epoch)

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
                  {k: np.mean(v) for k, v in train_results.items()
                   if k != "loss"},
                  {"val_"+k: v for k, v in val_results.items() if k != "loss"},
                  time.time() - epoch_init)

    check = val_results[CONTROL_METRIC] >= \
        training_logs["best_"+CONTROL_METRIC]
    if check:
        training_logs["best_dice"] = val_results["dice"]
        training_logs["best_accuracy"] = val_results["accuracy"]
        training_logs["best_auc"] = val_results["auc"]
        # Save model
        checkpoint = {"state_dict": model.state_dict(),
                      "optimizer": optimizer.state_dict(),
                      "best_metric": training_logs["best_"+CONTROL_METRIC]}
        save_checkpoint(checkpoint, "../Checkpoints_ConvMixer/{}/my_check.pth.tar".format(TRAINING_FOLDER), best_metric = training_logs["best_"+CONTROL_METRIC])
        save_predictions_as_imgs(val_loader, model,
                                 folder="../Checkpoints_ConvMixer/{}/image_predictions".format(TRAINING_FOLDER),
                                 device=DEVICE)

    training_logs["saved"].append(check)

    if not epoch % 20 or epoch == (NUM_EPOCHS - 1):
        torch.save(training_logs, "../Checkpoints_ConvMixer/{}/training_logs.pt".format(TRAINING_FOLDER))
 
    lr_scheduler.step(np.mean(train_results["loss"]))

params = {
          'LR': LR,
          'BATCH_SIZE': BATCH_SIZE,
          'NUM_EPOCHS': NUM_EPOCHS,
          'LOSS_WEIGHTS': LOSS_WEIGHTS,
          'LOSS_FUNC': LOSS_METRIC,
          'DEPTH': DEPTH,
          'KERNEL': KERNEL,
          'PATCH_SIZE': PATCH_SIZE,
          'H': H,
          'ROTATION': kwargs['rotation'],
          'HFLIP_PROB': kwargs['hflip_prob'],
          'BRIGHTNESS': kwargs['brightness'],
          'CONTRAST': kwargs['contrast'],
          'GAMMA': kwargs['gamma'],
          'AFFINE_PROB': kwargs['affine_prob'],
          'AFFINE_TRANSLATE': kwargs['affine_translate'],
          'AFFINE_SCALE': kwargs['affine_scale'],
          'AFFINE_SHEARS': kwargs['affine_shears'],
          'NOISE': kwargs['noise'],
          'DICE': training_logs["best_dice"],
          'ACCURACY': training_logs["best_accuracy"],
          'AUC': training_logs["best_auc"]
}
torch.save(params, "../Checkpoints_ConvMixer/{}/training_params.pt".format(TRAINING_FOLDER))
