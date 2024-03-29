import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import time
from monai.networks.nets import UNETR
from Code.Utils.utils import get_padded_loaders, save_checkpoint, train_fn, \
    save_predictions_as_imgs, eval_model, epoch_logging
import os

# Name for a run
TRAINING_FOLDER = "2023_03_23_1"

# Model Hyperparams
LR = 0.002
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
NUM_EPOCHS = 1000
NUM_WORKERS = 0
PIN_MEMORY = True
LOAD_CHECKPOINT = False
LOSS_WEIGHTS = 1
CONTROL_METRIC = "dice"

filepath = os.path.abspath(__file__)
proj_path = os.path.abspath(os.path.join(filepath, "..", "..", ".."))

if not os.path.exists(os.path.join(proj_path, "Code", "Checkpoints_UNETR", TRAINING_FOLDER)):
    os.makedirs(os.path.join(proj_path, "Code", "Checkpoints_UNETR", TRAINING_FOLDER))
    os.makedirs(os.path.join(proj_path, "Code", "Checkpoints_UNETR", TRAINING_FOLDER, "image_predictions"))

# Set the Training params
kwargs = {'train_dir': os.path.join(proj_path, 'Data/dataset_DRIVE/training/images/'),
          'train_maskdir': os.path.join(proj_path, 'Data/dataset_DRIVE/training/1st_manual/'),
          'val_dir': os.path.join(proj_path, 'Data/dataset_DRIVE/validation/images/'),
          'val_maskdir': os.path.join(proj_path, 'Data/dataset_DRIVE/validation/1st_manual/'),
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

model = UNETR(in_channels=3, out_channels=1, img_size=576, spatial_dims=2, feature_size=72).to(DEVICE)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([LOSS_WEIGHTS]).
                               to(DEVICE))  # Crossentropy loss
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# checkpoint loading
if LOAD_CHECKPOINT:
    state = torch.load(os.path.join(proj_path, f"Code/Checkpoints_UNETR/{TRAINING_FOLDER}/my_check.pth.tar"))
    model.load_state_dict(state["state_dict"])
    optimizer.load_state_dict(state["optimizer"])
    training_logs = torch.load(os.path.join(proj_path, f"Code/Checkpoints_UNETR/{TRAINING_FOLDER}/training_logs.pt"))

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
        save_checkpoint(checkpoint, os.path.join(proj_path, f"Code/Checkpoints_UNETR/{TRAINING_FOLDER}/my_check.pth.tar"),
                        best_metric=training_logs["best_"+CONTROL_METRIC])
        save_predictions_as_imgs(val_loader, model,
                                 folder=os.path.join(proj_path, f"Code/Checkpoints_UNETR/{TRAINING_FOLDER}/image_predictions"),
                                 device=DEVICE)

    training_logs["saved"].append(check)

    if not epoch % 20 or epoch == (NUM_EPOCHS - 1):
        torch.save(training_logs, os.path.join(proj_path, f"Code/Checkpoints_UNETR/{TRAINING_FOLDER}/training_logs.pt"))
 
    lr_scheduler.step(np.mean(train_results["loss"]))

params = {
          'LR': LR,
          'BATCH_SIZE': BATCH_SIZE,
          'NUM_EPOCHS': NUM_EPOCHS,
          'LOSS_WEIGHTS': LOSS_WEIGHTS,
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
torch.save(params, os.path.join(proj_path, f"Code/Checkpoints_UNETR/{TRAINING_FOLDER}/training_params.pt"))
