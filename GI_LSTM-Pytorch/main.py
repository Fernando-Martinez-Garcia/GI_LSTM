import torch
import torch.nn as nn
import json

from data_loader import get_dataloaders
from utils import TrainingVisualizer, evaluate_metrics, compute_validation_loss, set_seed
from arch import build_gilstm_model, HybridLoss
from trainer import Trainer

# --------------------- Load Configuration ---------------------
with open("parameters.json", "r", encoding='utf-8') as file:
    config = json.load(file)

# --------------------- Setup ---------------------
# Setup for reproducibility
set_seed(12345)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Experiment: {config.get('experiment_id', 'Unnamed')}")

# --------------------- Data pipeline ---------------------
data_cfg = config["data_config"]
train_cfg = config["training_config"]

train_loader, val_loader, test_loader, target_scaler, D, Cout, target_cols = get_dataloaders(
    file_path=data_cfg["file_path"],
    lookback=data_cfg["lookback"],
    horizon=data_cfg["horizon"],
    batch_size=train_cfg["batch_size"],
    output_channels=data_cfg["output_channels"],
    window_step=data_cfg.get("window_step", 1)
)

# --------------------- Model Build ---------------------
model = build_gilstm_model(
    config=config,
    D=D,
    Cout=Cout,
    target_cols=target_cols,
    device=device
)

crit = HybridLoss()
opt = torch.optim.AdamW(
    model.parameters(),
    lr=train_cfg["learning_rate"],
    weight_decay=0.0
)

# --------------------- Visualization setup---------------------
viz = TrainingVisualizer()
# A FIXED sample for visualization (e.g., from validation set)
viz.set_reference_sample(val_loader, device, sample_idx=1500)

# --------------------- Training ---------------------
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=opt,
    criterion=crit,
    device=device,
    config=config,
    visualizer=viz,
    target_scaler=target_scaler
)

trainer.train()

# --------------------- Final evaluation ---------------------
mse_tr, mae_tr = evaluate_metrics(model, train_loader, device, target_scaler)
mse_va, mae_va = evaluate_metrics(model, val_loader, device, target_scaler)
mse_te, mae_te = evaluate_metrics(model, test_loader, device, target_scaler)

print("\n== MemoryGroupLSTM seq(L)->vec(H) forecast ==")
print(f"Train | MSE {mse_tr:.6f} | MAE {mae_tr:.6f}")
print(f"Val   | MSE {mse_va:.6f} | MAE {mae_va:.6f}")
print(f"Test  | MSE {mse_te:.6f} | MAE {mae_te:.6f}")
