"""
Training engine — updated for pipeline spec.

Loss: 0.4×MSE(returns) + 0.4×MAE(volatility) + 0.2×CrossEntropy(direction)
Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
Scheduler: Cosine annealing
Validation: tracked on portfolio-proxy (Sharpe surrogate via direction accuracy)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Dict
import math


# ---------------------------------------------------------------------------
# Multi-task loss
# ---------------------------------------------------------------------------

def multi_task_loss(
    dir_logits, ret_pred, vol_pred,
    dir_target, ret_target, vol_target,
    w_ret=0.4, w_vol=0.4, w_dir=0.2,
    class_weights=None,
):
    """
    0.4×MSE(returns) + 0.4×MAE(volatility) + 0.2×CE(direction)
    """
    ce = nn.CrossEntropyLoss(weight=class_weights)
    mse = nn.MSELoss()
    mae = nn.L1Loss()

    loss_ret = mse(ret_pred.squeeze(), ret_target)
    loss_vol = mae(vol_pred.squeeze(), vol_target)
    loss_dir = ce(dir_logits, dir_target)

    total = w_ret * loss_ret + w_vol * loss_vol + w_dir * loss_dir
    return total, {
        "ret": loss_ret.item(),
        "vol": loss_vol.item(),
        "dir": loss_dir.item(),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_model(
    model: nn.Module,
    train_X, train_y,
    val_X, val_y,
    cfg,
    adj=None, edge_weight=None,
    is_gat: bool = True,
    device: str = "cpu",
) -> Dict:
    model = model.to(device)

    # AdamW per spec
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )

    # Cosine annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)

    # Class weights for 3-class direction
    dir_counts = np.bincount(train_y["direction"], minlength=3).astype(float)
    dir_counts = np.maximum(dir_counts, 1)
    cw = 1.0 / dir_counts
    cw = cw / cw.sum() * 3  # normalise so they sum to num_classes
    class_weights = torch.tensor(cw, dtype=torch.float).to(device)

    # DataLoaders
    train_ds = TensorDataset(
        torch.tensor(train_X, dtype=torch.float),
        torch.tensor(train_y["direction"], dtype=torch.long),
        torch.tensor(train_y["return"], dtype=torch.float),
        torch.tensor(train_y["volatility"], dtype=torch.float),
    )
    val_ds = TensorDataset(
        torch.tensor(val_X, dtype=torch.float),
        torch.tensor(val_y["direction"], dtype=torch.long),
        torch.tensor(val_y["return"], dtype=torch.float),
        torch.tensor(val_y["volatility"], dtype=torch.float),
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False)

    if adj is not None:
        adj = adj.to(device)

    w_ret = cfg.train.loss_weight_return
    w_vol = cfg.train.loss_weight_volatility
    w_dir = cfg.train.loss_weight_direction

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    name = "GAT-LSTM" if is_gat else "Baseline LSTM"
    print(f"\n{'='*60}")
    print(f"{name} Training  |  {cfg.train.epochs} epochs  |  AdamW lr={cfg.train.learning_rate}")
    print(f"{'='*60}")

    for epoch in range(1, cfg.train.epochs + 1):
        # ── Train ──
        model.train()
        epoch_loss, correct, total = 0, 0, 0

        for batch_x, batch_dir, batch_ret, batch_vol in train_loader:
            batch_x = batch_x.to(device)
            batch_dir = batch_dir.to(device)
            batch_ret = batch_ret.to(device)
            batch_vol = batch_vol.to(device)

            optimizer.zero_grad()

            if is_gat:
                dir_logits, ret_pred, vol_pred, _ = model(batch_x, adj, edge_weight)
            else:
                dir_logits, ret_pred, vol_pred = model(batch_x)

            loss, _ = multi_task_loss(
                dir_logits, ret_pred, vol_pred,
                batch_dir, batch_ret, batch_vol,
                w_ret, w_vol, w_dir, class_weights,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item() * len(batch_x)
            preds = dir_logits.argmax(dim=1)
            correct += (preds == batch_dir).sum().item()
            total += len(batch_dir)

        scheduler.step()
        avg_train = epoch_loss / total
        train_acc = correct / total

        # ── Validate ──
        model.eval()
        val_loss_sum, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for batch_x, batch_dir, batch_ret, batch_vol in val_loader:
                batch_x = batch_x.to(device)
                batch_dir = batch_dir.to(device)
                batch_ret = batch_ret.to(device)
                batch_vol = batch_vol.to(device)

                if is_gat:
                    dir_logits, ret_pred, vol_pred, _ = model(batch_x, adj, edge_weight)
                else:
                    dir_logits, ret_pred, vol_pred = model(batch_x)

                loss, _ = multi_task_loss(
                    dir_logits, ret_pred, vol_pred,
                    batch_dir, batch_ret, batch_vol,
                    w_ret, w_vol, w_dir, class_weights,
                )
                val_loss_sum += loss.item() * len(batch_x)
                preds = dir_logits.argmax(dim=1)
                val_correct += (preds == batch_dir).sum().item()
                val_total += len(batch_dir)

        avg_val = val_loss_sum / val_total
        val_acc = val_correct / val_total

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Early stopping
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            marker = " ★"
        else:
            patience_counter += 1
            marker = ""

        if epoch % 5 == 0 or epoch == 1 or marker:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:3d}  |  Train: {avg_train:.4f} ({train_acc:.1%})  "
                  f"|  Val: {avg_val:.4f} ({val_acc:.1%})  |  LR: {lr:.6f}{marker}")

        if patience_counter >= cfg.train.patience:
            print(f"  ⏹  Early stopping at epoch {epoch}")
            break

    if best_state:
        model.load_state_dict(best_state)
        model = model.to(device)

    best_epoch = history["val_loss"].index(min(history["val_loss"])) + 1
    print(f"\n✅ Best epoch: {best_epoch}  |  Val loss: {best_val_loss:.4f}  "
          f"|  Val acc: {history['val_acc'][best_epoch-1]:.1%}")

    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }
