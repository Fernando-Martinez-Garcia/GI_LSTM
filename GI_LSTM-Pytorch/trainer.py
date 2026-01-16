import torch
import time
from utils_rfc_e import evaluate_metrics, compute_validation_loss
from arch_rfc_e import enforce_constraints


class Trainer:
    def __init__(
            self,
            model,
            train_loader,
            val_loader,
            test_loader,
            optimizer,
            criterion,
            device,
            config,
            visualizer=None,
            target_scaler=None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.viz = visualizer
        self.scaler = target_scaler

        train_cfg = config.get("training_config", config)

        # Config extraction
        self.epochs = train_cfg["epoch_number"]
        self.patience = train_cfg.get("patience", 5000)
        self.log_every = train_cfg.get("log_every", 10)
        self.max_grad_norm = 1.0

        # State
        self.best_val = float("inf")
        self.best_state = None
        self.counter = 0


    def train(self):
        print(f"Starting training for {self.epochs} epochs on {self.device}...")
        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            # Train Step
            loss_tr = self._train_epoch()

            # Validation Step
            loss_va = compute_validation_loss(self.model, self.val_loader, self.criterion, self.device)

            # Early Stopping Check
            if loss_va < self.best_val - 1e-6:
                self.best_val = loss_va
                self.best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

            # Logging & Visualization
            if epoch == 1 or epoch % self.log_every == 0 or epoch == self.epochs:
                self._log_progress(epoch, loss_tr, loss_va)

        # End of training: Restore best model
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
            print("Restored best model from early stopping.")

        total_time = time.time() - start_time
        print(f"Training complete in {total_time:.2f}s")

    def _train_epoch(self):
        self.model.train()
        total_loss = 0
        steps = 0

        for xb, yb in self.train_loader:
            # GI-LSTM specific constraints
            with torch.no_grad():
                enforce_constraints(self.model)

            xb = xb.to(self.device)
            yb = yb.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            pred = self.model(xb)
            loss = self.criterion(pred, yb)

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            total_loss += loss.item()
            steps += 1

        return total_loss / max(steps, 1)

    def _log_progress(self, epoch, loss_tr, loss_va):
        # Update Plots
        if self.viz:
            self.viz.update_forecast(self.model, epoch, self.device, self.scaler)

            if hasattr(self.model, 'core'):
                core_list = [self.model.core.encoder.multi, self.model.core.decoder_head.decoders]
                self.viz.plot_theta(core_list, epoch, layer_indices=[0], use_weighted=True)

        # Compute Metrics
        mse_tr, mae_tr = evaluate_metrics(self.model, self.train_loader, self.device, self.scaler)
        mse_va, mae_va = evaluate_metrics(self.model, self.val_loader, self.device, self.scaler)
        mse_te, mae_te = evaluate_metrics(self.model, self.test_loader, self.device, self.scaler)

        print(f"Epoch {epoch:6d} | Train Loss {loss_tr:.6f} | Validation Loss {loss_va:.6f} | "
            f"Train MSE {mse_tr:.6f} MAE {mae_tr:.6f} | "
            f"Validation   MSE {mse_va:.6f} MAE {mae_va:.6f} | "
            f"Testing  MSE {mse_te:.6f} MAE {mae_te:.6f}")