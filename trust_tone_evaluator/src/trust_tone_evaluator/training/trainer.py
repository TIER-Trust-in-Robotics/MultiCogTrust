"""Training loop for trust assessment models."""

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from trust_tone_evaluator.training.losses import TrustMultiTaskLoss

# Optional wandb import
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


logger = logging.getLogger(__name__)


class TrustTrainer:
    """Training loop with multi-task loss, LR scheduling, checkpointing, and early stopping."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: Optional[str] = None,
        metrics_fn: Optional[Callable] = None,
    ):
        # Auto-detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.metrics_fn = metrics_fn

        # Loss function
        self.criterion = TrustMultiTaskLoss(
            level_weight=config.get("level_weight", 1.5),
            score_weight=config.get("score_weight", 1.0),
            indicator_weight=config.get("indicator_weight", 0.5),
            use_ordinal_loss=config.get("use_ordinal_loss", False),
            use_focal_loss=config.get("use_focal_loss", False),
            label_smoothing=config.get("label_smoothing", 0.1),
            class_weights=config.get("class_weights", None),
        )

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.get("learning_rate", 1e-4),
            weight_decay=config.get("weight_decay", 0.01),
        )

        # Learning rate scheduler
        scheduler_type = config.get("scheduler", "plateau")
        if scheduler_type == "cosine":
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=config.get("T_0", 10),
                T_mult=config.get("T_mult", 2),
            )
        else:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=config.get("scheduler_patience", 5),
            )

        # Gradient clipping
        self.max_grad_norm = config.get("max_grad_norm", 1.0)

        # Checkpointing
        self.checkpoint_dir = Path(config.get("checkpoint_dir", "./checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Early stopping
        self.early_stopping_patience = config.get("early_stopping_patience", 20)
        self.early_stopping_counter = 0

        # Best metrics tracking
        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0
        self.best_epoch = 0

        # Logging
        self.use_wandb = config.get("use_wandb", False) and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(
                project=config.get("wandb_project", "trust-tone-evaluator"),
                name=config.get("experiment_name", None),
                config=config,
            )

        # Training history
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "learning_rate": [],
        }

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch. Returns metrics dict."""
        self.model.train()

        total_loss = 0.0
        total_level_loss = 0.0
        total_score_loss = 0.0
        total_indicator_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]", leave=False)

        for batch in pbar:
            # Move to device
            inputs = self._prepare_batch(batch)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs["features"])

            # Compute loss
            losses = self.criterion(outputs, inputs)
            loss = losses["total_loss"]

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm,
                )

            self.optimizer.step()

            # Track losses
            total_loss += loss.item()
            total_level_loss += losses["level_loss"].item()
            total_score_loss += losses["score_loss"].item()
            total_indicator_loss += losses["indicator_loss"].item()
            num_batches += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Average metrics
        metrics = {
            "train_loss": total_loss / num_batches,
            "train_level_loss": total_level_loss / num_batches,
            "train_score_loss": total_score_loss / num_batches,
            "train_indicator_loss": total_indicator_loss / num_batches,
        }

        return metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model. Returns metrics dict."""
        self.model.eval()

        total_loss = 0.0
        all_predictions = []
        all_targets = []
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Validation", leave=False):
            inputs = self._prepare_batch(batch)
            outputs = self.model(inputs["features"])

            # Compute loss
            losses = self.criterion(outputs, inputs)
            total_loss += losses["total_loss"].item()

            # Collect predictions for metrics
            all_predictions.append(
                {
                    "trust_logits": outputs["trust_logits"].cpu(),
                    "trust_score": outputs["trust_score"].cpu(),
                    "indicators": outputs["indicators"].cpu(),
                }
            )
            all_targets.append(
                {
                    "trust_level": inputs["trust_level"].cpu(),
                    "trust_score": inputs["trust_score"].cpu(),
                    "indicators": inputs["indicators"].cpu(),
                }
            )
            num_batches += 1

        # Compute metrics
        metrics = {"val_loss": total_loss / num_batches}

        # Concatenate all predictions
        pred_logits = torch.cat([p["trust_logits"] for p in all_predictions])
        pred_scores = torch.cat([p["trust_score"] for p in all_predictions])
        target_levels = torch.cat([t["trust_level"] for t in all_targets])
        target_scores = torch.cat([t["trust_score"] for t in all_targets])

        # Classification accuracy
        pred_levels = pred_logits.argmax(dim=-1)
        accuracy = (pred_levels == target_levels).float().mean().item()
        metrics["val_accuracy"] = accuracy

        # One-off accuracy (within 1 level)
        one_off = (torch.abs(pred_levels - target_levels) <= 1).float().mean().item()
        metrics["val_one_off_accuracy"] = one_off

        # Score MAE
        score_mae = torch.abs(pred_scores - target_scores).mean().item()
        metrics["val_score_mae"] = score_mae

        # Custom metrics function
        if self.metrics_fn is not None:
            custom_metrics = self.metrics_fn(all_predictions, all_targets)
            metrics.update(custom_metrics)

        return metrics

    def train(self, num_epochs: int) -> Dict[str, List[float]]:
        """Run the full training loop. Returns history dict."""
        logger.info(f"Starting training for {num_epochs} epochs on {self.device}")
        logger.info(f"Model parameters: {self.model.get_num_parameters():,}")

        start_time = time.time()

        for epoch in range(num_epochs):
            epoch_start = time.time()

            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate()

            # Update scheduler
            current_lr = self.optimizer.param_groups[0]["lr"]
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_metrics["val_loss"])
            else:
                self.scheduler.step()

            # Update history
            self.history["train_loss"].append(train_metrics["train_loss"])
            self.history["val_loss"].append(val_metrics["val_loss"])
            self.history["val_accuracy"].append(val_metrics["val_accuracy"])
            self.history["learning_rate"].append(current_lr)

            # Log metrics
            epoch_time = time.time() - epoch_start
            logger.info(
                f"Epoch {epoch}: "
                f"Train Loss={train_metrics['train_loss']:.4f}, "
                f"Val Loss={val_metrics['val_loss']:.4f}, "
                f"Val Acc={val_metrics['val_accuracy']:.4f}, "
                f"LR={current_lr:.6f}, "
                f"Time={epoch_time:.1f}s"
            )

            # Wandb logging
            if self.use_wandb:
                wandb.log(
                    {
                        **train_metrics,
                        **val_metrics,
                        "learning_rate": current_lr,
                        "epoch": epoch,
                    }
                )

            # Checkpointing
            if val_metrics["val_loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["val_loss"]
                self.best_epoch = epoch
                self.early_stopping_counter = 0
                self._save_checkpoint(epoch, "best_loss.pt", val_metrics)
            else:
                self.early_stopping_counter += 1

            if val_metrics["val_accuracy"] > self.best_val_acc:
                self.best_val_acc = val_metrics["val_accuracy"]
                self._save_checkpoint(epoch, "best_acc.pt", val_metrics)

            # Early stopping
            if self.early_stopping_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        total_time = time.time() - start_time
        logger.info(
            f"Training complete in {total_time/60:.1f} minutes. "
            f"Best val loss: {self.best_val_loss:.4f} at epoch {self.best_epoch}"
        )

        # Save final model
        self._save_checkpoint(epoch, "final.pt", val_metrics)

        return self.history

    def _prepare_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move batch tensors to device."""
        prepared = {
            "features": batch["features"].to(self.device),
            "trust_level": batch["trust_level"].to(self.device),
            "trust_score": batch["trust_score"].to(self.device),
            "indicators": batch["indicators"].to(self.device),
        }
        if "emotion_idx" in batch:
            prepared["emotion_idx"] = batch["emotion_idx"].to(self.device)
        return prepared

    def _save_checkpoint(
        self,
        epoch: int,
        filename: str,
        metrics: Dict[str, float],
    ) -> None:
        """Save model checkpoint."""
        # Capture model construction args for standalone loading
        model_config = getattr(self.model, "config", {})
        model_config["input_dim"] = getattr(self.model, "input_dim", 88)

        # Capture normalization stats from training dataset
        norm_stats = {}
        train_ds = self.train_loader.dataset
        if hasattr(train_ds, "_feat_mean") and hasattr(train_ds, "_feat_std"):
            norm_stats["feat_mean"] = train_ds._feat_mean.cpu()
            norm_stats["feat_std"] = train_ds._feat_std.cpu()

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "model_config": model_config,
            "norm_stats": norm_stats,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "best_val_acc": self.best_val_acc,
            "metrics": metrics,
            "config": self.config,
        }
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        logger.debug(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load model from checkpoint. Returns checkpoint dict."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.best_val_acc = checkpoint.get("best_val_acc", 0.0)

        logger.info(f"Loaded checkpoint from {path} (epoch {checkpoint['epoch']})")
        return checkpoint


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    learning_rate: float = 1e-4,
    checkpoint_dir: str = "./checkpoints",
    device: Optional[str] = None,
    **config_kwargs,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """Convenience wrapper around TrustTrainer. Returns (model, history)."""
    config = {
        "learning_rate": learning_rate,
        "checkpoint_dir": checkpoint_dir,
        "num_epochs": num_epochs,
        **config_kwargs,
    }

    trainer = TrustTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )

    history = trainer.train(num_epochs)

    # Load best model
    best_path = Path(checkpoint_dir) / "best_loss.pt"
    if best_path.exists():
        trainer.load_checkpoint(str(best_path))

    return trainer.model, history
